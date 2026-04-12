import sys
import os
import numpy as np
from scipy import sparse
from dem_to_matrices import detector_error_model_to_check_matrices
from IBM_STIM import create_bivariate_bicycle_codes, build_circuit, select_configuration


def actualizar_main_header(filename, max_row_degree, max_col_degree, vnodes, check, nnz, num_samples, val_max):
    header_content = f"""
#include <ap_fixed.h>
#include <climits>
#include <cstdint>
#include <float.h>
#include <ap_float.h>
#include <math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#ifndef VNODES
#define VNODES {vnodes}
#endif
#ifndef CHECK
#define CHECK  {check}
#endif
#ifndef NNZ
#define NNZ {nnz}
#endif
#ifndef MAX_ROW_LENGTH
#define MAX_ROW_LENGTH {max_row_degree}
#endif
#ifndef MAX_COL_LENGTH
#define MAX_COL_LENGTH {max_col_degree}
#endif
#ifndef NUM_IT
#define NUM_IT 100
#endif
#ifndef NUM_SYNDROMES
#define NUM_SYNDROMES {num_samples}
#endif
#ifndef BUS_WIDTH
#define BUS_WIDTH 512
#endif
#ifndef NUM_WORDS
#define NUM_WORDS (VNODES + BUS_WIDTH - 1)/BUS_WIDTH
#endif

typedef ap_fixed<12,9, AP_RND, AP_SAT> value_t;
typedef ap_uint<1> bit_t;
typedef ap_uint<512> bit_512_t;
typedef hls::vector<bit_512_t, NUM_WORDS> bus_512_t;

const value_t VAL_MAX = {val_max};

void min_sum_opt(bit_t* syndrome, value_t Lj, value_t alpha, bit_t *error_computed);
void min_sum_opt_csr( bus_512_t *syndrome, value_t Lj, value_t alpha, bus_512_t *error_computed);
"""
    with open(filename, 'w') as f:
        f.write(header_content)
    print(f"Header principal actualizado: {filename}")


def compute_extra(max_col, offset_cols, nnz, n):
    extra = np.arange(nnz, dtype=int)
    for j in range(n):
        actual_degree = offset_cols[j+1] - offset_cols[j]
        lastk = offset_cols[j+1] # En tu C: lastk = k despues del loop, luego lastk++
        
        if lastk < nnz:
            for i_pad in range(actual_degree, max_col):
                # Incrementa el actual y todos los posteriores
                extra[lastk:] += 1
    return extra


def generar_archivos_raw_h(pcm, config_name, VAL_MAX):
    pcm_csr = sparse.csr_matrix(pcm)
    pcm_csc = sparse.csc_matrix(pcm)
    
    m, n = pcm.shape

#----------------VARIABLES PARA OUTPUT-------------------------------------------------------------
    row_degrees = np.diff(pcm_csr.indptr)
    col_degrees = np.diff(pcm_csc.indptr)
    
    max_row_degree = int(np.max(row_degrees))
    max_col_degree = int(np.max(col_degrees))

    L_matrix = np.zeros((m, max_row_degree), dtype=int)
    offset_rows = pcm_csr.indptr
    offset_cols = pcm_csc.indptr

    col_index = pcm_csr.indices
    #initialize L_matrix padding with VAL_MAX
    for i in range(m):
        for j in range(max_row_degree):
            if j < row_degrees[i]:
                L_matrix[i, j] = 0
            else:
                L_matrix[i, j] = VAL_MAX

    # Matrices con padding de ceros
    edges = np.zeros(pcm.nnz, dtype=int)
    col_counts = np.zeros(n, dtype=int)
    for i in range(m):
        for k in range(offset_rows[i], offset_rows[i+1]):
            col = col_index[k]
            pos = offset_cols[col] + col_counts[col]
            edges[pos] = k
            col_counts[col] += 1
        
    extra = compute_extra(max_col_degree, offset_cols, pcm.nnz, n)
    

    indices_aux = np.zeros(pcm.nnz, dtype=int)
    for i_aux in range(pcm.nnz):
        indices_aux[edges[i_aux]] = i_aux

    for i in range(m):
        aux = 0
        lastk = 0
        for k in range(offset_rows[i], offset_rows[i+1]):
            aux += 1
            lastk = k
        lastk += 1 # lastk++
        
        if lastk < pcm.nnz:
            # For j = aux to max_row
            for _ in range(aux, max_row_degree):
                # edges[indices_aux[lastk]]++
                edges[indices_aux[lastk]] += 1
                # for m = lastk + 1 to nnz: edges[indices_aux[m]]++
                for m_idx in range(lastk + 1, pcm.nnz):
                    edges[indices_aux[m_idx]] += 1
    
    
    

    
#----------------VARIABLES PARA OUTPUT-------------------------------------------------------------

    # Función auxiliar para escribir arrays aplanados con coma al final
    def write_raw_h(filename, data_array):
        with open(filename, 'w') as f:
            f.write(", ".join(map(str, data_array.flatten())))
            f.write(", ") # Añade la coma y espacio final requerida
        print(f"Archivo raw guardado: {filename}")

    write_raw_h(f"edges.h", edges)
    write_raw_h(f"row_degrees.h", row_degrees)
    write_raw_h(f"col_degrees.h", col_degrees)
    write_raw_h(f"L_matrix.h", L_matrix)
    write_raw_h(f"offset_rows.h", offset_rows)
    write_raw_h(f"offset_cols.h", offset_cols)
    write_raw_h(f"extra.h", extra)
    write_raw_h(f"col_index.h", col_index)

    return max_row_degree, max_col_degree


def guardar_sindrome_custom(filename, pcm, syndrome):
    m, n = pcm.shape
    with open(filename, 'w') as f:
        
        for i in range(syndrome.shape[0]):
            # Síndrome separado por espacios
            syndrome_str = " ".join(map(str, np.array(syndrome[i]).flatten().astype(int)))
            f.write(f"{syndrome_str}\n")
        
    print(f"Archivo de testbench guardado: {filename}")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main_export():

#---------------PARAMETROS DE CONFIGURACIÓN-------------------------------------------------------------
    codeConfig = "90" 
    VAL_MAX = 200.0
    p = 0.005 
    num_samples = 100 # numero de sindromes que se van a probar en el testbench
    
#---------------PARAMETROS DE CONFIGURACIÓN-------------------------------------------------------------
    print(f"Generando datos para configuración: {codeConfig} con p={p} numero de muestras: {num_samples}")

    # 1. Selección y construcción del código
    config = select_configuration(codeConfig)
    ell, m_param = config["ell"], config["m"]
    a1, a2, a3 = config["a"]
    b1, b2, b3 = config["b"]
    A_x_pows, A_y_pows = [a1], [a2, a3]
    B_x_pows, B_y_pows = [b2, b3], [b1]

    code, A_list, B_list = create_bivariate_bicycle_codes(ell, m_param, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
    circuit = build_circuit(code, A_list, B_list, p=p, num_repeat=1, z_basis=False, use_both=False)
    dem = circuit.detector_error_model()
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    
    pcm = matrices.check_matrix
    print(f"Dimensiones de PCM: {pcm.shape}")

    # 2. Generar una muestra (Síndrome)
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(num_samples, separate_observables=True)
    syndrome = detectors

    # 3. Ejecutar las nuevas exportaciones
    # Generar raw .h y obtener los degrees máximos
    max_row, max_col = generar_archivos_raw_h(pcm, codeConfig, VAL_MAX)
    
    # Actualizar el main header con los nuevos defines
    actualizar_main_header("krnl_min_sum_ncsc.h", max_row, max_col,pcm.shape[1],pcm.shape[0], pcm.nnz, num_samples, VAL_MAX)
    
    # Generar el TXT en el nuevo formato
    guardar_sindrome_custom(f"input_syndrome.txt", pcm, syndrome)

if __name__ == "__main__":
    main_export()