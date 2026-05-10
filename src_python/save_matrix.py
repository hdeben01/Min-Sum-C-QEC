import sys
import os
import numpy as np
from scipy import sparse
from dem_to_matrices import detector_error_model_to_check_matrices
from IBM_STIM import create_bivariate_bicycle_codes, build_circuit, select_configuration

# --------------------------------------------------------------------------
# 1. Función para actualizar tu archivo .h principal
# --------------------------------------------------------------------------
def actualizar_main_header(filename, max_row_degree, max_col_degree, vnodes, check):
    header_content = f"""#ifndef VNODES
#define VNODES {vnodes}
#endif
#ifndef CHECK 
#define CHECK  {check}
#endif

#ifndef MAX_ROW_DEGREE
#define MAX_ROW_DEGREE {max_row_degree}
#endif
#ifndef MAX_COL_DEGREE
#define MAX_COL_DEGREE {max_col_degree}
#endif

#include <ap_fixed.h>
#include <climits>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <hls_vector.h>

typedef ap_fixed<12,6> values_typeData; //ap_fixed<12,6>
typedef ap_uint<1> uint1_t; // 1 bit

void compute_row_operations(values_typeData L[CHECK][VNODES],  
                            const uint1_t non_zero[CHECK][VNODES], 
                            uint1_t syndrome[CHECK], 
                            int size_checks, int size_vnode);

void compute_col_operations(values_typeData L[CHECK][VNODES],const uint1_t non_zero[CHECK][VNODES], uint1_t syndrome[CHECK], 
    int size_checks, int size_vnode, double alpha, 
    values_typeData Lj[VNODES], values_typeData sum_out[VNODES]);

void min_sum(uint1_t syndrome[CHECK], uint1_t resulting_syndrome[CHECK], int size_checks, int size_vnode, double alpha);
void min_sum_dep(uint1_t syndrome[CHECK], uint1_t resulting_syndrome[CHECK], int size_checks, int size_vnode, double alpha);
"""
    with open(filename, 'w') as f:
        f.write(header_content)
    print(f"Header principal actualizado: {filename}")

# --------------------------------------------------------------------------
# 2. Función para generar los .h con datos crudos (solo números y comas)
# --------------------------------------------------------------------------
def generar_archivos_raw_h(pcm, config_name):
    pcm_csr = sparse.csr_matrix(pcm)
    pcm_csc = sparse.csc_matrix(pcm)
    
    m, n = pcm.shape
    row_weights = np.diff(pcm_csr.indptr)
    col_weights = np.diff(pcm_csc.indptr)
    
    max_row_degree = int(np.max(row_weights))
    max_col_degree = int(np.max(col_weights))
    
    # Matrices con padding de ceros
    row_edges = np.zeros((m, max_row_degree), dtype=int)
    for i in range(m):
        edges = pcm_csr.indices[pcm_csr.indptr[i]:pcm_csr.indptr[i+1]]
        row_edges[i, :len(edges)] = edges
        
    col_edges = np.zeros((n, max_col_degree), dtype=int)
    for j in range(n):
        edges = pcm_csc.indices[pcm_csc.indptr[j]:pcm_csc.indptr[j+1]]
        col_edges[j, :len(edges)] = edges

    # Función auxiliar para escribir arrays aplanados con coma al final
    def write_raw_h(filename, data_array):
        with open(filename, 'w') as f:
            f.write(" ".join(map(str, data_array.flatten())))
            f.write(", ") # Añade la coma y espacio final requerida
        print(f"Archivo raw guardado: {filename}")

    write_raw_h(f"row_edges_{config_name}.h", row_edges)
    write_raw_h(f"col_edges_{config_name}.h", col_edges)
    write_raw_h(f"row_weight_{config_name}.h", row_weights)
    write_raw_h(f"col_weight_{config_name}.h", col_weights)
    write_raw_h(f"datos_exportados_{config_name}.h", pcm.toarray())

    return max_row_degree, max_col_degree

# --------------------------------------------------------------------------
# 3. Función para guardar el TXT del Testbench en el nuevo formato
# --------------------------------------------------------------------------
def guardar_sindrome_custom(filename, pcm, syndrome):
    m, n = pcm.shape
    with open(filename, 'w') as f:
        f.write("1.0\n")          # Valor fijo cabecera
        f.write(f"{m} {n}\n")     # Dimensiones
        
        # Síndrome separado por espacios
        syndrome_str = " ".join(map(str, np.array(syndrome).flatten().astype(int)))
        f.write(f"{syndrome_str}\n")
        
        f.write("1.0\n")          # Valor fijo pie 1
        f.write("100\n")          # Valor fijo pie 2
        
    print(f"Archivo de testbench guardado: {filename}")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main_export():
    codeConfig = "144" 
    p = 0.005 
    print(f"Generando datos para configuración: {codeConfig} con p={p}")

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
    detectors, observables = sampler.sample(1, separate_observables=True)
    syndrome = detectors[0]

    # 3. Ejecutar las nuevas exportaciones
    # Generar raw .h y obtener los degrees máximos
    max_row, max_col = generar_archivos_raw_h(pcm, codeConfig)
    
    # Actualizar el main header con los nuevos defines
    actualizar_main_header("krnl_min_sum_matrix_std.h", max_row, max_col,pcm.shape[1],pcm.shape[0])
    
    # Generar el TXT en el nuevo formato
    guardar_sindrome_custom(f"input_{codeConfig}.txt", pcm, syndrome)

if __name__ == "__main__":
    main_export()