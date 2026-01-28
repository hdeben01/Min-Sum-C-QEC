import sys
import os
import numpy as np
from scipy import sparse
from dem_to_matrices import detector_error_model_to_check_matrices
from IBM_STIM import create_bivariate_bicycle_codes, build_circuit, select_configuration

# --------------------------------------------------------------------------
# Función para guardar en el formato especificado
# --------------------------------------------------------------------------
def guardar_datos_formato_custom(filename, pcm, syndrome, header_val=0.0, footer_val=1.0):
    """
    Guarda la PCM y el síndrome en un archivo de texto con el formato:
    0.0
    M N
    [Matriz densa fila por fila]
    [Vector Síndrome]
    1.0
    """
    m, n = pcm.shape
    
    with open(filename, 'w') as f:
        f.write(f"{header_val}\n")
        
       
        f.write(f"{m} {n}\n")
        
       
        if sparse.issparse(pcm):
            pcm_dense = pcm.toarray().astype(int)
        else:
            pcm_dense = pcm.astype(int)
            
        pcm_dense = pcm_dense.flatten()
        f.write(" ".join(map(str, pcm_dense)))
        f.write("\n")
        # 4. Escribir el síndrome (detectors)
        # Aseguramos que sea un array plano de enteros
        syndrome_dense = np.array(syndrome).flatten().astype(int)
        f.write(" ".join(map(str, syndrome_dense)) + "\n")
        
        # 5. Escribir el valor de pie (1.0 en tu ejemplo, coincide con alpha)
        f.write(f"{footer_val}\n")

    print(f"Archivo guardado exitosamente: {filename}")

def main_export():
 
    codeConfig = "72" 

    p = 0.005 
    
   
    d = 12 # Distancia del código 
    alpha = 1.0 # Valor que aparece al final del formato 

    print(f"Generando datos para configuración: {codeConfig} con p={p}")

    # 1. Selección y construcción del código Bivariate Bicycle
    config = select_configuration(codeConfig)
    ell, m_param = config["ell"], config["m"]
    a1, a2, a3 = config["a"]
    b1, b2, b3 = config["b"]
    
    A_x_pows, A_y_pows = [a1], [a2, a3]
    B_x_pows, B_y_pows = [b2, b3], [b1]

    code, A_list, B_list = create_bivariate_bicycle_codes(ell, m_param, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
    
    # 2. Construcción del circuito y obtención de matrices
    circuit = build_circuit(code, A_list, B_list, 
                p=p, 
                num_repeat=1, 
                z_basis=False,   
                use_both=False
                )

    dem = circuit.detector_error_model()
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    
    pcm = matrices.check_matrix
    print(f"Dimensiones de PCM: {pcm.shape}")

    # 3. Generar una muestra (Síndrome)
    sampler = circuit.compile_detector_sampler()
    # Tomamos 1 sola muestra (num_shots=1)
    detectors, observables = sampler.sample(1, separate_observables=True)
    
    # El síndrome es el primer (y único) elemento de detectors
    syndrome = detectors[0]

    # 4. Guardar en el archivo
    output_filename = f"datos_exportados_{codeConfig}.txt"
    
    # Llamamos a la función de guardado
    # header_val=0.0 (como en tu ejemplo)
    # footer_val=alpha (usando tu variable alpha=1.0)
    guardar_datos_formato_custom(output_filename, pcm, syndrome, header_val=0.0, footer_val=alpha)

if __name__ == "__main__":
    main_export()