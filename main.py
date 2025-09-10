import sys
sys.path.append('./wrapper.cpython-313-x86_64-linux-gnu.so')
import numpy as np
from wrapper import compute_min_sum_wrapper  # tu wrapper Cython

def random_error(pcm,rows,cols,per) -> np.ndarray:
    return np.random.binomial(1,per,size=cols)
    


def main():
    pcm = []
    with open("B1_matrix.txt", "r") as f:
        rows = len(f.readlines())
        f.seek(0,0)
        for line in f:
            p = line.split(',')
            list_float = [float(i) for i in p]
            pcm.extend(list_float)
        
    cols = int(len(pcm)/441)
    p = 0.0
    p = (1.0 - (2.0/3.0) * p)
    alpha = 1.25
    print(rows)
    print(len(pcm))
    print(cols)
     
    NMCs = [10**4, 10**4, 10**4, 10**4, 10**4]  
    
    # Physical error rate that is simulated
    
    ps = np.linspace(0.001, 0.005, num=5)  

    for index, p_error in enumerate(ps):
        logical_errors = 0
        for iteration in range(NMCs[index]):
       
            error = random_error(pcm,rows,int(cols),p_error)
            #print(error)
            pcm = np.array(pcm, dtype=np.int32)
            syndrome = np.empty(rows, dtype=np.int32)

            for i in range(rows):
                row = pcm[i * cols:(i + 1)*cols].astype(np.int32) & 1
                syndrome[i] = (row @ error) & 1
            
            #print(syndrome)
            Lj = np.full(cols,p)
            num_it = 100

            # formatting imputs correctly for the c function, L_flat and Lj use float(32 bits) and pcm and syndrome int_32
            L_flat = pcm.astype(np.double).copy()
            L_flat = np.ascontiguousarray(L_flat)
            pcm = np.ascontiguousarray(pcm.astype(np.int32))
            syndrome = np.ascontiguousarray(syndrome.astype(np.int32))
            Lj = np.ascontiguousarray(Lj.astype(np.double))
            error_computed = np.zeros(cols,dtype=np.int32)


            # Llamar a la función de Cython con vectores aplanados
            L_array = compute_min_sum_wrapper(L_flat, pcm.astype(np.int32), syndrome.astype(np.int32), rows, cols, Lj.astype(np.double), alpha, num_it, error_computed)
            logical_error = (error + error_computed) %2
            if np.any(logical_error == 1):
                logical_errors += 1
        print("logical errors with physical error rate: ",p_error)
        print(logical_errors)
            



    

if __name__ == "__main__":
    main()