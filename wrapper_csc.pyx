# cython: language_level=3
#
# This Cython wrapper provides an interface to the C function `min_sum` defined in "min_sum.h".
#
# Functions:
#   - compute_min_sum_wrapper(L, non_zero, syndrome, size_checks, size_vnodes, priors, alpha, num_it)
#       Wraps the C function `min_sum` for use with NumPy arrays.
#       Parameters:
#           L (np.ndarray): Data structure representing messages, the algorithm uses it in-place for calculating messages
#           non_zero (np.ndarray): Parity-check matrix in compressed format, must be 1D and int32.
#           syndrome (np.ndarray): Syndrome vector, must be 1D and int32.
#           size_checks (int): Number of check nodes.
#           size_vnodes (int): Number of variable nodes.
#           priors (np.ndarray): Array of log-likelihood ratios (LLRs), must be 1D and float32.
#           alpha (float): Scaling factor for the min-sum algorithm.
#           num_it (int): Number of iterations to perform.
#       Returns:
#           np.ndarray: Updated LLRs after running the min-sum algorithm.
#
# Notes:
#   - Input arrays are converted to contiguous memory if necessary.
#   - NumPy arrays are passed to the C function as memoryviews for efficient access.
#   - The function assumes that the input arrays are properly shaped and typed.
from cython.view cimport array as cvarray
import numpy as np
from libc.stdlib cimport malloc, free

from scipy import sparse
from libc.stdlib cimport malloc, free, calloc
#we define the header for the C function we want to call, it has to match the C function signature

cdef extern from "min_sum_csc.h":
    ctypedef struct sparse_matrix_t:
        int rows
        int cols
        int *offset_cols
        int *row_index
        int *offset_rows
        int *col_index
        int *edges
        int nnz
        double *values_csc
        double *values_csr

    void min_sum(sparse_matrix_t *L, 
                            int* syndrome, int size_checks, int size_vnode, 
                            double *Lj, double alpha, int num_it, int * error_computed)

    void to_sparse_matrix_t(double *L,sparse_matrix_t * out,  int *pcm)


#in order to use the wrapper is necessary to flatten the matrices first
def compute_min_sum_wrapper(SparseMatrixWrapper L,syndrome,size_checks,size_vnodes,priors,alpha,num_it,error_computed):
   
    
    #create memoryviews so we can pass numpy array pointers to C function
 
    cdef int[::1] syndrome_array = syndrome
    cdef double[::1] priors_array = priors
    cdef int[::1] error_computed_array = error_computed

    min_sum(L.mat,&syndrome_array[0], size_checks, size_vnodes, &priors_array[0],alpha,num_it, &error_computed_array[0])
    return L, error_computed_array

def init_sparse_matrix_t(L,pcm):
    """
    Inicializa un sparse_matrix_t a partir de:
      - L (np.ndarray[float64]): matriz de valores completa (m*n)
      - pcm (np.ndarray[int32]): matriz de paridad completa (m*n)

    Devuelve:
      SparseMatrix: objeto que envuelve el struct C
    """


    if not L.flags.c_contiguous:
        L = np.ascontiguousarray(L, dtype=np.float64)
    if not pcm.flags.c_contiguous:
        pcm = np.ascontiguousarray(pcm, dtype=np.int32)

    cdef double[::1] L_array = L
    cdef int[::1] pcm_array = pcm

    # Crear wrapper
    sm = SparseMatrixWrapper()

    # Llamar a la función C para rellenar el struct
    to_sparse_matrix_t(&L_array[0], sm.mat, &pcm_array[0])

    return sm

def init_sparse_matrix_from_csc(pcm, L_values):
    """
    Inicializa un sparse_matrix_t a partir de una matriz CSC (scipy.sparse.csc_matrix)
    y un vector de valores (L_values) que corresponden a los nnz elementos.

    Se generan ambas representaciones (CSC y CSR), además del array edges.

    Args:
        pcm: scipy.sparse.csc_matrix (dtype int32)
        L_values: np.ndarray[np.float64] con valores asociados a los elementos no nulos

    Returns:
        SparseMatrixWrapper: objeto Cython que contiene el struct sparse_matrix_t*
    """

    if not sparse.isspmatrix_csc(pcm):
        raise ValueError("pcm debe ser una scipy.sparse.csc_matrix")

    pcm = pcm.astype(np.int32, copy=False)
    L_values = np.ascontiguousarray(L_values, dtype=np.float64)

    cdef int rows = pcm.shape[0]
    cdef int cols = pcm.shape[1]
    cdef int nnz = pcm.nnz

    # Crear wrapper
    cdef SparseMatrixWrapper sm = SparseMatrixWrapper()
    sm.mat.rows = rows
    sm.mat.cols = cols
    sm.mat.nnz = nnz

    # Memoryviews desde los arrays CSC
    cdef int[::1] indptr = pcm.indptr
    cdef int[::1] indices = pcm.indices
    cdef double[::1] values = L_values

    # Asignar CSC directamente
    sm.mat.offset_cols = <int*>malloc((cols + 1) * sizeof(int))
    sm.mat.row_index   = <int*>malloc(nnz * sizeof(int))
    sm.mat.values_csc  = <double*>malloc(nnz * sizeof(double))

    if sm.mat.offset_cols == NULL or sm.mat.row_index == NULL or sm.mat.values_csc == NULL:
        raise MemoryError("No se pudo reservar memoria para CSC")

    cdef int i, k
    for i in range(cols + 1):
        sm.mat.offset_cols[i] = indptr[i]
    for k in range(nnz):
        sm.mat.row_index[k] = indices[k]
        sm.mat.values_csc[k] = values[k]

    # --- Construir CSR a partir del CSC ---
    sm.mat.offset_rows = <int*>malloc((rows + 1) * sizeof(int))
    sm.mat.col_index   = <int*>malloc(nnz * sizeof(int))
    sm.mat.values_csr  = <double*>malloc(nnz * sizeof(double))
    sm.mat.edges       = <int*>malloc(nnz * sizeof(int))

    if (sm.mat.offset_rows == NULL or sm.mat.col_index == NULL or
        sm.mat.values_csr == NULL or sm.mat.edges == NULL):
        raise MemoryError("No se pudo reservar memoria para CSR/edges")

    #Contar elementos en cada fila
    cdef int[:] row_counts = np.zeros(rows, dtype=np.int32)
    for j in range(cols):
        for k in range(indptr[j], indptr[j + 1]):
            row_counts[indices[k]] += 1

    #Construir offset_rows
    sm.mat.offset_rows[0] = 0
    for i in range(rows):
        sm.mat.offset_rows[i + 1] = sm.mat.offset_rows[i] + row_counts[i]

    #Rellenar CSR
    cdef int[:] fill_ptr = np.zeros(rows, dtype=np.int32)
    cdef int pos
    for j in range(cols):
        for k in range(indptr[j], indptr[j + 1]):
            i = indices[k]  # fila
            pos = sm.mat.offset_rows[i] + fill_ptr[i]
            sm.mat.col_index[pos] = j
            sm.mat.values_csr[pos] = sm.mat.values_csc[k]
            fill_ptr[i] += 1

    #Construir edges (relación CSC ↔ CSR)
    cdef int[:] col_counts = np.zeros(cols, dtype=np.int32)
    for i in range(rows):
        for k in range(sm.mat.offset_rows[i], sm.mat.offset_rows[i + 1]):
            col = sm.mat.col_index[k]
            pos = sm.mat.offset_cols[col] + col_counts[col]
            sm.mat.edges[pos] = k
            col_counts[col] += 1

    return sm

# Wrapper class for the c struct sparse_matrix_t
cdef class SparseMatrixWrapper:
    cdef sparse_matrix_t *mat

    def __cinit__(self):
        # se inicializa a cero por seguridad
        self.mat = <sparse_matrix_t*>malloc(sizeof(sparse_matrix_t))
    
    @property
    def values_csc(self):
        cdef int nnz = self.mat.nnz
        cdef double[:] values_csc_view = <double[:nnz]> self.mat.values_csc
        return np.asarray(values_csc_view, copy=True)
    
    @property
    def values_csr(self):
        cdef int nnz = self.mat.nnz
        cdef double[:] values_csr_view =  <double[:nnz]>self.mat.values_csr
        return np.asarray(values_csr_view, copy=True)
    
    @property
    def nnz(self):
        return self.mat.nnz
    
    
    def set_values_csc(self, values):
        cdef int nnz = self.mat.nnz
        if values.size != nnz:
            raise ValueError("Size of input array must match number of non-zero elements")
        cdef double[::1] values_array = values
        for i in range(nnz):
            self.mat.values_csc[i] = values_array[i]
    
    def set_values_csr(self, values):
        cdef int nnz = self.mat.nnz
        if values.size != nnz:
            raise ValueError("Size of input array must match number of non-zero elements")
        cdef double[::1] values_array = values
        for i in range(nnz):
            self.mat.values_csr[i] = values_array[i]

