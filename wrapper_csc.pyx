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
      - L (np.ndarray[float64]): matriz de valores
      - pcm (np.ndarray[int32]): matriz de paridad en formato comprimido

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

# Wrapper class for the c struct sparse_matrix_t
cdef class SparseMatrixWrapper:
    cdef sparse_matrix_t *mat

    def __cinit__(self):
        # se inicializa a cero por seguridad
        self.mat = <sparse_matrix_t*>malloc(sizeof(sparse_matrix_t))
    
    @property
    def values_csc(self):
        cdef int nnz = self.mat.nnz
        return np.array(self.mat.values_csc, copy=True)
    
    @property
    def values_csr(self):
        cdef int nnz = self.mat.nnz
        return np.array(self.mat.values_csr, copy=True)