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
#we define the header for the C function we want to call, it has to match the C function signature
cdef extern from "min_sum.h":
    void min_sum(float *L, const int *pcm_matrix, 
                            int* syndrome, int size_checks, int size_vnode, 
                            float *Lj, float alpha, int num_it)


#in order to use the wrapper is necessary to flatten the matrices first
def compute_min_sum_wrapper(L,non_zero,syndrome,size_checks,size_vnodes,priors,alpha,num_it):
    if not L.flags.c_contiguous:
        L = np.ascontiguousarray(L)
    if not non_zero.flags.c_contiguous:
        non_zero = np.ascontiguousarray(non_zero)
    
    #create memoryviews so we can pass numpy array pointers to C function
    cdef float[::1] L_array = L
    cdef int[::1] non_zero_array = non_zero
    cdef int[::1] syndrome_array = syndrome
    cdef float[::1] priors_array = priors


    min_sum(&L_array[0], &non_zero_array[0],&syndrome_array[0], size_checks, size_vnodes, &priors_array[0],alpha,num_it)
    return L_array