from cython.view cimport array as cvarray
import numpy as np
#we define the header for the C function we want to call, it has to match the C function signature
cdef extern from "min_sum.h":
    void min_sum(float *L, int *non_zero, float *result)



def compute_min_sum_wrapper(L,non_zero):
    cdef float result
    if not L.flags.c_contiguous:
        L = np.ascontiguousarray(L)
    if not non_zero.flags.c_contiguous:
        non_zero = np.ascontiguousarray(non_zero)
    
    cdef float[::1] L_array = L
    cdef int[::1] non_zero_array = non_zero
    min_sum(&L_array[0], &non_zero_array[0], &result)
    return result