from cython.view cimport array as cvarray
import numpy as np
#we define the header for the C function we want to call, it has to match the C function signature
cdef extern from "min_sum.h":
    void min_sum(float *L, const int *pcm_matrix, 
                            int* syndrome, int size_checks, int size_vnode, 
                            float *Lj, float alpha, int num_it)



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