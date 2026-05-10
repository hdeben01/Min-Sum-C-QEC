#ifndef VNODES
#define VNODES 540
#endif
#ifndef CHECK 
#define CHECK  90
#endif

#ifndef MAX_ROW_DEGREE
#define MAX_ROW_DEGREE 25
#endif
#ifndef MAX_COL_DEGREE
#define MAX_COL_DEGREE 6
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
