
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
#define VNODES 540
#endif
#ifndef CHECK
#define CHECK  90
#endif
#ifndef NNZ
#define NNZ 1845
#endif
#ifndef MAX_ROW_LENGTH
#define MAX_ROW_LENGTH 25
#endif
#ifndef MAX_COL_LENGTH
#define MAX_COL_LENGTH 6
#endif
#ifndef NUM_IT
#define NUM_IT 100
#endif
#ifndef NUM_SYNDROMES
#define NUM_SYNDROMES 100
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

const value_t VAL_MAX = 200.0;

void min_sum_opt(bit_t* syndrome, value_t Lj, value_t alpha, bit_t *error_computed);
void min_sum_opt_csr( bus_512_t *syndrome, value_t Lj, value_t alpha, bus_512_t *error_computed);
