#pragma once

#ifndef VNODES
#define VNODES 2232
#endif
#ifndef CHECK 
#define CHECK 252
#endif

#ifndef COLORS_H
#define COLORS_H

#include <stdarg.h> // for va_list, va_start, va_end
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

//this data structure is a mixture of csc and csr representation in order to iterate over a sparse matrix in row and column order
//https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html#compressed-sparse-column-csc
typedef struct {
    int rows;
    int cols;
    //csc
    int *offset_cols;
    int *row_index;
    //csr
    int *offset_rows;
    int *col_index;

    //pairs 
    int *edges;
    int nnz; //number of non zero values
    //double *values;
    double *values_csc;
    double *values_csr;

} sparse_matrix_t;



#define RESET   "\033[0m"
#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define YELLOW  "\033[0;33m"
#define CYAN    "\033[0;36m"

// Function to print colored text
static inline void color_printf( char *color,  char *format, ...) {
    va_list args;
    va_start(args, format);

    printf("%s", color);
    vprintf(format, args);
    printf("%s", RESET);

    va_end(args);
}

#endif // COLORS_H

void compute_row_operations(sparse_matrix_t *L,
                            int* syndrome, int size_checks, int size_vnode);

void compute_col_operations(sparse_matrix_t *L, 
                            int* syndrome, int size_checks, int size_vnode, double alpha, 
                            double Lj[VNODES], double sum[VNODES]);

void show_matrix( double *matrix, int *non_zero,
                  int rows,  int cols);

void min_sum(sparse_matrix_t *L, 
                            int* syndrome, int size_checks, int size_vnode, 
                            double Lj[VNODES], double alpha, int num_it, int *error_computed);

void to_sparse_matrix_t(double *L, sparse_matrix_t *out, int *pcm);

void csc_to_csr(sparse_matrix_t *L);

void csr_to_csc(sparse_matrix_t *L);