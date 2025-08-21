#pragma once

#ifndef VNODES
#define VNODES 7
#endif
#ifndef CHECK 
#define CHECK 3
#endif

#ifndef COLORS_H
#define COLORS_H

#include <stdarg.h> // for va_list, va_start, va_end
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#define RESET   "\033[0m"
#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define YELLOW  "\033[0;33m"
#define CYAN    "\033[0;36m"

// Function to print colored text
static inline void color_printf(const char *color, const char *format, ...) {
    va_list args;
    va_start(args, format);

    printf("%s", color);
    vprintf(format, args);
    printf("%s", RESET);

    va_end(args);
}

#endif // COLORS_H

void compute_row_operations(float L[CHECK][VNODES], const int non_zero[CHECK][VNODES],
                            int* syndrome, int size_checks, int size_vnode);

void compute_col_operations(float L[CHECK][VNODES], const int non_zero[CHECK][VNODES], 
                            int* syndrome, int size_checks, int size_vnode, float alpha, 
                            float Lj[VNODES], float sum[VNODES]);

void show_matrix(const float matrix[CHECK][VNODES], const int non_zero[CHECK][VNODES],
                 const int rows, const int cols);

void min_sum(float L[CHECK][VNODES], const int pcm_matrix[CHECK][VNODES], 
                            int* syndrome, int size_checks, int size_vnode, 
                            float Lj[VNODES], float alpha, int num_it);