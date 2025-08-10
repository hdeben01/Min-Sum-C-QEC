#pragma once

#ifndef VNODES
#define VNODES 10
#endif
#ifndef CHECK 
#define CHECK 3
void compute_check_to_value(float L[CHECK][VNODES], const int pcm_matrix[CHECK][VNODES],
                            int* syndrome, int size_checks, int size_vnode, 
                            float Lj[CHECK], float alpha, int num_it);
#endif

#ifndef COLORS_H
#define COLORS_H

#include <stdarg.h> // for va_list, va_start, va_end

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


