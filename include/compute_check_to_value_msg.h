#pragma once

#ifndef VNODES
#define VNODES 10
#endif
#ifndef CHECK 
#define CHECK 3
void compute_check_to_value(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, float Lj[CHECK], float alpha, int num_it, int pcm_matrix[CHECK][VNODES]);
#endif