#pragma once

#ifndef VNODES
#define VNODES 40
#endif
#ifndef CHECK 
#define CHECK 20
void compute_check_to_value(float L[CHECK][VNODES], int* syndrome, int size_checks,int size_vnode, float codeword[CHECK], float alpha);
#endif