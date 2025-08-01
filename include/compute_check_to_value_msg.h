#ifndef _H_CHECK_TO_VALUE_H_
#define _H_CHECK_TO_VALUE_H_
#pragma once

#define CHECK 20
#define VNODES 40

float compute_check_to_value(float L[CHECK][VNODES], int* syndrome, float out[CHECK][VNODES], int size_checks,int size_vnode);
#endif