
#pragma once

#include "Go100x/macros.hpp"
#include <iostream>
#include <tuple>
#include <vector>

//======================================================================================//
// dummy cpu function that launches calcuations
//
extern void cpu_calculate(const float* input_a, const float* input_b, float* output,
                          int size);

// fun takes Ri and rj as inputs, will loop over J and K, and return D
extern void cpu_fun(const float* R, const float* r, float* D, const int J, const int N);

//======================================================================================//
// dummy cuda kernel
//
extern void gpu_calculate(const dim3& block, const dim3& ngrid, const float* input_a,
                          const float* input_b, float* output, int size);

//======================================================================================//
//// launch the kernel
////
extern void gpu_fun(const dim3& ngrid, const dim3& block, const float* R, const float* r,
                    float* D, const int J, const int N);

extern void gpu_funv1(const dim3& ngrid, const dim3& block, const float* R,
                      const float* r, float* D, const int N, const int J);
