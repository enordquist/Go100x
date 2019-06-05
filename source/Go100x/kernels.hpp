
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

//======================================================================================//
// dummy cuda kernel
//
extern void gpu_calculate(int block, int ngrid, const float* input_a,
                          const float* input_b, float* output, int size);
