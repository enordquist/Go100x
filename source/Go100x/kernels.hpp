
#pragma once

#include "Go100x/macros.hpp"
#include <iostream>
#include <tuple>
#include <vector>

//======================================================================================//
// dummy cpu function that launches calcuations
//
extern void cpu_calculate(int block, int ngrid, float* matrix_a, float* matrix_b,
                          int size);

//======================================================================================//
// dummy cuda kernel
//
extern void gpu_calculate(int block, int ngrid, float* matrix_a, float* matrix_b,
                          int size);
