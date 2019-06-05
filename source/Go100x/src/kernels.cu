
#include "Go100x/kernels.hpp"

//======================================================================================//
// dummy cuda kernel
//
__global__
void calculateKernel(float* matrix_a, float* matrix_b, int size)
{

}

//======================================================================================//
// launch the kernel
//
void gpu_calculate(int block, int ngrid, float* matrix_a, float* matrix_b, int size)
{
    calculateKernel<<<ngrid, block>>>(matrix_a, matrix_b, size);
}
