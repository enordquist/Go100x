
#include "Go100x/kernels.hpp"

//======================================================================================//
// dummy cuda kernel
//
__global__
void calculateKernel(const float* input_a, const float* input_b, float* output, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < size; i += stride)
  {
    output[i] = input_a[i] * input_b[i];
  }
}

//======================================================================================//
// launch the kernel
//
void gpu_calculate(int block, int ngrid, const float* input_a, const float* input_b,
                   float* output, int size)
{
    
  calculateKernel<<<ngrid, block>>>(input_a, input_b, output, size);
 }
