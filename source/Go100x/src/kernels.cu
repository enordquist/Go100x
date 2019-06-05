
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
  float* input_a_d, *input_b_d, *output_d;
  cudaMalloc(&input_a_d, size*sizeof(float));
  cudaMalloc(&input_b_d, size*sizeof(float));
  cudaMalloc(&output_d, size*sizeof(float));
  
  cudaMemcpy(input_a_d, input_a, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_b_d, input_b, size*sizeof(float), cudaMemcpyHostToDevice);
  
  calculateKernel<<<ngrid, block>>>(input_a, input_b, output, size);
  
  cudaDeviceSynchronize();

  cudaFree(&input_a_d);
  cudaFree(&input_b_d);
  cudaMemcpy(output, output_d, size*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(&output_d);
}
