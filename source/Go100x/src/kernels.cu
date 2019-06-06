
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



//
//======================================================================================//
// dummy cuda kernel
//
__global__
void funKernel(const float* R, const float* r, float* D, const int J, const int N)
{
  __shared__ float output_shared[256];
  float DistR;
  int atomI = blockIdx.x, i;

  output_shared[threadIdx.x] = 0.f;
  for(int j = threadIdx.x; j < J; j += blockDim.x)
  {
     for(int k = 0; k < N; k++) // Loop the LookupTable
     {
        DistR = R[atomI] + r[j] - R[k];
        output_shared[threadIdx.x] += fabsf(DistR);
     }
  }
  __syncthreads();
  if(threadIdx.x==0) 
  {
    DistR = 0.f;
    for(i = 0; i < blockDim.x; i++) 
    {
      DistR += output_shared[i];
    }
    D[atomI] = DistR;
  }
}

//======================================================================================//
// launch the kernel
//
void gpu_fun(int ngrid, int block, const float* R, const float* r,
                   float* D, const int J, const int N)
{
  funKernel<<<ngrid, block>>>(R, r, D, J, N);
}

