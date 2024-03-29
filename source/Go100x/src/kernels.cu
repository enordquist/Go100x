
#include "Go100x/kernels.hpp"

//======================================================================================//
// dummy cuda kernel
//
__global__ void calculateKernel(const float* input_a, const float* input_b, float* output,
                                int size)
{
    int index  = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < size; i += stride)
    {
        output[i] = input_a[i] * input_b[i];
    }
}

__global__ void funv1Kernel(
    // R_d: coordinates of atoms
    // r_d: grid points for Lebedev quadratur
    // J: number of grid points
    // N: number of atoms
    // D_d: output/Born_radii
    const float* R_d, const float* r_d, float* D_d, int N, int J)
{
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    int k0 = threadIdx.z + blockIdx.z * blockDim.z;

    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;
    int stridez = blockDim.z * gridDim.z;

    for(int i = i0; i < N; i += stridex)
    {
        D_d[i] = 0.0f;
        for(int j = j0; j < J; j += stridey)
        {
            for(int k = k0; k < N; k += stridez)
            {
                // do distance calulation between neighboring atoms
                // and grid point
                // we have data race here
                D_d[i] += abs(R_d[i] + r_d[j] - R_d[k]);
                // atomicAdd(&D_d[i], abs (R_d[i] + r_d[j] - R_d[k]));
            }
        }
    }
}

//======================================================================================//
// launch the kernel
//
void gpu_calculate(const dim3& ngrid, const dim3& block, const float* input_a,
                   const float* input_b, float* output, int size)
{
    calculateKernel<<<ngrid, block>>>(input_a, input_b, output, size);
}

//======================================================================================//
// dummy cuda kernel
//
__global__ void funKernel(const float* R, const float* r, float* D, const int J,
                          const int N)
{
    __shared__ float output_shared[256];
    float            DistR;
    int              atomI = blockIdx.x, i;

    output_shared[threadIdx.x] = 0.f;
    for(int j = threadIdx.x; j < J; j += blockDim.x)
    {
        for(int k = 0; k < N; k++)  // Loop the LookupTable
        {
            DistR = R[atomI] + r[j] - R[k];
            output_shared[threadIdx.x] += fabsf(DistR);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0)
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
void gpu_fun(const dim3& ngrid, const dim3& block, const float* R, const float* r,
             float* D, const int J, const int N)
{
    funKernel<<<ngrid, block>>>(R, r, D, J, N);
}

// void gpu_funv1(int3 ngrid, int3 block, const float* R_d, const float* r_d,
void gpu_funv1(const dim3& ngrid, const dim3& block, const float* R_d, const float* r_d,
               float* D_d, const int N, const int J)
{
    funv1Kernel<<<ngrid, block>>>(R_d, r_d, D_d, N, J);
}
