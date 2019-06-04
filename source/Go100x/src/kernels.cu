
#include "Go100x/kernels.hpp"

//======================================================================================//
// dummy cuda kernel
//
__global__
void gpu::calculate(int size, int* indices)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
	int istride = blockDim.x * gridDim.x;

	for(int i = i0; i < size; i += istride)
	{
		// ...
		printf("index[%i] = %i\n", i, indices[i]);
	}
}
