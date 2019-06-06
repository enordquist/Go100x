
#include "Go100x/kernels.hpp"

//======================================================================================//
// calculate on the CPU
//
void cpu_calculate(const float* input_a, const float* input_b, float* output, int size) {
  for(int i = 0; i < size; i++)
  {
      output[i] = input_a[i] * input_b[i];
  }
}

void cpu_fun(const float* R, const float* r, float* D, const int J, const int N) {

  for(int i = 0; i < N; i++) {
  D[i] = 0.f;
    for(int j = 0; j < J; j++)
    {
      for(int k = 0; k < N; k++)
      {
        // do stuff
        D[i] += abs (R[i] + r[j] - R[k]);
      }
    }
  }
}
