
#include "Go100x/kernels.hpp"

//======================================================================================//
// calculate on the CPU
//
void cpu_calculate(int block, int ngrid, float* matrix_a, float* matrix_b, int size) 
{
  for(int i = 0; i < size; i++)
  {
    std::cout << matrix_a[i] << matrix_b[i] << std::endl;

  };
}

int main()
{
  int size = 100;

  float a[size];
  float b[size];
  for (int i = 0; i < size; i++ ) 
  {
    a[i] = i;
    b[i] = i;
  };
  cpu_calculate(1, 1, a, b, 5);

  return 0;
}
