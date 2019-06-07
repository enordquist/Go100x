import go100x
import numpy as np

size=10#00000000

def calc_grid(blocki):
  return int((size-1+blocki)/blocki)
     
# number of threads per block
for blockx in [512]:
  gridx=calc_grid(blockx)
  for blocky in [1]:
    gridy=calc_grid(blocky)
    for blockz in [1]:
      gridz=calc_grid(blockz)

      block = np.ndarray([blockx, blocky, blockz])
      ngrid = np.ndarray([gridx,  gridy,  gridz])
          
      # coordinates of all atoms
      a = np.zeros([size], dtype=np.float)
      a += 2
      # grid points
      b = np.zeros([size], dtype=np.float)
      b += 3
            
      # cpu version of function
      c = go100x.fun_cpu(a,b)
      # gpu version of function
      d = go100x.funv1_gpu(block,ngrid,a,b)
            
      # error check
      print("ngrid:",ngrid,"block:", block,"max error:",max(abs(c-d)))
