import go100x
import numpy as np

sizea=855 #*3
sizeb=798 #*3
#sizea=9
#sizeb=10

def calc_grid(blockx,size):
    return int((size-1+blockx)/blockx)
     
a = np.zeros(sizea)
b = np.zeros(sizeb)
a=a+2
b=b+3

a = np.random.rand(sizea)
b = np.random.rand(sizeb)

# cpu version of function
c = go100x.fun_cpu(a,b)

##  gpu_funv0: deprecated algorithm
g0 = go100x.fun_gpu([855,1,1],[256,1,1],a,b)

print("max error:",max(abs(c-g0)))
#quit()
## gpu_funv1: better algorithm
# number of threads per block
for blockx in [128,256,512,1024]:

    # number of blocks
    gridx=calc_grid(blockx,sizea)
  
    for blocky in [1,128,256,512,1024]:
        if blocky == 1:
           gridy=1
        else:
           gridy=calc_grid(blocky,sizeb)
 
        for blockz in [1]:
            if blockz == 1:
               gridz=1
            #else:
            #   gridz=calc_grid(blocky)

            if blockx*blocky*blockz<=1024:
               block = [blockx,blocky,blockz]
               ngrid = [gridx,gridy,gridz]
               
               # gpu version of function
               g1 = go100x.funv1_gpu(ngrid,block,a,b)
               
               # error check
               print(c[1:10])
               print(g0[1:10])
               print(g1[1:10])
               print("                               v0 max error:",max(abs(c-g0)))
               print("ngrid:",ngrid,"block:", block,"v1 max error:",max(abs(c-g1)))
