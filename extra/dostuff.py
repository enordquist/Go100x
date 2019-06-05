import numpy as np

# filepaths
path = '/ccsopen/proj/gen126/go100x/'
data = 'test_data/'
xyzq = 'posq_cuda.dat'   # cartesian coordinates and partial charges
para = 'params_cuda.dat' # parameters: vdW radius, and an unknown column

# read
xyzq_arr = np.genfromtxt(path+data+xyzq, delimiter=" ")
para_arr = np.genfromtxt(path+data+para, delimiter=" ")

# to be used by GBMV plugin:
#     x, y, z coors and vdw radius
   x_arr = xyzq_arr[:,0]
   y_arr = xyzq_arr[:,1]
   z_arr = xyzq_arr[:,2]
vdwr_arr = para_arr[:,0]

#for i in Natoms:
  #for i in Ngrid:
    #for i in Nneighbors:
#     fillNeighbors(xyzq_arr)
# test example: 3gb1 protein including 855 atoms
# original one: <<<855,1021>>> Vs. <<<512,256>>>

