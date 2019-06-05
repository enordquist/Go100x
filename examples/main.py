#!/usr/bin/env python


import go100x
import numpy as np

n = 100000000
a = np.zeros([n], dtype=np.float)
b = np.zeros([n], dtype=np.float)
a += 2
b += 3

c = go100x.calculate_cpu(a, b)

o = [32, 64, 128, 256, 512, 1024]
for block in o:
    for ngrid in o + [-1]:
        if ngrid < 0:
            ngrid = int((n + block - 1) / block)
        d = go100x.calculate_gpu(block, ngrid, a, b)

print("\nResults for array of size {} ({})", n, float(n))
#print(c)
#print(d)
