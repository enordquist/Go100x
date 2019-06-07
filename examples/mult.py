#!/usr/bin/env python

import os
import go100x
import numpy as np

os.environ["TIMEMORY_PRECISION"] = "8"

n = int(2.0e8)
a = np.zeros([n], dtype=np.float)
b = np.zeros([n], dtype=np.float)
a += 2
b += 3

c = go100x.calculate_cpu(a, b)

go100x.set_device(0)

o = [32, 64, 128, 256, 512, 1024]
for block in o:
    for ngrid in o + [-1]:
        if ngrid < 0:
            ngrid = int((n + block - 1) / block)
        d = go100x.calculate_gpu([ngrid], [block], a, b)

print("\nResults for array of size {} ({})", n, float(n))
