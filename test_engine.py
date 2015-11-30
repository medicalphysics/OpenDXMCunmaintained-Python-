# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:33:05 2015

@author: ander
"""
import numpy as np
from opendxmc.engine import array_indices_py


particle = np.array([1000, 0, 0, -1, 0, 0, 80000, 1], dtype=np.double)
N = np.zeros(3,dtype=np.double) + 9
spacing = np.zeros(3,dtype=np.double) + 1
offset = -N*spacing /2.

ind, l = array_indices_py(particle, N, spacing, offset)
print(len(l))


print(array_indices_py(particle, N, spacing, offset))