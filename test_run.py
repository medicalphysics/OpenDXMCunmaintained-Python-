# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:07:57 2015

@author: ERLEAN
"""

import numpy as np
import time
from fuzzy_time import human_time
from tube.tungsten import specter
from phase_space.ct import spiral
from phantoms.golem import read_golem
from engine import score_energy, is_intersecting_py

import os
import sys
import tables as tb

import pylab as plt
import pdb


def save_array_to_h5(path, array, name):
    
    h5 = tb.open_file(path, mode='a')
    try:
        h5.get_node(h5.root, name)
    except tb.NoSuchNodeError:
        pass
    else:
        h5.remove_node(h5.root, name, True)
    h5.create_carray(h5.root, name=name, obj=array)
    h5.close()
 
def save_dict_to_h5(path, d, name, dtype=None):
    h5 = tb.open_file(path, mode='a')
    try:
        h5.get_node(h5.root, name)
    except tb.NoSuchNodeError:
        pass
    else:
        h5.remove_node(h5.root, name, True)
    # creating structured array:
    n = len(d)
    if dtype is None:
        dtype = {'names':['index', 'tissue'], 'formats': ['i4', 'a64']}
    arr = np.zeros(n, dtype=dtype)
    teller = 0
    for key, value in d.items():
        arr['tissue'][teller] = value
        arr['index'][teller] = key
        teller += 1
    h5.create_table(h5.root, name, description=arr)
    h5.close()
    
H5PATH = os.path.join(os.path.dirname(sys.argv[0]), 'data.h5')

def test():

    print 'Reading phantom Golem'
    spacing, density_array, lut, material_array, organ_array, organ_map = read_golem()
    print 'Done'
    print 'Setting up phase space'
    phase_space_args = (50., 108.56, 3.84)

    phase_space_kwargs = {'pitch': 0.8,
                          'start': 0.,
                          'stop': spacing[2] * density_array.shape[2],
                          'exposures': 1200,
                          'histories': 10,
                          'energy_specter': specter(120., angle_deg=12.,
                                                    filtration_materials='Al',
                                                    filtration_mm=4.),
                          'batch_size': 30000,
                        }

    particles = spiral(*phase_space_args, **phase_space_kwargs)



    dose = np.zeros_like(density_array, dtype=np.double)
    N = np.array(dose.shape, dtype=np.double)
    offset = -spacing * N / 2.
    offset[2] = 0

    t0 = time.clock()
    print 'Done'
    print 'Starting simulation'
    for batch, i, n in particles:
#        particle = np.squeeze(batch[:, 0])
#        print is_intersecting_py(particle, N, spacing, offset)

#        pdb.set_trace()

        score_energy(batch, N, spacing, offset, material_array, density_array,
                     lut, dose)
        t1 = time.clock() - t0
        t2 = t1 * (n-i) / float(i + 1)
        p = round(float(i)/n*100, 1)
        print '{0}% Elapsed time is {1}. ETA in {2}'.format(p, human_time(t1),
                                                           human_time(t2))
    print 'Done'
    
#    print 'Saving data to {0}'.format(H5PATH)
#    saveobj = {'spacing': spacing,
#               'density_array': density_array,
#               'lut': lut,
#               'material_array': material_array,
#               'organ_array': organ_array,
#               'organ_map': organ_map,
#               'dose': dose}
#    for key, value in saveobj.items():
#        if isinstance(value, np.ndarray):
#            save_array_to_h5(H5PATH, value, key)
#        elif isinstance(value, dict):
#            save_dict_to_h5(H5PATH, value, key)
#    print 'Done'

    
    
#    dose *= (organ_array != 0)
    for i in range(9):
        plt.subplot(3,3,i+1)
        k = np.int(dose.shape[2] * (i+1) / 11.)
        plt.imshow(np.squeeze(dose[:, :, k]))
    plt.show()
    pdb.set_trace()






if __name__ == '__main__':
    test()