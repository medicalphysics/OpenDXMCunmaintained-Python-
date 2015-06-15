# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:07:57 2015

@author: ERLEAN
"""

import numpy as np
import time
import fuzzy_time
from tube.tungsten import specter
from phase_space.ct import spiral
from phantoms.golem import read_golem
from engine._interaction_func import score_energy


import pdb

def test():


    spacing, density_array, lut, material_array, organ_array, organ_map = read_golem()

    phase_space_args = (50., 93., 3.98)

    phase_space_kwargs = {'pitch': 1.2,
                          'start': 0.,
                          'stop': spacing[2] * density_array.shape[2],
                          'exposures': 1000,
                          'histories': 100,
                          'energy_specter': specter(120., angle_deg=12.,
                                                    filtration_materials='Al',
                                                    filtration_mm=4.),
                          'batch_size': 10000,
                        }

    particles = spiral(*phase_space_args, **phase_space_kwargs)



    dose = np.zeros_like(density_array, dtype=np.double)
    N = np.array(dose.shape, dtype=np.double)
    offset = -spacing * N / 2.
    offset[2] = 0

    t0 = time.clock()
    for batch, i, n in particles:
        pdb.set_trace()
        score_energy(batch, N, spacing, offset, material_array, density_array,
                     lut, dose)
        t1 = time.clock()
        t2 = (t1-t0)*(i+1-n)/float(i+1)
        print '{0}% Elapsed time is: {1}. ETA: {2}'.format(float(i)/n*100, fuzzy_time.human_time(t1-t0), fuzzy_time.human_time(t2))







if __name__ == '__main__':
    test()