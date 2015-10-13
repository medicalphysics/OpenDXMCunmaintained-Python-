# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:03:25 2015

@author: erlean
"""

import os
import sys
import numpy as np
from opendxmc.study import Simulation
from opendxmc.data.phantom_definitions import golem_organs
import logging
logger = logging.getLogger('OpenDXMC')

import pylab as plt
import pdb


PATH = os.path.join(os.path.dirname(sys.argv[0]), 'opendxmc', 'data', 'phantoms')

def read_phantoms():
    try:
        sim = read_golem()
    except FileNotFoundError:
        logger.debug('Golem phantom not found')
    else:
        yield sim





def read_golem():
    voxpath = os.path.join(PATH, 'golem', 'segm_golem')
    logger.debug('Attempting to read Golem phantom from {}'.format(voxpath))
    header_len = 4096
#    with open(voxpath) as f:
#        f.read(4096)
#        while True:
#            c = f.read(1)
#            if not c:
#                print ("End of file")
#                break
#            print("Read a character:", c)



    a = np.fromfile(voxpath, dtype=np.uint8)
#    pdb.set_trace()
#
#    with open(voxpath, 'rb') as fp:
#        c = fp.read()
#    a = np.array(c[header_len:], dtype='S', order='C').view(np.uint8)
#    print(a.min(), a.max())

    sim = Simulation('golem')
    sim.spacing = np.array([.208, .208, .8], dtype=np.double)
    sim.organ = np.rollaxis(np.reshape(a[header_len:], (-1, 256, 256)), 0, 3)[:,:,::-1]
    N = sim.organ.shape[2]
    organ_map={}
    organ_material_map={}
    for organ_number, organ, tissue in golem_organs():
        organ_map[organ_number] = organ
        organ_material_map[organ_number] = tissue
    sim.organ_map = organ_map
    sim.organ_material_map = organ_material_map
    sim.is_phantom = True
    sim.start_scan = 0
    sim.stop_scan = sim.spacing[2] * N
    sim.stop = sim.spacing[2] * N
    sim.data_center = np.array(sim.organ.shape) * sim.spacing / 2
    aec = np.ones((N, 2))
    aec[:, 0] = np.linspace(0, sim.stop, N)

    sim.exposure_modulation = aec
    return sim

