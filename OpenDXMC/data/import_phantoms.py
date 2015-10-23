# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:03:25 2015

@author: erlean
"""

import os
import sys
import numpy as np
from opendxmc.database.h5database import 
from opendxmc.data.phantom_definitions import golem_organs, vishum_organs, donna_organs, helga_organs, irene_organs, eva_organs, adam_organs, frank_organs, katja_organs, child_organs, baby_organs
from opendxmc.runner.ct_study_runner import ct_runner_validate_simulation
import logging
logger = logging.getLogger('OpenDXMC')


PATH = os.path.join(os.path.dirname(sys.argv[0]), 'opendxmc', 'data', 'phantoms')

def read_phantoms(materials=None):

    phantoms = {'golem': ('segm_golem', golem_organs, (2.08, 2.08, 8.), (256, 256, 220)),
                'VisHum': ('segm_vishum', vishum_organs, (.91, .94, 5.), (512, 512, 250)),
                'Donna':  ('segm_donna', donna_organs, (1.875, 1.875, 10.), (256, 256, 179)),
                'Helga':  ('segm_helga', helga_organs, (.98, .98, 10.), (512, 512, 114)),
                'Irene':  ('Irene', irene_organs, (1.875, 1.875, 5.), (132, 262, 348)),
                'Eva8bit': ('EVA_8bit', eva_organs, (1.6, 1.6, 2.), (256, 256, 900)),
                'Adam8bit': ('ADAM_8bit', adam_organs, (1.6, 1.6, 2.), (256, 256, 900)),
                'Frank': ('segm_frank', frank_organs, (.742, .742, 5.), (512, 512, 193)),
                'Katja': ('Katja', katja_organs, (1.775, 1.775, 4.84), (150, 299, 348)),
                'Child': ('segm_child', child_organs, (1.54, 1.54, 8), (256, 256, 144)),
                'Babynew': ('babynew_May2003', baby_organs, (.85, .85, 4.), (138, 267, 142)),
                }


    for key, value in phantoms.items():
        path = os.path.join(PATH, key, value[0])
        try:
            sim = read_voxels(path, key, *value[1:])
        except FileNotFoundError as e:
            logger.debug('{0} phantom not found in {1}'.format(key, path))
#            raise e
        else:
            logger.debug('{0} phantom read successfully'.format(key))
            if materials is not None:
                try:
                    ct_runner_validate_simulation(sim, materials)
                except AssertionError or ValueError:
                    logger.debug('Failed to create material mapping for phantom {0}'.format(key))
                else:
                    logger.debug('Generated material mapping for phantom {0}'.format(key))
            yield sim



def read_voxels(path, name, organ_func, spacing, shape, header_len=4096):
    voxpath = path
    logger.debug('Attempting to read {0} phantom from {1}'.format(name, voxpath))

    a = np.fromfile(voxpath, dtype=np.uint8)
    sim = {}Simulation(name)
    sim.spacing = np.array(spacing, dtype=np.double) / 10.
    sim.organ = np.rollaxis(np.reshape(a[header_len:], (-1, shape[0], shape[1])), 0, 3)#[:,:,::-1]
    sim.shape = np.array(sim.organ.shape)
    N = sim.organ.shape[2]
    organ_map={}
    organ_material_map={}
    organ_values = []
    for i in range(N):
        # we do loops due to memory constrains on np.unique
        organ_values += list(np.unique(sim.organ[:, : ,i]))

    organ_dict = {organ_number: (organ, tissue) for organ_number, organ, tissue in organ_func()}
    for organ_number in set(organ_values):
        if organ_number in organ_dict:
            organ_str, tissue_str = organ_dict[organ_number]
            organ_map[organ_number] = organ_str
            organ_material_map[organ_number] = tissue_str
        else:
            organ_map[organ_number] = 'Unknown as air {}'.format(organ_number)
            organ_material_map[organ_number] = 'air'

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


    sim.scaling=np.array([2,2,1], dtype=np.double)

    return sim


