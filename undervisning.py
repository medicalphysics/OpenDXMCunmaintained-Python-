# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:06:07 2015

@author: erlean
"""
import sys
import os
from opendxmc.database import Database
from opendxmc.data import get_stored_materials
from opendxmc.study import import_ct_series
from opendxmc.runner import ct_phase_space
from opendxmc.runner import ct_runner
from opendxmc.runner import obtain_ctdiair_conversion_factor
from opendxmc.runner import obtain_ctdiw_conversion_factor
from opendxmc.engine import score_energy
import logging
import pdb

from matplotlib import pylab as plt

import numpy as np

logger = logging.getLogger('OpenDXMC')
logger.addHandler(logging.StreamHandler())
logger.setLevel(10)


def test_database_init(db_path):
    db = Database(db_path)
    return db


def test_database_ct_import(db, im_path):
    sims = []
    for sim in import_ct_series([im_path]):
        db.add_simulation(sim)
        sims.append(sim)

    r_sims = [db.get_simulation(sim.name) for sim in sims]
    for sim in sims:
        assert sim.name in [r_sim.name for r_sim in r_sims]


def test_database(db_path, test_pat_path):
    # testing generation of database
    db = test_database_init(db_path)

#     test database import test ct images
    test_database_ct_import(db, test_pat_path)
    return db


def test_phase_space(db):
    names = db.simulations_list()
    assert len(names) > 0
    for name in names:
        sim = db.get_simulation(name)
        p_s = ct_phase_space(sim)
        logger.info('Test ct phase space not fully implemented')


def test_simulation(db_instance):
    sims = db_instance.simulation_list()
    if len(sims) == 0:
        raise ValueError('No patient in database')
    sim = db_instance.get_simulation(sims[0])
    materials = db_instance.get_materials(organic_only=False)


    sim.histories = 100
    sim.batch_size = 1000000
    sim.pitch = 0.9
    sim.ctdi_air100 = 8.75e-3
    sim.ctdi_w100 = 15. / 2.5 * 1e-3

    ct_runner(sim, materials)

    for m in materials:
        if m.name == 'air':
            air = m
        elif m.name == 'pmma':
            pmma = m

