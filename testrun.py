# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:36:55 2015

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


def test_database_default_materials(db):
    materials = []

    for mat in get_stored_materials():
        materials.append(mat)
        db.add_material(mat)

    r_materials = [r_mat for r_mat in db.get_materials()]

    for mat in materials:
        assert mat.name in [r_mat.name for r_mat in r_materials]

    for r_mat in r_materials:
        for prop in ['density', 'organic', 'attinuation']:
            try:
                assert getattr(r_mat, prop) is not None
            except AssertionError as e:
                raise e


def test_database_ct_import(db, im_path):
    sims = []
    for sim in import_ct_series(im_path):
        db.add_simulation(sim)
        sims.append(sim)

    r_sims = [db.get_simulation(sim.name) for sim in sims]
    for sim in sims:
        assert sim.name in [r_sim.name for r_sim in r_sims]


def test_database(db_path, test_pat_path):
    # testing generation of database
    db = test_database_init(db_path)

#    test databse import of default materials
    test_database_default_materials(db)

#     test database import test ct images
    test_database_ct_import(db, test_pat_path)
    return db


def test_phase_space(db):
    names = db.list_simulations()
    assert len(names) > 0
    for name in names:
        sim = db.get_simulation(name)
        p_s = ct_phase_space(sim)
        logger.info('Test ct phase space not fully implemented')


def test_simulation(db_instance):
    sims = db_instance.list_simulations()
    if len(sims) == 0:
        raise ValueError('No patient in database')
    sim = db_instance.get_simulation(sims[0])
    materials = db_instance.get_materials(organic_only=False)


    sim.histories = 1
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



#    air_factor = []
#    w_factor = []
#    for i in range(10):
#        print('Running number', i+1)
#        obtain_ctdiw_conversion_factor(sim, air, pmma)
#        obtain_ctdiair_conversion_factor(sim, air)
#        air_factor.append(sim.conversion_factor_ctdiair)
#        w_factor.append(sim.conversion_factor_ctdiw)
#    print(air_factor)
#    print(w_factor)
#    pdb.set_trace()

    obtain_ctdiw_conversion_factor(sim, air, pmma)
#    obtain_ctdiair_conversion_factor(sim, air)
    print(sim.energy_imparted[128, 128, 10]*sim.conversion_factor_ctdiair)
    print(sim.energy_imparted[128, 128, 10]*sim.conversion_factor_ctdiw)
    for i in range(9):
        plt.subplot(3,3,i+1)
        k = int(i * (sim.energy_imparted.shape[2] - 1) / 9.)
        plt.imshow(np.squeeze(sim.energy_imparted[:,:,k]))
    db_instance.add_simulation(sim)
    plt.show(block=True)

    pdb.set_trace()



def test_suite():
    p = os.path.abspath('C://test//test.h5')
    test_pat = os.path.abspath('C://test//test_abdomen')

#     starting off clean
    try:
        os.remove(p)
    except FileNotFoundError:
        pass
    db = test_database(p, test_pat)
    del db


    db = Database(p)

    test_simulation(db)


if __name__ == '__main__':
    test_suite()


