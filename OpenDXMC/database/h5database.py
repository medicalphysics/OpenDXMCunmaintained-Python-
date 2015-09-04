# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:16:46 2015

@author: erlean
"""

import numpy as np
import tables as tb
import itertools
import os

import logging
logger = logging.getLogger('OpenDXMC')


from opendxmc.study.simulation import Simulation
from opendxmc.materials import Material


class Database(object):
    def __init__(self, database_path=None):
        self.db_path = os.path.abspath(database_path)
        self.db_instance = None

    def open(self):
        if self.db_instance is not None:
            if self.db_instance.isopen:
                return
        self.db_instance = tb.open_file(self.db_path, mode='a',
                                        filters=tb.Filters(complevel=9))

    def close(self):
        if self.db_instance is not None:
            if self.db_instance.isopen:
                self.db_instance.close()

    def test_node(self, where, name):
        self.open()
        try:
            self.db_instance.get_node(where, name=name)
        except tb.NoSuchNodeError:
            return False
        return True

    def get_node(self, where, name, create=True, obj=None):
        self.open()
        try:
            node = self.db_instance.get_node(where, name=name)
        except tb.NoSuchNodeError:

            if not create:
                raise ValueError("Node {0} do not exist in {1}. Was not allowed to create a new node".format(name, where))

            if obj is None:
                node = self.db_instance.create_group(where, name,
                                                     createparents=True)
            elif isinstance(obj, np.recarray) or isinstance(obj, np.dtype):
                node = self.db_instance.create_table(where, name,
                                                     description=obj,
                                                     createparents=True)
            elif isinstance(obj, np.ndarray):
                node = self.db_instance.create_carray(where, name, obj=obj,
                                                      createparents=True)
            else:
                raise ValueError("Node {0} do not exist in {1}. Unable to create new node, did not understand obj type".format(name, where))

        return node

    def add_material(self, material, overwrite=True):
        mat_table = self.get_node('/', 'meta_materials', create=True,
                                  obj=material.numpy_dtype())

        # test for existing data
        matching_names = mat_table.get_where_list('name == b"{}"'.format(material.name))
        if len(matching_names) > 0:
            if not overwrite:
                raise ValueError('Material {} is already present i database'.format(material.name))
            else:
                logger.warning('Overwriting material {} already in database'.format(material.name))
        matching_names.sort()
        for ind in matching_names[::-1]:
            mat_table.remove_row(ind)
        mat_table.flush()

        if self.test_node('/attinuations', material.name):
            self.db_instance.remove_node('/attinuations', name=material.name,
                                         recursive=True)
        # adding descriptiondata
        mat_row = mat_table.row
        for key, value in zip(['name', 'density', 'organic'],
                              [material.name, material.density,
                               material.organic]):
            mat_row[key] = value
        mat_row.append()
        mat_table.flush()

        #adding arrays
        self.get_node('/attinuations', material.name, obj=material.attinuation)
        logger.info('Successfully wrote material {} to database'.format(material.name))
        self.close()


    def get_materials(self, material_names_list=None, organic_only=False):
        if not self.test_node('/', 'meta_materials'):
            logger.warning('There is no materials in database')
            return []

        materials = []
        mat_table = self.get_node('/', 'meta_materials', create=False)

        if material_names_list is None:
            material_names_list = [str(row['name'], encoding='utf-8') for row in mat_table]

        for row in mat_table:
            name = str(row['name'], encoding='utf-8')
            if name in material_names_list:
                if (organic_only and row['organic']) or not organic_only:
                    att_node = self.get_node('/attinuations', name)
                    materials.append(Material(name,
                                              density=row['density'],
                                              organic=row['organic'],
                                              attinuations=att_node.read()))
        self.close()
        return materials

    def add_simulation(self, simulation, overwrite=True):
        meta_table = self.get_node('/', 'meta_data',
                                   obj=simulation.numpy_dtype())
        #test for existing data
        matching_names = meta_table.get_where_list('name == b"{}"'.format(simulation.name))
        if len(matching_names) > 0:
            if not overwrite:
                raise ValueError('Simulation {} is already present i database'.format(simulation.name))
            else:
                logger.warning('Overwriting simulation {} already in database'.format(simulation.name))

        if self.test_node('/simulations', simulation.name):
            self.db_instance.remove_node('/simulations', name=simulation.name,
                                         recursive=True)

        for row in meta_table.where('name == b"{}"'.format(simulation.name)):
            for key, value in simulation.description.items():
                row[key] = value
            row.update()
            break
        else:
            row = meta_table.row
            for key, value in simulation.description.items():
                row[key] = value
            row.append()
        meta_table.flush()

#
#        if len(matching_names) >
#        matching_names.sort()
#        for ind in matching_names[::-1]:
#            meta_table.remove_row(ind)
#            meta_table.flush()


        # adding descriptiondata


        #adding arrays
        for key, value in itertools.chain(iter(simulation.arrays.items()),
                                          iter(simulation.tables.items())):
            if value is not None:
                self.get_node('/simulations/{0}'.format(simulation.name), key,
                              obj=value)
        logger.info('Successfully wrote simulation {} to database'.format(simulation.name))
        self.close()

    def get_simulation(self, name):
        meta_table = self.get_node('/', 'meta_data')
        simulation = Simulation(name)

        for row in meta_table.where('name == b"{}"'.format(name)):
            for key in meta_table.colnames:
                try:
                    setattr(simulation, key, row[key])
                except AssertionError:
                    pass
            break
        else:
            self.close()
            raise ValueError('No study named {}'.format(name))

        pat_node = self.get_node('/simulations', name, create=False)
        for data_node in pat_node:
            node_name = data_node._v_name
            setattr(simulation, node_name, data_node.read())
        self.close()
        return simulation

    def list_simulations(self):
        if not self.test_node('/', 'meta_data'):
            logger.warning('There is no simulations in database')
            self.close()
            return []
        meta_table = self.get_node('/', 'meta_data', create=False)
        names = [str(row['name'], encoding='utf-8') for row in meta_table]
        self.close()
        return names


    def __del__(self):
        self.close()


