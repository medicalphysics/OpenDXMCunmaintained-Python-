# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:16:46 2015

@author: erlean
"""

import numpy as np
import tables as tb
import itertools
import os

from opendxmc.study.simulation import Simulation, DESCRIPTION_RECARRAY
from opendxmc.materials import Material
from opendxmc.data import get_stored_materials

import logging
logger = logging.getLogger('OpenDXMC')


class Database(object):
    def __init__(self, database_path=None):
        self.db_path = os.path.abspath(database_path)
        self.db_instance = None
        self.init_new_database()

    def init_new_database(self):
        # setting up materials if not exist

        try:
            self.get_node('/', 'meta_materials', create=False, obj=None)
        except ValueError:
            logger.debug('Materials not found, attempting to import local materals')
            for m in get_stored_materials():
                self.add_material(m)
        if not self.test_node('/', 'meta_description'):
            logger.debug('Generating description data for simulation.')
            self.get_node('/', 'meta_description', create=True,
                          obj=DESCRIPTION_RECARRAY)

        logger.debug('Using database: {}'.format(self.db_path))

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

    def get_material(self, name):
        logger.debug('Attempting to read material {} from database'.format(name))
        if not self.test_node('/', 'meta_materials'):
            logger.warning('There is no materials in database')
            raise ValueError('No material {} in database'.format(name))

        mat_table = self.get_node('/', 'meta_materials', create=False)

        for row in mat_table.where('name == b"{}"'.format(name)):
            name = str(row['name'], encoding='utf-8')
            att_node = self.get_node('/attinuations', name)
            material = Material(name,
                                density=row['density'],
                                organic=row['organic'],
                                attinuations=att_node.read())
            break
        else:
            logger.warning('There is no material by name {} in database'.format(name))
            self.close()
            raise ValueError('No material {} in database'.format(name))

        self.close()
        logger.debug('Successfully read material {} from database'.format(name))
        return material

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

    def material_list(self):
        if not self.test_node('/', 'meta_materials'):
            logger.warning('There is no materials in database')
            self.close()
            return []
        meta_table = self.get_node('/', 'meta_materials', create=False)
        names = [str(row['name'], encoding='utf-8') for row in meta_table]
        self.close()
        return names


    def add_simulation(self, simulation, overwrite=True):
        meta_table = self.get_node('/', 'meta_data',
                                   obj=simulation.numpy_dtype())
        #test for existing data
        print(simulation.name)
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

        #adding arrays
        for key, value in iter(simulation.arrays.items()):
            if value is not None:
                self.get_node('/simulations/{0}'.format(simulation.name), key,
                              obj=value)
        for key, value in iter(simulation.volatiles.items()):
            if value is not None:
                self.get_node('/simulations/{0}/volatiles'.format(simulation.name),
                              key, obj=value)
        logger.info('Successfully wrote simulation {} to database'.format(simulation.name))
        self.close()

    def get_simulation(self, name, ignore_arrays=False, unsafe_read=True):
        logger.debug('Attempting to read simulation {} from database.'.format(name))
        if not self.test_node('/', 'meta_data'):
            logger.warning('There is no simulations in database')
            self.close()
            raise ValueError('No simulation by name {} in database'.format(name))

        meta_table = self.get_node('/', 'meta_data')


        for row in meta_table.where('name == b"{}"'.format(name)):
            if unsafe_read:
                description = {}
                for key in meta_table.colnames:
                    if isinstance(row[key], bytes):
                        description[key] = str(row[key], encoding='utf-8')
                    else:
                        description[key] = row[key]
                simulation = Simulation(name, description)
                break
            else:
                simulation = Simulation(name)
                for key in meta_table.colnames:
                    try:
                        setattr(simulation, key, row[key])
                    except AssertionError:
                        pass
                break
        else:
            self.close()
            logger.debug('Failed to read simulation {} from database. Simulation not found.'.format(name))
            raise ValueError('No study named {}'.format(name))

        if not ignore_arrays:
            pat_node = self.get_node('/simulations', name, create=False)
            for data_node in pat_node._f_walknodes('Array'):
                node_name = data_node._v_name
                logger.debug('Reading data node {}'.format(node_name))
                setattr(simulation, node_name, data_node.read())
        logger.debug('Successfully read simulation {} from database.'.format(name))
        self.close()
        return simulation

    def purge_simulation(self, name):
        logger.debug('Attempting to purge simulation {}'.format(name))
        sim_node = self.get_node('/simulations/', name, create=False)
        if self.test_node(sim_node, 'volatiles'):
            self.db_instance.remove_node(sim_node, 'volatiles', recursive=True)
        self.close()
        logger.debug('Purged simulation {}'.format(name))

    def update_simulation(self, description_dict):
        try:
            assert isinstance(description_dict, dict)
        except AssertionError:
            raise ValueError('Must provide a dictionary to update simulation metadata')
        name = description_dict.get('name', '')
        print('name', name)
        logger.debug('Attempting to update metadata for simulation {}'.format(name))
        meta_table = self.get_node('/', 'meta_data', create=False)
        description_array = self.get_node('/', 'meta_description', create=False).read()

        purge_simulation = False

        for row in meta_table.where('name == b"{}"'.format(name)):
            for item in meta_table.colnames:
                print(description_array['name'])
                ind = np.argwhere(description_array['name'] == bytes(item, encoding='utf-8'))[0]
                if description_array['editable'][ind]:
                    try:
                        value = description_dict[item]
                    except KeyError:
                        pass
                    else:
                        if row[item] != value:
                            row[item] = value
                            if description_array['volatile'][ind]:
                                purge_simulation = True
            row.update()
            break
        else:
            self.close()
            logger.warning('Could not update {0}. No simulation named {0} in database'.format(name))
            raise ValueError('No simulation named {} in database'.format(name))
        meta_table.flush()
        if purge_simulation:
            self.purge_simulation(name)
        self.close()
        logger.debug('Updated metadata for simulation {}'.format(name))

    def get_unique_simulation_name(self):
        logger.debug('Finding unique simulation name')
        if not self.test_node('/', 'meta_data'):
            logger.debug('No simulations in database, no names to test')
            self.close()
            return 'NewName'

        meta_table = self.get_node('/', 'meta_data', create=False)
        i = 1
        name = 'NewName'
        while True:
            for row in meta_table.where('name == b"{}"'.format(name)):
                break
            else:
                self.close()
                return name
            name = 'NewName{}'.format(i)
            i += 1

    def copy_simulation(self, source_name, dest_name=None):
        if dest_name is None:
            dest_name = self.get_unique_simulation_name()
        logger.debug('Attempting to copy simualtion {0} to {1}'.format(source_name, dest_name))
        if isinstance(dest_name, bytes):
            dest_name = str(dest_name, encoding='utf-8')
        else:
            dest_name = str(dest_name)
        dest_name = "".join([l for l in dest_name.split() if len(l) > 0])
        if len(dest_name) < 1:
            self.close()
            raise ValueError('Destination name must be valid and longer than zero')


        meta_table = self.get_node('/', 'meta_data', create=False)
        #test for names
        name_exist = [False, False]  # [source, dest]
        for row in meta_table:
            name = str(row['name'], encoding='utf-8')
            if not name_exist[0]:
                if name == source_name:
                    name_exist[0] = True
            if not name_exist[1]:
                if name == dest_name:
                    name_exist[1] = True

        if name_exist[0] and not name_exist[1]:
            assert self.test_node('/simulations', source_name)
            assert not self.test_node('/simulations', dest_name)
            logger.info('Database is copying simulation nodes, it may take a few seconds')
            self.db_instance.copy_node('/simulations', '/simulations', dest_name, source_name, recursive=True)
            new_row = meta_table.row
            for row in meta_table.where('name == b"{}"'.format(source_name)):
                name = str(row['name'], encoding='utf-8')
                if name == source_name:
                    for item in meta_table.colnames:
                        new_row[item] = row[item]
                    new_row['name'] = dest_name
                    new_row.append()
                    meta_table.flush()
                    break
        else:
            logger.warning('Unable to copy simulation {0} to {1}, error in destination name'.format(source_name, dest_name))
            self.close()
            return
        self.close()
        logger.warning('Copied simulation {0} to {1}'.format(source_name, dest_name))


    def simulation_list(self):
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


