# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:16:46 2015

@author: erlean
"""

import numpy as np
import tables as tb
import itertools
import os

from opendxmc.materials import Material
from opendxmc.data import get_stored_materials

import logging
logger = logging.getLogger('OpenDXMC')



PROPETIES_DICT_TEMPLATE = {
    #TAG: INIT VALUE, DTYPE, VOLATALE, EDITABLE, DESCRIPTION, ORDER
    'name': ['', 'a64', False, False, 'Simulation ID', 0],
    'scan_fov': [50., np.double, True, True, 'Scan field of view [cm]', 0],
    'sdd': [100., np.double, True, True, 'Source detector distance [cm]', 0],
    'detector_width': [0.06, np.double, True, True, 'Detector width [cm]', 0],
    'detector_rows': [64, np.int, True, True, 'Detector rows', 0],
    'collimation_width': [0.06 * 64, np.double, True, True, 'Total collimation width [cm]', 2],
    'al_filtration': [7., np.double, True, True, 'Filtration of primary beam [mmAl]', 0],
    'xcare': [False, np.bool, True, True, 'XCare', 0],
    'ctdi_air100': [0., np.double, True, True, 'CTDIair [mGy/100mAs]', 0],
    'ctdi_vol100': [0., np.double, True, True, 'CTDIvol [mGy/100mAs/pitch]', 2],
    'ctdi_w100': [0., np.double, True, True, 'CTDIw [mGy/100mAs]', 2],
    'kV': [120., np.double, True, True, 'Simulation tube potential [kV]', 0],
    'aquired_kV': [0., np.double, False, False, 'Images aquired with tube potential [kV]', 0],
    'region': ['abdomen', 'a64', False, False, 'Examination region', 0],
    # per 1000000 histories
    'conversion_factor_ctdiair': [0., np.double, True, False, 'CTDIair to dose conversionfactor', 0],
    # per 1000000 histories to dose
    'conversion_factor_ctdiw': [0., np.double, True, False, 'CTDIw to dose conversionfactor', 0],
    'is_spiral': [True, np.bool, True, True, 'Helical aqusition', 0],
    'pitch': [1, np.double, True, True, 'Pitch', 0],
    'exposures': [1200, np.int, True, True, 'Number of exposures in one rotation', 0],
    'histories': [100, np.int, True, True, 'Number of photon histories per exposure', 0],
    'batch_size': [500, np.int, True, True, 'Number of exposures in each calculation batch', 0],
    'start_scan': [0, np.double, False, False, 'CT scan start position [cm]', 0],
    'stop_scan': [0, np.double, False, False, 'CT scan stop position [cm]', 0],
    'start': [0, np.double, True, True, 'Start position [cm]', 2],
    'stop': [0, np.double, True, True, 'Stop position [cm]', 2],
    'step': [1, np.int, True, True, 'Sequential aqusition step size [cm]', 0],
    'start_at_exposure_no': [0, np.int, True, False, 'Start simulating exposure number', 0],
    'MC_finished': [False, np.bool, True, False, 'Simulation finished', 0],
    'MC_ready': [False, np.bool, True, False, 'Simulation ready', 0],
    'MC_running': [False, np.bool, True, False, 'Simulation is running', 0],
    'ignore_air': [False, np.bool, True, True, 'Ignore air material in simulation', 0],
    'spacing': [np.ones(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Image matrix spacing [cm]', 0],
    'shape': [np.ones(3, dtype=np.int), np.dtype((np.int, 3)), False, False, 'Image matrix dimensions', 0],
    'scaling': [np.ones(3, dtype=np.double), np.dtype((np.double, 3)), True, True, 'Calculation matrix scaling (stacked on image matrix prescaling)', 0],
    'import_scaling': [np.ones(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Image matrix prescaling when imported', 0],
    'image_orientation': [np.array([1, 0, 0, 0, 1, 0], dtype=np.double), np.dtype((np.double, 6)), False, False, 'Image patient orientation cosines', 0],
    'image_position': [np.zeros(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Image position (position of first voxel in volume) [cm]', 0],
    'data_center': [np.zeros(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Data collection center (relative to first voxel in volume) [cm]', 0],
    'is_phantom': [False, np.bool, False, False, 'Matematical phantom', 0],}

ARRAY_TEMPLATES = {
    # TAG: DTYPE, VOLATILE
    'ctarray': [np.int16, False],
    'exposure_modulation': [np.double, False],
    'organ': [np.uint8, False],
    'organ_map': [[('organ', np.uint8), ('organ_name', 'a128')], False],
    'organ_material_map': [[('organ', np.uint8), ('material_name', 'a128')], True],
    'energy_imparted': [np.double, True],
    'density': [np.double, True],
    'dose': [np.double, True],
    'material': [np.uint8, True],
    'material_map': [[('material', np.uint8), ('material_name', 'a128')], True],
   }


# Generating a recarray for SIMULATION_DESCRIPTION to insert in database
DESCRIPTION_RECARRAY = np.array([(k, v[2], v[3], v[4], v[5])
                                 for k, v in PROPETIES_DICT_TEMPLATE.items()],
                                dtype=[('name', 'a64'), ('volatale', np.bool),
                                       ('editable', np.bool),
                                       ('description', 'a128'), ('priority', np.int)]).view(np.recarray)

def SIMULATION_DTYPE():
    d = {'names': [],
         'formats': []}
    for key, value in PROPETIES_DICT_TEMPLATE.items():
        d['names'].append(key)
        d['formats'].append(value[1])
    return np.dtype(d)


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
            logger.debug('Materials not found, attempting to import local materials')
            for m in get_stored_materials():
                self.add_material(m)
        if not self.test_node('/', 'meta_description'):
            logger.debug('Generating description data for simulation.')
            self.get_node('/', 'meta_description', create=True,
                          obj=DESCRIPTION_RECARRAY)
        if not self.test_node('/', 'meta_data'):
            logger.debug('Generating meta data table for simulation.')
            self.get_node('/', 'meta_description', create=True,
                          obj=SIMULATION_DTYPE())

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

    def get_node(self, where, name, create=True, obj=None, overwrite=False):
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
                if obj.dtype.names is not None:
                    node = self.db_instance.create_table(where, name,
                                                         description=obj,
                                                         createparents=True)
                else:
                    node = self.db_instance.create_carray(where, name, obj=obj,
                                                          createparents=True)
            else:
                raise ValueError("Node {0} do not exist in {1}. Unable to create new node, did not understand obj type".format(name, where))
        else:
            if overwrite and create:
                self.db_instance.remove_node(where, name)
                return self.get_node(where, name, create, obj, overwrite)
        return node

    def remove_node(self, where, name):
        self.open()
        try:
            self.db_instance.remove_node(where, name=name, recursive=True)
        except tb.NoSuchNodeError:
            pass
        else:
            logger.debug('Deleted node {0}/{1}'.format(where, name))
        return

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


    def add_simulation(self, properties, array_dict=None, overwrite=True):
        if not self.test_node('/', 'meta_data'):
            meta_table = self.get_node('/', 'meta_data',
                                       obj=SIMULATION_DTYPE(), create=True)
        else:
            meta_table = self.get_node('/', 'meta_data', create=False)
        #test for existing data
        matching_names = meta_table.get_where_list('name == b"{}"'.format(properties['name']))
        if len(matching_names) > 0:
            if not overwrite:
                logger.info('Simulation {} not imported, already present i database'.format(properties['name']))
                self.close()
                return
            else:
                logger.info('Overwriting simulation {} already in database'.format(properties['name']))

        if self.test_node('/simulations', properties['name']):
            logger.debug('Deleting arrays for simulation {}'.format(properties['name']))
            self.db_instance.remove_node('/simulations', name=properties['name'],
                                         recursive=True)

        self.set_simulation_metadata(self, properties)

        for key, value in array_dict.items():
            self.set_simulation_array(properties['name'], value, key)

        logger.info('Successfully wrote simulation {} to database'.format(properties['name']))
        self.close()

    def remove_simulation(self, name):
        self.open()
        if not self.test_node('/', 'meta_data'):
            logger.info('There is no simulations in database')
            self.close()
            return

        index_to_remove = None
        meta_table = self.get_node('/', 'meta_data')
        for row in meta_table.where('name == b"{}"'.format(name)):
            index_to_remove = row.nrow
            break
        else:
            self.close()
            logger.info('There is no simulations named {} in database to remove'.format(name))
            return
        # removing table row
        if meta_table.nrows <= 1 and index_to_remove is not None:
            self.remove_node('/', 'meta_data')
            self.get_node('/', 'meta_data', obj=SIMULATION_DTYPE(), create=True)
        elif index_to_remove is not None:
            meta_table.remove_row(index_to_remove)

        # removing array data
        self.remove_node('/simulations', name)
        self.close()
        return

#    def get_simulation(self, name, ignore_arrays=False, unsafe_read=True):
#        logger.debug('Attempting to read simulation {} from database.'.format(name))
#        if not self.test_node('/', 'meta_data'):
#            logger.info('There is no simulations in database')
#            self.close()
#            raise ValueError('No simulation by name {} in database'.format(name))
#
#        meta_table = self.get_node('/', 'meta_data')
#
#
#        for row in meta_table.where('name == b"{}"'.format(name)):
#            if unsafe_read:
#                description = {}
#                for key in meta_table.colnames:
#                    if isinstance(row[key], bytes):
#                        description[key] = str(row[key], encoding='utf-8')
#                    else:
#                        description[key] = row[key]
#                simulation = Simulation(name, description)
#                break
#            else:
#                simulation = Simulation(name)
#                for key in meta_table.colnames:
#                    try:
#                        setattr(simulation, key, row[key])
#                    except AssertionError:
#                        pass
#                break
#        else:
#            self.close()
#            logger.debug('Failed to read simulation {} from database. Simulation not found.'.format(name))
#            raise ValueError('No study named {}'.format(name))
#
#        pat_node = self.get_node('/simulations', name, create=False)
#        if not ignore_arrays:
#            props = itertools.chain(pat_node._f_walknodes('Array'), pat_node._f_walknodes('Table'))
#        else:
#            aecnode_list = []
#            try:
#                aecnode_list.append(self.get_node(pat_node, 'exposure_modulation', create=False))
#            except ValueError:
#                pass
#            props = itertools.chain(aecnode_list, pat_node._f_walknodes('Table'))
#        for data_node in props:
##            for data_node in pat_node._f_walknodes():
#            node_name = data_node._v_name
#            logger.debug('Reading data node {}'.format(node_name))
#            setattr(simulation, node_name, data_node.read())
#
#        logger.debug('Successfully read simulation {} from database.'.format(name))
#        self.close()
#        return simulation

    def set_simulation_metadata(self, properties, purge=False, cancel_if_running=False):
        if 'name' not in properties:
            logger.debug('Unable to update simulation meta data, requires name key in update dictionary')
        self.open()
        meta_table = self.get_node('/', 'meta_data', create=False)
        for row in meta_table.where('name == b"{}"'.format(properties['name'])):
            if row['MC_running']:
                logger.debug('Could not update simulation {}, not allowed when MC is running'.format(properties['name']))
            for key, value in properties.items():
                if key in meta_table.colnames:
                    row[key] = value
            row.update()
            if purge:
                self.purge_simulation(properties['name'])
            break
        else:
            row = meta_table.row
            for key, value in properties.items():
                row[key] = value
            row.append()
        meta_table.flush()
        self.close()

    def set_simulation_array(self, name, array, array_name):
        if name not in ARRAY_TEMPLATES:
            logger.debug('Not allowed to write array {0} for simulation {1}. Allowed arrays are {2}'.format(array_name, name, ARRAY_TEMPLATES.keys()))
            return
        volatile = ARRAY_TEMPLATES[array_name][1]
        if volatile:
            node_path = '/simulations/{0}/volatiles'.format(name)
        else:
            node_path = '/simulations/{0}'.format(name)
        self.open()
        self.get_node(node_path, name, create=True, overwrite=True, obj=array)
        self.close()
        return

    def get_simulation_meta_data(self, name):
        self.open()
        if not self.test_node('/', 'meta_data'):
            logger.info('Could not get metadata for {}. No data in database'.format(name))
        meta_table = self.get_node('/', 'meta_data', create=False)
        for row in meta_table.where('name == b"{}"'.format(name)):
            data = {col_name:
                    (str(row[col_name], encoding='utf-8')
                    if (meta_table.coldtypes[col_name].char == 'S')
                    else row[col_name])
                    for col_name in meta_table.colnames}
            break
        else:
            logger.info('Could not get metadata for {}, simulation not in database'.format(name))
            self.close()
            raise ValueError('Could not get metadata for {}, simulation not in database')
        self.close()
        return data


    def get_simulation_array(self, name, array_name):
        self.open()
        if self.test_node('/simulations/{0}'.format(name), array_name):
            node = self.get_node('/simulations/{0}'.format(name),  array_name, create=False)
        elif self.test_node('/simulations/{0}/volatiles/'.format(name), array_name):
            node = self.get_node('/simulations/{0}/volatiles/'.format(name),  array_name, create=False)
        else:
            self.close()
            raise ValueError('No array named {0} for simulation{1}'.format(array_name, name))
            return
        arr = node.read()
        self.close()
        return arr

    def get_simulation_array_slice(self, name, array_name, index, orientation):
        self.open()
        if self.test_node('/simulations/{0}'.format(name), array_name):
            node = self.get_node('/simulations/{0}'.format(name),  array_name, create=False)
        elif self.test_node('/simulations/{0}/volatiles/'.format(name), array_name):
            node = self.get_node('/simulations/{0}/volatiles/'.format(name),  array_name, create=False)
        else:
            self.close()
            raise ValueError('No array named {0} for simulation{1}'.format(array_name, name))
            return None
        orientation %= 3
        index %= node.shape[orientation]
        if orientation == 1:
            arr = np.squeeze(node[:, index, :])
        elif orientation == 0:
            arr = np.squeeze(node[index, :, :])
        else:
            arr = np.squeeze(node[:,:,index])
        self.close()
        return arr

    def get_MCready_simulation(self):
        logger.debug('Attempting to find MC ready simulations from database.')
        if not self.test_node('/', 'meta_data'):
            logger.warning('There is no simulations in database')
            self.close()

        if not self.test_node('/', 'meta_data'):
            logger.info('Could not scan for MC jobs, no simulations in database')
            self.close()
            raise ValueError('No simulations in database')
        meta_table = self.get_node('/', 'meta_data', create=False)

        for row in meta_table.where('MC_ready & ~ MC_finished & ~ MC_running'):
            name = str(row['name'], encoding='utf-8')
            break
        else:
            self.close()
            logger.debug('No ready simulations found.')
            raise ValueError('No simulations ready')

        properties = self.get_simulation_meta_data(name)
        arrays = {}
        for key in ARRAY_TEMPLATES.keys():
            if ARRAY_TEMPLATES[key][1]:
                try:
                    arrays[key] = self.get_simulation_array(name, key)
                except ValueError:
                    pass

        self.close()
        return properties, arrays

    def purge_simulation(self, name):
        logger.debug('Attempting to purge simulation {}'.format(name))
        sim_node = self.get_node('/simulations/', name, create=False)
        if self.test_node(sim_node, 'volatiles'):
            self.db_instance.remove_node(sim_node, 'volatiles', recursive=True)
        self.close()
        logger.debug('Purged simulation {}'.format(name))

#    def update_simulation(self, description_dict, volatiles_dict=None,
#                          purge_volatiles=False, cancel_if_running=False):
#        try:
#            assert isinstance(description_dict, dict)
#        except AssertionError:
#            raise ValueError('Must provide a dictionary to update simulation metadata')
#        name = description_dict.get('name', '')
#        self.open()
#        logger.debug('Attempting to update metadata for simulation {}'.format(name))
#        meta_table = self.get_node('/', 'meta_data', create=False)
#        description_array = self.get_node('/', 'meta_description', create=False).read()
#
#        purge_simulation = False
#
#        got_row = False # fix since braking loop while updating table is not allowed
#        row_updated = False
#        for ind, row in enumerate(meta_table.where('name == b"{}"'.format(name))):
#            if cancel_if_running:
#                if row['MC_running']:
#                    self.close()
#                    logger.warning('Could not update data for simulation {}, simulation is running'.format(name))
#                    return
#            if ind == 0:
#                got_row = True
#
#                for item in meta_table.colnames:
#                    ind = np.argwhere(description_array['name'] == bytes(item, encoding='utf-8'))[0]
#                    if description_array['volatile'][ind]:
#                        try:
#                            value = description_dict[item]
#                        except KeyError:
#                            pass
#                        else:
#                            if isinstance(row[item], np.ndarray):
#                                is_equal = np.all(np.equal(row[item], value))
#                            else:
#                                is_equal = row[item] == value
#                            if not is_equal:
#                                row[item] = value
#                                row_updated = True
#                                logger.debug('Updated {0} value to {1} for simulation {2}'.format(item, value, name))
#                                if description_array['volatale'][ind]:
#                                    purge_simulation = True
#                if row_updated:
#                    row.update()
#
#        if not got_row:
#            self.close()
#            logger.warning('Could not update {0}. No simulation named {0} in database'.format(name))
#            return
##            raise ValueError('No simulation named {} in database'.format(name))
#        else:
#            meta_table.flush()
#
#        if purge_simulation and purge_volatiles:
#            self.purge_simulation(name)
#
#        if volatiles_dict is not None:
#            for key, value in volatiles_dict.items():
#                if self.test_node('/simulations/{}/volatiles'.format(name), key):
#                    self.remove_node('/simulations/{}/volatiles'.format(name), key)
#                    self.get_node('/simulations/{}/volatiles'.format(name), key, create=True, obj=value)
#                    logger.info('Updated {0} for simulation node {1}'.format(key, name))
#                else:
#                    self.get_node('/simulations/{}/volatiles'.format(name), key, create=True, obj=value)
#                    logger.info('Created {0} for simulation node {1}'.format(key, name))
#
#        self.close()
#        logger.debug('Updated metadata for simulation {}'.format(name))

    def get_unique_simulation_name(self, name=None):
        logger.debug('Finding unique simulation name')
        if not self.test_node('/', 'meta_data'):
            logger.debug('No simulations in database, no names to test')
            self.close()
            if not isinstance(name, str):
                return 'NewName'
            elif len(name) == 0:
                return 'NewName'
            else:
                return name


        meta_table = self.get_node('/', 'meta_data', create=False)
        i = 1
        if name is None:
            name = 'NewName'
        else:
            assert isinstance(name, str)
            name = "".join([l for l in name.split() if len(l) > 0])
            if len(name) == 0:
                name = 'NewName'
        while True:
            for row in meta_table.where('name == b"{}"'.format(name)):
                break
            else:
                self.close()
                return name
            name = '{0}{1}'.format(name, i)
            i += 1

    def copy_simulation(self, source_name, dest_name=None):
        dest_name = self.get_unique_simulation_name(dest_name)
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


