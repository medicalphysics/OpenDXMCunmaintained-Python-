# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:16:46 2015

@author: erlean
"""

import numpy as np
import tables as tb
import os

from opendxmc.materials import Material
from opendxmc.database.import_materials import get_stored_materials

import logging
logger = logging.getLogger('OpenDXMC')

PROPETIES_DICT_TEMPLATE_GROUPING = {0: 'Simulation',
                                    1: 'Scanner geometry',
                                    2: 'Aqusition Parameters',
                                    3: 'MonteCarlo Settings',
                                    4: 'Dose Calibration Settings',
                                    5: 'Miscellaneous'}


PROPETIES_DICT_TEMPLATE = {
    #TAG: INIT VALUE, DTYPE, VOLATALE, EDITABLE, DESCRIPTION, ORDER, GROUPING
    'name': ['', np.dtype('a64'), False, False, 'Simulation ID', 0, 0],
    'scan_fov': [50., np.dtype(np.double), True, True, 'Scan field of view [cm]', 0, 1],
    'sdd': [110.,  np.dtype(np.double), True, True, 'Source detector distance [cm]', 0, 1],
    'detector_width': [0.06,  np.dtype(np.double), True, True, 'Detector width [cm]', 2, 1],
    'detector_rows': [64, np.dtype(np.int), True, True, 'Detector rows', 0, 1],
    'collimation_width': [0.06 * 64,  np.dtype(np.double), True, True, 'Total collimation width [cm]', 0, 1],
    'al_filtration': [7., np.dtype(np.double), True, True, 'Filtration of primary beam [mmAl]', 0, 2],
    'xcare': [False, np.dtype(np.bool), True, True, 'XCare [NOT IMPLEMENTED YET]', 0, 2],
    'ctdi_air100': [0., np.dtype(np.double), True, True, 'CTDIair [mGy/100mAs]', 0, 4],
    'ctdi_vol100': [0., np.dtype(np.double), True, True, 'CTDIvol [mGy/100mAs/pitch]', 2, 4],
    'ctdi_w100': [0., np.dtype(np.double), True, True, 'CTDIw [mGy/100mAs]', 2, 4],
    'ctdi_phantom_diameter': [32., np.dtype(np.double), True, True, 'CTDI phantom diameter [cm]', 2, 4],
    'kV_A': [120., np.dtype(np.double), True, True, 'Simulation tube potential (Tube A) [kV]', 0, 3],
    'kV_B': [120., np.dtype(np.double), True, True, 'Simulation tube potential (Tube B) [kV]', 0, 3],
    'tube_weight_A': [1., np.dtype(np.double), True, True, 'Tube weight factor (Tube A)', 0, 3],
    'tube_weight_B': [0., np.dtype(np.double), True, True, 'Tube weight factor (Tube B)', 0, 3],
    'use_tube_B': [False, np.dtype(np.bool), True, True, 'Use second x-ray tube', 0, 3],
    'aquired_kV': [0., np.dtype(np.double), False, False, 'Images aquired with tube potential [kV]', 0, 2],
    'region': ['abdomen', np.dtype('a64'), False, False, 'Examination region', 0, 5],
    # per 1000000 histories
    'conversion_factor_ctdiair': [0., np.dtype(np.double), True, False, 'CTDIair to dose conversion factor', 0, 4],
    # per 1000000 histories to dose
    'conversion_factor_ctdiw': [0., np.dtype(np.double), True, False, 'CTDIw to dose conversion factor', 0, 4],
    'is_spiral': [True, np.dtype(np.bool), True, True, 'Helical aqusition', 0, 3],
    'pitch': [1, np.dtype(np.double), True, True, 'Pitch', 0, 3],
    'exposures': [1200, np.dtype(np.int), True, True, 'Number of exposures in one rotation', 0, 3],
    'histories': [1000, np.dtype(np.int), True, True, 'Number of photon histories per exposure', 0, 3],
    'start_scan': [0, np.dtype(np.double), False, False, 'CT scan start position [cm]', 0, 2],
    'stop_scan': [0, np.dtype(np.double), False, False, 'CT scan stop position [cm]', 0, 2],
    'start': [0, np.dtype(np.double), True, True, 'Start position [cm]', 2, 3],
    'stop': [0, np.dtype(np.double), True, True, 'Stop position [cm]', 2, 3],
    'step': [1, np.dtype(np.int), True, True, 'Sequential aqusition step size [cm]', 0, 3],
    'start_at_exposure_no': [0, np.dtype(np.int), True, False, 'Start simulating exposure number', 0, 3],
    'MC_finished': [False, np.dtype(np.bool), True, False, 'Simulation finished', 0, 3],
    'MC_ready': [False, np.dtype(np.bool), True, False, 'Simulation ready', 0, 3],
    'MC_running': [False, np.dtype(np.bool), True, False, 'Simulation is running', 0, 3],
    'ignore_air': [True, np.dtype(bool), True, True, 'Ignore air material in simulation', 0, 3],
    'spacing': [np.ones(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Image matrix spacing [cm]', 0, 2],
    'shape': [np.ones(3, dtype=np.int), np.dtype((np.int, 3)), False, False, 'Image matrix dimensions', 0, 2],
    'scaling': [np.ones(3, dtype=np.double), np.dtype((np.double, 3)), True, True, 'Calculation matrix scaling (stacked on image matrix prescaling)', 0, 3],
    'import_scaling': [np.ones(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Image matrix prescaling when imported', 0, 2],
    'image_orientation': [np.array([1, 0, 0, 0, 1, 0], dtype=np.double), np.dtype((np.double, 6)), False, False, 'Image patient orientation cosines', 0, 2],
    'image_position': [np.zeros(3, dtype=np.double), np.dtype((np.double, 3)), False, False, 'Image position (position of first voxel in volume) [cm]', 0, 2],
    'data_center': [np.zeros(3, dtype=np.double), np.dtype((np.double, 3)), True, True, 'Data collection center (relative to first voxel in volume) [cm]', 0, 3],
    'is_phantom': [False, np.dtype(np.bool), False, False, 'Matematical phantom', 0, 5],
    'use_siddon': [False, np.dtype(np.bool), True, True, 'Use Siddon tracking, default is Woodcock tracking', 0, 3],
    'anode_angle': [12., np.dtype(np.double), True, True, 'Angle of anode in x-ray tube [deg]', 0, 3],
    'tube_start_angle_A': [0, np.dtype(np.double), True, True, 'Tube start angle (Tube A) [deg]', 0, 3],
    'tube_start_angle_B': [0, np.dtype(np.double), True, True, 'Tube start angle (Tube B) [deg]', 0, 3],                           
    'bowtie_radius': [15, np.dtype(np.double), True, True, 'Bowtie filter radius', 0, 3],
    'bowtie_distance': [10, np.dtype(np.double), True, True, 'Bowtie filter distance factor', 2, 3],
    }

ARRAY_TEMPLATES = {
    # TAG: DTYPE, VOLATILE, PRERECORD_VALUES
    'ctarray': [np.int16, False, True],
    'exposure_modulation': [np.double, False, True],
    'organ': [np.uint8, False, True],
    'organ_map': [[('organ', np.uint8), ('organ_name', 'a128')], False, False],
    'organ_material_map': [[('organ', np.uint8), ('material_name', 'a128')], False, False],
    'energy_imparted': [np.double, True, True],
    'density': [np.double, True, True],
    'dose': [np.double, True, True],
    'material': [np.uint8, True, True],
    'material_map': [[('material', np.uint8), ('material_name', 'a128')], True, False],
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
    def __init__(self, database_path):
        self.db_path = os.path.abspath(database_path)
        self.db_instance = None
        self.filters = tb.Filters(complevel=1, complib='blosc', fletcher32=False, shuffle=False)
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
                                        filters=self.filters)

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
        if name == '':
            return
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

        self.set_simulation_metadata(properties)

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

        # removing array data
        self.remove_node('/simulations', name)

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

        self.close()
        logger.debug('Removed simulation {}'.format(name))
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
        breaker = False
        for row in meta_table.where('name == b"{}"'.format(properties['name'])):
            if not breaker:
                if row['MC_running'] and cancel_if_running:
                    logger.debug('Could not update simulation {}, not allowed when MC is running'.format(properties['name']))
                    self.close()
                    return
                for key, value in properties.items():
                    if key in meta_table.colnames:
                        row[key] = value
                row.update()

                breaker = True
        if not breaker:
            row = meta_table.row
            for key, value in properties.items():
                row[key] = value
            row.append()
        meta_table.flush()
        if purge:
            self.purge_simulation(properties['name'])
        self.close()

    def set_simulation_array(self, name, array, array_name):
        if array_name not in ARRAY_TEMPLATES:
            logger.debug('Not allowed to write array {0} for simulation {1}. Allowed arrays are {2}'.format(array_name, name, ARRAY_TEMPLATES.keys()))
            return
        if array is None:
            logger.debug('Not allowed to write array {0} for simulation {1}. Array is None'.format(array_name, name))
            return
        volatile = ARRAY_TEMPLATES[array_name][1]
        if volatile:
            node_path = '/simulations/{0}/volatiles'.format(name)
        else:
            node_path = '/simulations/{0}'.format(name)
        self.open()

        if isinstance(array, dict):
            array = Validator().validate_structured_array(array, array_name)

        self.get_node(node_path, array_name, create=True, overwrite=True, obj=array)

        if ARRAY_TEMPLATES[array_name][2]:
            ex_table = self.get_node('/simulations/{0}'.format(name), '_array_extreme_values', create=True, obj=np.dtype([('name', 'a64'), ('min', np.double), ('max', np.double)]))
            arr_names = [str(row['name'], encoding='utf-8') for row in ex_table]
            if array_name in arr_names:
                index = arr_names.index(array_name)
                ex_table[index] = [array_name, array.min(), array.max()]
            else:
                ex_table.append([(array_name, array.min(), array.max())])
            ex_table.flush()


        logger.debug('Wrote array {0} to database for simulation {1}'.format(array_name, name))
        self.close()
        return

    def get_simulation_metadata(self, name):
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

    def get_simulation_array_bytescaled(self, name, array_name, amin=0., amax=1., minmax_is_modifier=False):
        self.open()
        if self.test_node('/simulations/{0}'.format(name), array_name):
            node = self.get_node('/simulations/{0}'.format(name),  array_name, create=False)
        elif self.test_node('/simulations/{0}/volatiles/'.format(name), array_name):
            node = self.get_node('/simulations/{0}/volatiles/'.format(name),  array_name, create=False)
        else:
            self.close()
            raise ValueError('No array named {0} for simulation{1}'.format(array_name, name))
            return

        if minmax_is_modifier:
            minmod = amin
            maxmod = amax
            amin, amax = 0, 0
        else:
            minmod = 1.
            maxmod = 1.

        if amin == amax:
            ex_table = self.get_node('/simulations/{0}'.format(name), '_array_extreme_values', create=False)

            arr_names = [str(row['name'], encoding='utf-8') for row in ex_table]
            if array_name in arr_names:
                index = arr_names.index(array_name)
                _, amin, amax = list(ex_table[index])
            else:
                raise ValueError('Array {} in database have no recorded min or max values, remove simulation {} from database.'.format(array_name, name))

        amin *= minmod
        amax *= maxmod

        arr = np.empty(node.shape, dtype=np.ubyte)

        r_divisor = 255./(amax-amin)
        for ind, r in enumerate(node):
            arr[ind, ...] = np.clip((r-amin)*r_divisor, 0, 255)
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

        for row in meta_table.where('MC_ready'):
            name = str(row['name'], encoding='utf-8')
            break
        else:
            self.close()
            logger.debug('No ready simulations found.')
            raise ValueError('No simulations ready')

        properties = self.get_simulation_metadata(name)
        arrays = {}
        for key in ARRAY_TEMPLATES.keys():
            if not ARRAY_TEMPLATES[key][1]:
                try:
                    arrays[key] = self.get_simulation_array(name, key)
                except ValueError:
                    pass
        if properties['start_at_exposure_no'] != 0:
            arrays['energy_imparted'] = self.get_simulation_array(name, 'energy_imparted')
        self.close()
        return properties, arrays

    def purge_simulation(self, name):
        logger.debug('Attempting to purge simulation {}'.format(name))
        try:
            sim_node = self.get_node('/simulations/', name, create=False)
        except ValueError:
            logger.debug('Error in array data for {}, removing simulation.'.format(name))
            self.remove_simulation(name)
            return



        if self.test_node(sim_node, 'volatiles'):
            self.db_instance.remove_node(sim_node, 'volatiles', recursive=True)
        if self.test_node('/', 'meta_data'):
            meta_table = self.get_node('/', 'meta_data', create=False)
            for row in meta_table.where('name == b"{}"'.format(name)):
                row['MC_finished'] = False
                row['conversion_factor_ctdiair'] = 0
                row['conversion_factor_ctdiw'] = 0
                row['start_at_exposure_no'] = 0
                row.update()
            meta_table.flush()
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
                return 'newname'
            elif len(name) == 0:
                return 'newname'
            else:
                return name


        meta_table = self.get_node('/', 'meta_data', create=False)
        i = 1
        if name is None:
            name = 'newname'
        else:
            assert isinstance(name, str)
            name = "".join([l for l in name.split() if len(l) > 0])
            if len(name) == 0:
                name = 'newname'
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


class Validator(object):
    def __init__(self):
        self._pt = PROPETIES_DICT_TEMPLATE
        self._at = ARRAY_TEMPLATES

        self._props = {key: value[0] for key, value in self._pt.items()}
        self._arrays = {key: None for key in self._at.keys()}

    def reset(self):
        self.set_data(None, True)

    def set_data(self, props=None, reset=True):
        if reset:
            self._props = {key: value[0] for key, value in self._pt.items()}
            self._arrays = {key: None for key in self._at.keys()}
        valid_attrs = list(self._pt.keys()) + list(self._at.keys())
        if props is not None:
            if reset:
                assert 'name' in props
            prop_list = [(key, value) for key, value in props.items()]
            prop_list.sort(key=lambda x: self._pt[x[0]][5] if x[0] in self._pt else 0)
            for key, value in prop_list:
                if key in valid_attrs:
                    setattr(self, key, value)
                else:
                    logger.debug('Error in propety {0} for simulation {1}'.format(key, props['name']))

    def get_data(self):
        return self._props, {key: value for key, value in self._arrays.items() if value is not None}

    def string_validator(self, value, strict=False):
        if isinstance(value, bytes):
            value = str(value, encoding='utf-8')
        else:
            value = str(value)
        if strict:
            name = "".join([l for l in value.split() if len(l) > 0])
        else:
            name = value
        assert len(name) > 0
        return name.lower()

    def float_validator(self, value, gt_0=False, cut=0.):
        val = float(value)
        if gt_0:
            assert val >= cut
        return val
    def int_validator(self, value, gt_0=False):
        val = int(value)
        if gt_0:
            assert val >= 0
        return val
    def bool_validator(self, value):
        assert isinstance(value, bool)
        return value

    def validate_structured_array(self, value, name):
        if value is None:
            return None

        if isinstance(value, dict):
            value_rec = np.recarray((len(value),),
                                    dtype=self._at[name][0])
            for ind, item in enumerate(value.items()):
                value_rec[ind] = item
            return value_rec
        elif isinstance(value, np.ndarray):
            value_rec = np.recarray((len(value),),
                                    dtype=self._at[name][0])
            assert value.dtype.names is not None
            c_title = self._at[name][0].names
            for var_name in c_title:
                assert var_name in value.dtype.names
            for i in range(len(value)):
                value_rec[i] = (value[c_title[0]][i], value[c_title[1]][i])
        else:
            raise ValueError('Mapping must be a structured array with correct keys or a dict')
        return value_rec

    def string_to_array_converter(self, name, string):
        dtype = self._pt[name][1]

        if len(dtype.shape) > 0:
            string = " ".join(string.split(":"))
            string = " ".join(string.split(","))
            string = " ".join(string.split(";"))
            val = np.asarray(self._pt[name][0])
            teller = 0
            for s in string.split():
                try:
                    val[teller] = s
                except ValueError:
                    pass
                else:
                    teller +=1
                if teller >= val.shape[0]:
                    break
            return val
        raise AssertionError




    @property
    def name(self):
        return self._props['name']

    @name.setter
    def name(self, value):
        self._props['name'] = self.string_validator(value, True)

    @property
    def scan_fov(self):
        return self._props['scan_fov']
    @scan_fov.setter
    def scan_fov(self, value):
        self._props['scan_fov'] = self.float_validator(value, True)

    @property
    def sdd(self):
        return self._props['sdd']
    @sdd.setter
    def sdd(self, value):
        self._props['sdd'] = self.float_validator(value, True)

    @property
    def detector_width(self):
        return self._props['detector_width']
    @detector_width.setter
    def detector_width(self, value):
        self._props['detector_width'] = self.float_validator(value, True)
        self._props['collimation_width'] = self.detector_width * self.detector_rows

    @property
    def detector_rows(self):
        return self._props['detector_rows']
    @detector_rows.setter
    def detector_rows(self, value):
        self._props['detector_rows'] = self.int_validator(value, True)
        self._props['collimation_width'] = self.detector_width * self.detector_rows

    @property
    def collimation_width(self):
        return self._props['collimation_width']
    @collimation_width.setter
    def collimation_width(self, value):
        self._props['collimation_width'] = self.float_validator(value, True)
        self._props['detector_width'] = self.collimation_width / self.detector_rows

    @property
    def al_filtration(self):
        return self._props['al_filtration']
    @al_filtration.setter
    def al_filtration(self, value):
        self._props['al_filtration'] = self.float_validator(value, True)

    @property
    def xcare(self):
        return self._props['xcare']
    @xcare.setter
    def xcare(self, value):
        self._props['xcare'] = self.bool_validator(value)

    @property
    def ctdi_air100(self):
        return self._props['ctdi_air100']
    @ctdi_air100.setter
    def ctdi_air100(self, value):
        self._props['ctdi_air100'] = self.float_validator(value, True)

    @property
    def ctdi_phantom_diameter(self):
        return self._props['ctdi_phantom_diameter']
    @ctdi_phantom_diameter.setter
    def ctdi_phantom_diameter(self, value):
        self._props['ctdi_phantom_diameter'] = self.float_validator(value, True)
        assert self._props['ctdi_phantom_diameter'] >= 10.

    @property
    def ctdi_vol100(self):
        return self._props['ctdi_vol100']
    @ctdi_vol100.setter
    def ctdi_vol100(self, value):
        self._props['ctdi_vol100'] = self.float_validator(value, True)
        self._props['ctdi_w100'] = self._props['ctdi_vol100'] * self.pitch

    @property
    def ctdi_w100(self):
        return self._props['ctdi_w100']
    @ctdi_w100.setter
    def ctdi_w100(self, value):
        self._props['ctdi_w100'] = self.float_validator(value, True)
        self._props['ctdi_vol100'] = self._props['ctdi_w100'] / self.pitch

    @property
    def aquired_kV(self):
        return self._props['aquired_kV']
    @aquired_kV.setter
    def aquired_kV(self, value):
        self._props['aquired_kV'] = self.float_validator(value, True)

    @property
    def kV_A(self):
        return self._props['kV_A']
    @kV_A.setter
    def kV_A(self, value):
        self._props['kV_A'] = self.float_validator(value, True, 40)
    @property
    def kV_B(self):
        return self._props['kV_B']
    @kV_B.setter
    def kV_B(self, value):
        self._props['kV_B'] = self.float_validator(value, True, 40)

    @property
    def use_tube_B(self):
        return self._props['use_tube_B']
    @use_tube_B.setter
    def use_tube_B(self, value):
        self._props['use_tube_B'] = self.bool_validator(value)

    @property
    def tube_weight_A(self):
        return self._props['tube_weight_A']
    @tube_weight_A.setter
    def tube_weight_A(self, value):
        self._props['tube_weight_A'] = self.float_validator(value, True, 0.0)
    @property
    def tube_weight_B(self):
        return self._props['tube_weight_B']
    @tube_weight_B.setter
    def tube_weight_B(self, value):
        self._props['tube_weight_B'] = self.float_validator(value, True, 0.0)



    @property
    def region(self):
        return self._props['region']
    @region.setter
    def region(self, value):
        self._props['region'] = self.string_validator(value)

    @property
    def conversion_factor_ctdiair(self):
        return self._props['conversion_factor_ctdiair']
    @conversion_factor_ctdiair.setter
    def conversion_factor_ctdiair(self, value):
        self._props['conversion_factor_ctdiair'] = self.float_validator(value, True)

    @property
    def conversion_factor_ctdiw(self):
        return self._props['conversion_factor_ctdiw']
    @conversion_factor_ctdiw.setter
    def conversion_factor_ctdiw(self, value):
        self._props['conversion_factor_ctdiw'] = self.float_validator(value, True)


    @property
    def is_spiral(self):
        return self._props['is_spiral']
    @is_spiral.setter
    def is_spiral(self, value):
        self._props['is_spiral'] = self.bool_validator(value)

    @property
    def pitch(self):
        return self._props['pitch']
    @pitch.setter
    def pitch(self, value):
        self._props['pitch'] = self.float_validator(value, True)
        self.ctdi_w100 = self._props['ctdi_w100']

    @property
    def exposures(self):
        return self._props['exposures']
    @exposures.setter
    def exposures(self, value):
        self._props['exposures'] = self.int_validator(value, True)

    @property
    def histories(self):
        return self._props['histories']
    @histories.setter
    def histories(self, value):
        self._props['histories'] = self.int_validator(value, True)

    @property
    def batch_size(self):
        return self._props['batch_size']
    @batch_size.setter
    def batch_size(self, value):
        assert int(value) > 0
        self._props['batch_size'] = self.int_validator(value, True)

    @property
    def start_scan(self):
        return self._props['start_scan']
    @start_scan.setter
    def start_scan(self, value):
        self._props['start_scan'] = self.float_validator(value)

    @property
    def stop_scan(self):
        return self._props['stop_scan']
    @stop_scan.setter
    def stop_scan(self, value):
        self._props['stop_scan'] = self.float_validator(value)

    @property
    def start(self):
        return self._props['start']
    @start.setter
    def start(self, value):
        vval = self.float_validator(value)
        vval = min([self.stop_scan, vval])
        vval = max([self.start_scan, vval])
        
        self._props['start'] = vval
        


    @property
    def stop(self):
        return self._props['stop']
    @stop.setter
    def stop(self, value):
        vval = self.float_validator(value)
        vval = min([self.stop_scan, vval])
        vval = max([self.start_scan, vval])
        
        self._props['stop'] = vval
        
    
        

    @property
    def step(self):
        return self._props['step']
    @step.setter
    def step(self, value):
        self._props['step'] = self.float_validator(value, True)

    @property
    def start_at_exposure_no(self):
        return self._props['start_at_exposure_no']
    @start_at_exposure_no.setter
    def start_at_exposure_no(self, value):
        self._props['start_at_exposure_no'] = self.int_validator(value, True)

    @property
    def MC_finished(self):
        return self._props['MC_finished']
    @MC_finished.setter
    def MC_finished(self, value):
        self._props['MC_finished'] = self.bool_validator(value)

    @property
    def MC_running(self):
        return self._props['MC_running']
    @MC_running.setter
    def MC_running(self, value):
        self._props['MC_running'] = self.bool_validator(value)

    @property
    def MC_ready(self):
        return self._props['MC_ready']
    @MC_ready.setter
    def MC_ready(self, value):
        self._props['MC_ready'] = self.bool_validator(value)

    @property
    def spacing(self):
        return self._props['spacing']
    @spacing.setter
    def spacing(self, value):
        if isinstance(value, np.ndarray):
            self._props['spacing'] = value.astype(np.double)
        elif isinstance(value, str):
            self._props['spacing'] = self.string_to_array_converter('spacing', value)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['spacing'] = value.astype(np.double)
            assert np.all(self._props['spacing'] > 0)

    @property
    def shape(self):
        return self._props['shape']
    @shape.setter
    def shape(self, value):
        if isinstance(value, np.ndarray):
            self._props['shape'] = value.astype(np.int)
        elif isinstance(value, str):
            self._props['shape'] = self.string_to_array_converter('shape', value)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['shape'] = value.astype(np.int)
        assert np.all(self._props['shape'] > 0)


    @property
    def import_scaling(self):
        return self._props['import_scaling']
    @import_scaling.setter
    def import_scaling(self, value):
        if isinstance(value, np.ndarray):
            assert len(value.shape) == 1
            assert value.shape[0] == 3
            self._props['import_scaling'] = value.astype(np.double)
        elif isinstance(value, str):
            self._props['import_scaling'] = self.string_to_array_converter('import_scaling', value)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == 3
            assert len(value) == 3
            self._props['import_scaling'] = value.astype(np.double)
        assert np.all(self._props['import_scaling'] > 0)

    @property
    def scaling(self):
        return self._props['scaling']
    @scaling.setter
    def scaling(self, value):
        if isinstance(value, np.ndarray):
            assert len(value.shape) == 1
            assert value.shape[0] == 3
            self._props['scaling'] = value.astype(np.double)
        elif isinstance(value, str):
            self._props['scaling'] = self.string_to_array_converter('scaling', value)
        else:
            value = np.asarray(value)
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == 3
            assert len(value) == 3
            self._props['scaling'] = value.astype(np.double)
        assert np.all(self._props['scaling'] > 0)

        #array scaling must be a factor of matrix dimension
        for i in range(3):
            teller = 0
            while (self.shape[i] % self._props['scaling'][i]) != 0:
                teller += 1
                if teller > 5:
                    self._props['scaling'][i] -= 1
                else:
                    self._props['scaling'][i] += 1
#            if self._props['scaling'][i] == self.shape[i]:
#                self._props['scaling'][i] = 1


    @property
    def image_orientation(self):
        return self._props['image_orientation']
    @image_orientation.setter
    def image_orientation(self, value):
        if isinstance(value, np.ndarray):
            self._props['image_orientation'] = value.astype(np.double)
        elif isinstance(value, str):
            self._props['image_orientation'] = self.string_to_array_converter('image_orientation', value)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['image_orientation'] = value.astype(np.double)
    @property
    def image_position(self):
        return self._props['image_position']
    @image_position.setter
    def image_position(self, value):
        if isinstance(value, np.ndarray):
            self._props['image_position'] = value.astype(np.double)
        elif isinstance(value, str):
            self._props['image_position'] = self.string_to_array_converter('image_position', value)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['image_position'] = value.astype(np.double)

    @property
    def data_center(self):
        return self._props['data_center']
    @data_center.setter
    def data_center(self, value):
        if isinstance(value, np.ndarray):
            self._props['data_center'] = value.astype(np.double)
        elif isinstance(value, str):
            self._props['data_center'] = self.string_to_array_converter('data_center', value)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['data_center'] = value.astype(np.double)

    @property
    def ignore_air(self):
        return self._props['ignore_air']
    @ignore_air.setter
    def ignore_air(self, value):
        self._props['ignore_air'] = self.bool_validator(value)

    @property
    def is_phantom(self):
        return self._props['is_phantom']
    @is_phantom.setter
    def is_phantom(self, value):
        self._props['is_phantom'] = self.bool_validator(value)

    @property
    def use_siddon(self):
        return self._props['use_siddon']
    @use_siddon.setter
    def use_siddon(self, value):
        self._props['use_siddon'] = self.bool_validator(value)

    @property
    def anode_angle(self):
        return self._props['anode_angle']
    @anode_angle.setter
    def anode_angle(self, value):
        self._props['anode_angle'] = self.float_validator(value)
        assert self._props['anode_angle'] > 0.

    @property
    def tube_start_angle_A(self):
        return self._props['tube_start_angle_A']
    @tube_start_angle_A.setter
    def tube_start_angle_A(self, value):
        ang = self.float_validator(value)
        ang = ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        self._props['tube_start_angle_A'] = self.float_validator(ang)

    @property
    def tube_start_angle_B(self):
        return self._props['tube_start_angle_B']
    @tube_start_angle_B.setter
    def tube_start_angle_B(self, value):
        ang = self.float_validator(value)
        ang = ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        self._props['tube_start_angle_B'] = self.float_validator(ang)

    @property
    def bowtie_radius(self):
        return self._props['bowtie_radius']
    @bowtie_radius.setter
    def bowtie_radius(self, value):
        self._props['bowtie_radius'] = self.float_validator(value)
        assert self._props['bowtie_radius'] > 0.

    @property
    def bowtie_radius(self):
        return self._props['bowtie_radius']
    @bowtie_radius.setter
    def bowtie_radius(self, value):
        self._props['bowtie_radius'] = self.float_validator(value)
        if self._props['bowtie_radius'] < 0:
            self._props['bowtie_radius'] = 0

    @property
    def bowtie_distance(self):
        return self._props['bowtie_distance']
    @bowtie_distance.setter
    def bowtie_distance(self, value):
        self._props['bowtie_distance'] = self.float_validator(value)
        if self._props['bowtie_distance'] < 0:
            self._props['bowtie_distance'] = 0

    @property
    def material(self):
        return self._arrays['material']
    @material.setter
    def material(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['material'] = value.astype(self._at['material'][0])

    @property
    def density(self):
        return self._arrays['density']
    @density.setter
    def density(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        if self._arrays['density'] is not None:
            self._arrays['density'] = None
        self._arrays['density'] = value.astype(self._at['density'][0])

    @property
    def organ(self):
        return self._arrays['organ']
    @organ.setter
    def organ(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['organ'] = value.astype(self._at['organ'][0])

    @property
    def ctarray(self):
        return self._arrays['ctarray']
    @ctarray.setter
    def ctarray(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['ctarray'] = value.astype(self._at['ctarray'][0])

    @property
    def exposure_modulation(self):
        return self._arrays['exposure_modulation']
    @exposure_modulation.setter
    def exposure_modulation(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 2
        self._arrays['exposure_modulation'] = value.astype(self._at['exposure_modulation'][0])

    @property
    def energy_imparted(self):
        return self._arrays['energy_imparted']
    @energy_imparted.setter
    def energy_imparted(self, value):
        if value is None:
            self._arrays['energy_imparted'] = None
            return
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['energy_imparted'] = value.astype(self._at['energy_imparted'][0])

    @property
    def material_map(self):
        return self._arrays['material_map']
    @material_map.setter
    def material_map(self, value):
        self._arrays['material_map'] = self.validate_structured_array(value, 'material_map')

    @property
    def organ_map(self):
        return self._arrays['organ_map']
    @organ_map.setter
    def organ_map(self, value):
        self._arrays['organ_map'] = self.validate_structured_array(value, 'organ_map')

    @property
    def organ_material_map(self):
        return self._arrays['organ_material_map']
    @organ_material_map.setter
    def organ_material_map(self, value):
        self._arrays['organ_material_map'] = self.validate_structured_array(value, 'organ_material_map')
