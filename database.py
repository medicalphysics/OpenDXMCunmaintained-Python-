# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:27:26 2015

@author: erlean
"""
import numpy as np
from engine import score_energy
from tube.tungsten import specter
import phase_space
import phantom_importer
import tables as tb
import utils
import os
import sys
import time
import re
import logging


import pdb


logger = logging.getLogger('OpenDXMC')


def add_materials_to_database(db_instance, materials_folder_path):
    path = os.path.abspath(materials_folder_path)

    density_file = os.path.join(path, "densities.txt")
    organic_file = os.path.join(path, "organics.txt")
    for p in utils.find_all_files([os.path.join(path, 'attinuation')]):
        name = os.path.splitext(os.path.basename(p))[0]
        # test for valif material name
        if re.match('^[\w-]+$', name) is None:
            logger.warning(
                "material file {0} contains illegal characters"
                ", only alphanummeric characters and "
                "dashes are allowed.".format(p)
                )
            continue
        material = Material(name, att_file=path, density_file=density_file, 
                            organic_file=organic_file)
        db_instance.add_material(material)


def generate_attinuation_lut(materials, material_map, min_eV=None,
                             max_eV=None):

    if min_eV is None:
        min_eV = 0.
    if max_eV is None:
        max_eV = 100.e6

    names = [m.name for m in materials]
    atts = {}

    for key, value in list(material_map.items()):
        key = int(key)
        try:
            ind = names.index(value)
        except ValueError:
            raise ValueError('No material named '
                             '{0} in first argument. '
                             'The material_map requires {0}'.format(value))
        atts[key] = materials[ind].attinuation


    energies = np.unique(np.hstack([a['energy'] for a in list(atts.values())]))
    e_ind = (energies <= max_eV) * (energies >= min_eV)
    if not any(e_ind):
        raise ValueError('Supplied minimum or maximum energies '
                         'are out of range')
    energies = energies[e_ind]
    lut = np.empty((len(atts), 5, len(energies)), dtype=np.double)
    for i, a in list(atts.items()):
        lut[i, 0, :] = energies
        for j, key in enumerate(['total', 'rayleigh', 'photoelectric',
                                 'compton']):
            lut[i, j+1, :] = np.interp(energies, a['energy'], a[key])
    return lut


class Database(object):
    def __init__(self, db_file_path):
        self.__dbpath = os.path.abspath(db_file_path)
        self.__filters = tb.Filters(complevel=7, complib='zlib')
        self.__h5 = None
        if not self.database_valid():
            raise ValueError("Database path is not writeable")

        self.validate()

    def open(self, mode='a'):
        if self.__h5 is None:
            self.__h5 = tb.open_file(self.__dbpath, mode=mode,
                                     filters=self.__filters)
            return

        if not self.__h5.isopen:
            self.__h5 = tb.open_file(self.__dbpath, mode=mode,
                                     filters=self.__filters)
            return
        if self.__h5.mode != mode:
            self.__h5.close()
            self.__h5 = tb.open_file(self.__dbpath, mode=mode,
                                     filters=self.__filters)
            return

    def close(self):
        if self.__h5 is None:
            return
        if self.__h5.isopen:
            self.__h5.close()

    def database_valid(self):
        try:
            self.open(mode='a')
        except ValueError:
            return False
        else:
            self.close()
            return True

    def get_node(self, name, where=None, create=False):
        self.open()
        if where is None:
            where = self.__h5.root
        try:
            n = self.__h5.get_node(where, name)
        except tb.NoSuchNodeError as e:
            if create:
                n = self.__h5.create_group(where, name, createparents=False)
            else:
                raise e
        else:
            pass
        return n

    def add_simulation(self, simulation):
        try:
            assert simulation.simulation_name != ""
            assert simulation.patient_name != ""
        except AssertionError:
            raise ValueError('Simulation object needs to reference a patient '
                             'and a simulation')
        self.open()
        try:
            table = self.__h5.get_node(self.__h5.root, name='simulations')
        except tb.NoSuchNodeError:
            table = self.__h5.create_table(self.__h5.root, 'simulations',
                                           description=simulation.dtype)

        condition = 'name == b"{}"'.format(simulation.name)

        for row in table.where(condition):
            for key, value in list(simulation.values.items()):
                row[key] = value
            row.update()
            break
        else:
            row = table.row
            for key, value in list(simulation.values.items()):
                row[key] = value
            row.append()
        table.flush()

        # saving dose
        dose_node = self.get_node('dose', create=True)
        try:
            self.get_node(simulation.name, dose_node)
        except tb.NoSuchNodeError:
            pass
        else:
            self.__h5.remove_node(dose_node, name=simulation.name)
        if simulation.dose is not None:
            self.__h5.create_carray(dose_node, simulation.name,
                                    obj=simulation.dose)

    def get_simulation(self, name):
        table = self.get_node('simulations')
        condition = 'name == b"{}"'.format(name)
        description = {}
        for row in table.where(condition):
            for key, value in zip(table.colnames, row[:]):
                description[key] = value
            break
        else:
            raise ValueError('No protocol named {}'.format(name))

        dose_node = self.get_node('dose', create=True)
        try:
            dose_arr = self.get_node(name, where=dose_node)
        except tb.NoSuchNodeError:
            Simulation(name, description=description)
        else:
            return Simulation(name, description=description,
                              dose=dose_arr.read())

    def add_protocol(self, protocol):
        self.open()
        try:
            table = self.__h5.get_node(self.__h5.root, name='protocols')
        except tb.NoSuchNodeError:
            table = self.__h5.create_table(self.__h5.root, 'protocols',
                                           description=protocol.dtype)

        condition = 'name == b"{}"'.format(protocol.name)

        for row in table.where(condition):
            for key, value in list(protocol.values.items()):
                row[key] = value
            row.update()
            break
        else:
            row = table.row
            for key, value in list(protocol.values.items()):
                row[key] = value
            row.append()
        table.flush()

    def get_protocol(self, name):
        table = self.get_node('protocols')
        condition = 'name == b"{}"'.format(name)
        description = {}
        for row in table.where(condition):
            for key, value in zip(table.colnames, row[:]):
                description[key] = value
            break
        else:
            raise ValueError('No protocol named {}'.format(name))
        return CTProtocol(name, description=description)

    def add_material(self, material):
        node = self.get_node('materials', create=True)
        att_node = self.get_node('attinuation', where=node, create=True)
        try:
            self.__h5.get_node(att_node, name=material.name)
        except tb.NoSuchNodeError:
            pass
        else:
            self.__h5.remove_node(att_node, name=material.name)
        mat_table = self.__h5.create_table(att_node, material.name,
                                           obj=material.attinuation)
        mat_table.flush()
        try:
            dens_table = self.__h5.get_node(node, name='info')
        except tb.NoSuchNodeError:
            dtype = np.dtype([('material', 'a64'), ('density', np.double), 
                              ('organic', np.bool)])
            dens_table = self.__h5.create_table(node, name='info',
                                                description=dtype)

        condition = 'material == b"{}"'.format(material.name)
        cond_index = dens_table.get_where_list(condition)
        for ind in cond_index:
            dens_table.remove_row(ind)
        row = dens_table.row
        row['material'] = material.name
        row['density'] = material.density
        row['organic'] = material.organic
        row.append()
        dens_table.flush()

    def get_material(self, name):
        node = self.get_node('materials', create=True)
        att_node = self.get_node('attinuation', where=node, create=True)
        try:
            mat_table = self.__h5.get_node(att_node, name=name)
        except tb.NoSuchNodeError as e:
            raise e
        try:
            dens_table = self.get_node('info', where=node, create=False)
        except tb.NoSuchNodeError as e:
            density = None
            organic = None
        else:
            for row in dens_table:
                if row['material'] == bytes(name, encoding='ascii'):
                    density = row['density']
                    organic = row['organic']
                    break
            else:
                density = None
                organic = None
        return Material(name, attinuations=mat_table.read(), density=density, 
                        organic=organic)

    def get_all_materials(self, organic_only=False):
        materials = []
        node = self.get_node('materials', create=True)
        _ = self.get_node('attinuation', where=node, create=True)
        try:
            info_table = self.get_node('info', where=node)
        except tb.NoSuchNodeError:
            return []
        else:
            if organic_only:
                for row in info_table:
                    materials.append(self.get_material(row['material']))
            else:
                for row in info_table.where('organic == True'):
                    materials.append(self.get_material(row['material']))
        return materials

    def validate_patient(self, patient):
        # test for all properties
        for prop in ['spacing']:
            if getattr(patient, prop) is None:
                raise ValueError('Patient {0} needs to have the {1} property'.format(patient.name, prop))
        if patient.ct_array is None:
            assert all([patient.material_array, patient.material_map])
        else:            
            return          
            
        #testing material_array
        if patient.material_array is not None:
            unique_materials = np.unique(patient.material_array)
            for mat_ind in unique_materials:
                if mat_ind not in list(patient.material_map.keys()):
                    raise ValueError('Material map must have all elements in material array.')

            #sanitize material arrays:
            if np.sum(unique_materials - np.arange(len(unique_materials))) != 0:
                mat_map_red = {}
                mat_arr_red = np.zeros_like(patient.material_array)
                for ind, mat in enumerate(unique_materials):
                    mat_map_red[ind] = patient.material_map[mat]
                    indices = patient.material_array == mat
                    mat_arr_red[indices] = ind
                patient.material_array = mat_arr_red
                patient.material_map = mat_map_red

        if patient.density_array is None:
            dens_arr = np.zeros_like(patient.material_array, dtype=np.double)
            materials = self.get_all_materials()
            materials_dict = {m.name: m.density for m in materials if m.density is not None}
            for key, value in list(patient.material_map.items()):
                if value in materials_dict:
                    indices = patient.material_array == key
                    dens_arr[indices] = materials_dict[value]
                else:
                    raise ValueError('Density data for required material {} for patient {} is not in database'.format(value, patient.name))
            patient.density_array = dens_arr
        return patient

    def add_patient(self, patient, overwrite=False):
        try:
            patient = self.validate_patient(patient)
        except Exception as e:
            raise e

        node = self.get_node('patients', create=True)
        try:
            pat_node = self.__h5.get_node(node, name=patient.name)
        except tb.NoSuchNodeError:
            pass
        else:
            if overwrite:
                self.__h5.remove_node(node, name=patient.name, recursive=True)
            else:
                raise tb.NodeError('Patient already exists')
                return
        pat_node = self.__h5.create_group(node, patient.name)
        # saving arrays
        for var in ['organ_array', 'density_array', 'material_array',
                    'spacing', 'ct_array', 'exposure_modulation']:
            obj = getattr(patient, var)
            if obj is not None:
                self.__h5.create_carray(pat_node, var, obj=obj)
        # saving maps
        for var in ['material_map', 'organ_map']:
            obj = getattr(patient, var)
            if obj is None:
                continue
            elif len(obj) == 0:
                continue
            array = np.recarray(len(obj), dtype=[('key', 'i4'),
                                                 ('value', 'a64')])
            for ind, item in enumerate(obj.items()):
                key, value = item
                array['key'][ind] = key
                array['value'][ind] = value
            table = self.__h5.create_table(pat_node, var, obj=array)
            table.flush()

    def get_patient(self, name):
        node = self.get_node('patients', create=False)
        try:
            pat_node = self.__h5.get_node(node, name=name)
        except tb.NoSuchNodeError as e:
            raise e

        pat_obj = Patient(name)

        # getting arrays
        for var in ['organ_array', 'density_array', 'material_array',
                    'spacing', 'exposure_modulation']:
            try:
                obj = self.__h5.get_node(pat_node, name=var)
            except tb.NoSuchNodeError:
                pass
            else:
                setattr(pat_obj, var, obj.read())

        # getting maps
        for var in ['material_map', 'organ_map']:
            try:
                obj = self.__h5.get_node(pat_node, name=var)
            except tb.NoSuchNodeError:
                pass
            else:
                data = {}
                for row in obj.iterrows():
                    data[row['key']] = row['value']
                setattr(pat_obj, var, data)
        return pat_obj

    def validate(self):
        self.open()
        broken_sims = []
        try:
            sim_table = self.get_node('simulations')
        except tb.NoSuchNodeError:
            return
        for row in sim_table:
            protocol_name = row['protocol_name']
            patient_name = row['patient_name']
            try:
                self.get_protocol(protocol_name)
            except ValueError:
                broken_sims.append(row.nrow)
                continue
            try:
                self.get_patient(patient_name)
            except ValueError:
                broken_sims.append(row.nrow)
                continue
        sim_table.autoindex = False
        print(('boken sims', broken_sims))
        for ind in broken_sims:
            sim_table.remove_row(ind)
        sim_table.flush_rows_to_index()
        sim_table.autoindex = True
        sim_table.flush()

    def __del__(self):
        self.close()


class Material(object):
    """
    Material object for interaction with database.

    INPUT:

    name:
        Name of material, only alphanummeric values are allowed
    density [optional]:
        material density in g/cm3
    att_file [optional]:
        filepath to text file containing attinuation data
    attinuations [optional]:
        attinuation nd array
    density_file [optional]:
        read density from this file

    """
    def __init__(self, name, density=None, organic=None, att_file=None, 
                 attinuations=None, density_file=None, organic_file=None):
        self.name = name
        self.__density = density
        self.__atts = attinuations
        self.__organic = None

        if att_file is not None:
            self.attinuation = att_file
        if density_file is not None:
            self.density_from_file(density_file)
        if organic_file is not None:
            self.organic_from_file(organig_file)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__name = name.lower()

    @property
    def density(self):
        return self.__density

    @density.setter
    def density(self, value):
        self.__density = float(value)

    @property
    def organic(self):
        return self.__organic

    @organic.setter
    def organic(self, value):
        self.__organic = bool(value)

    def organic_from_file(self, path):
        try:
            with open(path) as f:
                s = f.read()
                l = s.lower().split()
                if self.name in l:
                    self.organic = True
        except FileNotFoundError:
            logger.warning("Could not open organic file {0} "
                           "for material {1}".format(path, self.name))
        
    def density_from_file(self, path):
        try:
            with open(path) as f:
                s = f.read()
                l = s.lower().split()
                if self.name in l:
                    self.density = l[l.index(self.name) + 1]
        except FileNotFoundError:
            logger.warning("Could not open densities file {0} "
                           "for material {1}".format(path, self.name))

    @property
    def attinuation(self):
        return self.__atts

    @attinuation.setter
    def attinuation(self, path):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        letters += letters.lower()
        header = ('energy', 'rayleigh', 'compton', 'photoelectric', 'ppn',
                  'ppe', 'total', 'totalnocoherent')
        data_list = dict([(h, list()) for h in header])
        with open(path) as f:
            # reading lines
            for line in f:
                line = line.strip()
                # testing for header or empty line
                if len(line) == 0:
                    continue

                if line[0] in letters:
                    continue
                for i, el in enumerate(line.split()):
                    data_list[header[i]].append(float(el))
        N = len(data_list['energy'])
        dtype = [(h, np.double) for h in header]
        array = np.recarray(N, dtype=dtype)
        for h in header:
            array[h] = np.array(data_list[h], dtype=np.double)[:]
        array['energy'] *= 1.e6
        array.sort(order='energy')
        self.__atts = array


class CTProtocol(object):
    def __init__(self, name, description=None):
        self.__description = {'name': '',
                              'scan_fov': 50.,
                              'sdd': 100.,
                              'detector_width': 0.6,
                              'detector_rows': 64,
                              'modulation_xy': False,
                              'modulation_z': False,
                              'al_filtration': 7.,
                              'xcare': False,
                              'ctdi_air100': 0.,
                              'ctdi_w100': 0.,
                              'kV': 120.,
                              'region': 'abdomen',
                              # per 1000000 histories to dose
                              'conversion_factor_ctdiair': 0,
                              # per 1000000 histories to dose
                              'conversion_factor_ctdiw': 0,
                              'is_spiral': False,
                              'pitch': 0
                              }
        self.__dtype = {'name': 'a64',
                        'scan_fov': np.float,
                        'sdd': np.float,
                        'detector_width': np.float,
                        'detector_rows': np.int,
                        'modulation_xy': np.bool,
                        'modulation_z': np.bool,
                        'al_filtration': np.float,
                        'xcare': np.bool,
                        'ctdi_air100': np.float,
                        'ctdi_w100': np.float,
                        'kV': np.float,
                        'region': 'a64',
                        # per 1000000 histories to dose
                        'conversion_factor_ctdiair': np.float,
                        # per 1000000 histories to dose
                        'conversion_factor_ctdiw': np.float,
                        'is_spiral': np.bool,
                        'pitch': np.float
                        }
        if description is not None:
            for key, value in list(description.items()):
                self.__description[key] = value
        self.name = name

    @property
    def dtype(self):
        d = {'names': [], 'formats': []}
        for key, value in list(self.__dtype.items()):
            d['names'].append(key)
            d['formats'].append(value)
        return np.dtype(d)

    @property
    def values(self):
        return self.__description

    @property
    def pitch(self):
        return self.__description['pitch']

    @pitch.setter
    def pitch(self, value):
        value = float(value)
        assert value > 0
        self.__description['pitch'] = value

    @property
    def name(self):
        return self.__description['name']

    @name.setter
    def name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__description['name'] = name.lower()

    @property
    def scan_fov(self):
        return self.__description['scan_fov']

    @scan_fov.setter
    def scan_fov(self, value):
        value = float(value)
        assert value > 0
        self.__description['scan_fov'] = value

    @property
    def sdd(self):
        return self.__description['sdd']

    @sdd.setter
    def sdd(self, value):
        value = float(value)
        assert value > 0
        self.__description['sdd'] = value

    @property
    def detector_width(self):
        return self.__description['detector_width']

    @detector_width.setter
    def detector_width(self, value):
        value = float(value)
        assert value > 0
        self.__description['detector_width'] = value

    @property
    def detector_rows(self):
        return self.__description['detector_rows']

    @detector_rows.setter
    def detector_rows(self, value):
        value = float(value)
        assert value > 0
        self.__description['detector_rows'] = value

    @property
    def modulation_xy(self):
        return self.__description['modulation_xy']

    @modulation_xy.setter
    def modulation_xy(self, value):
        value = bool(value)
        self.__description['modulation_xy'] = value

    @property
    def modulation_z(self):
        return self.__description['modulation_z']

    @modulation_z.setter
    def modulation_z(self, value):
        value = bool(value)
        self.__description['modulation_z'] = value

    @property
    def al_filtration(self):
        return self.__description['al_filtration']

    @al_filtration.setter
    def al_filtration(self, value):
        value = float(value)
        assert value > 0
        self.__description['al_filtration'] = value

    @property
    def xcare(self):
        return self.__description['xcare']

    @xcare.setter
    def xcare(self, value):
        value = bool(value)
        self.__description['xcare'] = value

    @property
    def is_spiral(self):
        return self.__description['is_spiral']

    @is_spiral.setter
    def is_spiral(self, value):
        value = bool(value)
        self.__description['is_spiral'] = value

    @property
    def ctdi_air100(self):
        return self.__description['ctdi_air100']

    @ctdi_air100.setter
    def ctdi_air100(self, value):
        value = float(value)
        assert value >= 0
        self.__description['ctdi_air100'] = value

    @property
    def ctdi_w100(self):
        return self.__description['ctdi_w100']

    @ctdi_w100.setter
    def ctdi_w100(self, value):
        value = float(value)
        assert value >= 0
        self.__description['ctdi_w100'] = value

    @property
    def kV(self):
        return self.__description['kV']

    @kV.setter
    def kV(self, value):
        value = float(value)
        assert value >= 0
        self.__description['kV'] = value

    @property
    def region(self):
        return self.__description['region']

    @region.setter
    def region(self, value):
        value = str(value)
        region = "".join([l for l in value.split() if len(l) > 0])
        assert len(region) > 0
        self.__description['region'] = region.lower()

    @property
    def conversion_factor_ctdiair(self):
        return self.__description['conversion_factor_ctdiair']

    @conversion_factor_ctdiair.setter
    def conversion_factor_ctdiair(self, value):
        value = float(value)
        assert value >= 0
        self.__description['conversion_factor_ctdiair'] = value

    @property
    def conversion_factor_ctdiw(self):
        return self.__description['conversion_factor_ctdiw']

    @conversion_factor_ctdiw.setter
    def conversion_factor_ctdiw(self, value):
        value = float(value)
        assert value >= 0
        self.__description['conversion_factor_ctdiw'] = value

    @property
    def total_collimation(self):
        return self.detector_rows * self.detector_width

    def obtain_ctdiair_conversion_factor(self, material, callback=None):

        spacing = np.array((1, 1, 10), dtype=np.double)
        N = np.rint(np.array((self.sdd / spacing[0], self.sdd / spacing[1], 1),
                             dtype=np.double))

        offset = -N * spacing / 2.
        material_array = np.zeros(N, dtype=np.intc)
        material_map = {0: material.name}
        density_array = np.zeros(N, dtype=np.double) + material.density
        lut = generate_attinuation_lut([material], material_map, max_eV=0.5e6)
        dose = np.zeros_like(density_array, dtype=np.double)

        en_specter = specter(self.kV, angle_deg=10., filtration_materials='Al',
                             filtration_mm=6.)
        norm_specter = (en_specter[0], en_specter[1]/en_specter[1].sum())
        particles = phase_space.ct_seq(self.scan_fov, self.sdd,
                                       self.total_collimation,
                                       histories=1000, exposures=1200,
                                       batch_size=100000,
                                       energy_specter=norm_specter)
#        pdb.set_trace()
        t0 = time.clock()
        for batch, i, tot in particles:
            score_energy(batch, N, spacing, offset, material_array,
                         density_array, lut, dose)
            p = round(i * 100 / float(tot), 1)
            t1 = (time.clock() - t0) / float(i) * (tot - i)
            print(('{0}% {1}, ETA in {2}'.format(p, time.ctime(),
                                                utils.human_time(t1))))

        center = np.floor(N / 2).astype(np.int)
        d = dose[center[0], center[1],
                 center[2]] / material.density * np.prod(spacing)
        d /= float(tot) / 1000000.
        print(d)
        self.__description['conversion_factor_ctdiair'] = self.ctdi_air100 / d

    def generate_ctdi_phantom(self, pmma, air, size=32.):
        spacing = np.array((1, 1, 10), dtype=np.double)
        N = np.rint(np.array((self.sdd / spacing[0], self.sdd / spacing[1], 1),
                             dtype=np.double))

        offset = -N * spacing / 2.
        material_array = np.zeros(N, dtype=np.intc)
        radii_phantom = size * spacing[0]
        radii_meas = 2. * spacing[0]
        center = (N * spacing / 2.)[:2]
        radii_pos = 28*spacing[0]
        pos = [(center[0], center[1])]
        for ang in [0, 90, 180, 270]:
            dx = radii_pos * np.sin(np.deg2rad(ang))
            dy = radii_pos * np.cos(np.deg2rad(ang))
            pos.append((center[0] + dx, center[1] + dy))

        for i in range(int(N[2])):
            material_array[:, :, i] += utils.circle_mask((N[0], N[1]),
                                                         radii_phantom)
            for p in pos:
                material_array[:, :, i] += utils.circle_mask((N[0], N[1]),
                                                             radii_meas,
                                                             center=p)

        material_map = {0: air.name, 1: pmma.name, 2: air.name}
        density_array = np.zeros_like(material_array, dtype=np.double)
        density_array[material_array == 0] = air.density
        density_array[material_array == 1] = pmma.density
        density_array[material_array == 2] = air.density

#        density_array = np.zeros(N, dtype=np.double) + material.density
        lut = generate_attinuation_lut([air, pmma], material_map, max_eV=0.5e6)
        return N, spacing, offset, material_array, density_array, lut, pos
#        dose = np.zeros_like(density_array, dtype=np.double)

    def obtain_ctdiw_conversion_factor(self, pmma, air,
                                       callback=None, phantom_size=32.):

        args = self.generate_ctdi_phantom(pmma, air)
        N, spacing, offset, material_array, density_array, lut, meas_pos = args

        # removing outside air
        lut[0, 1:, :] = 0

        dose = np.zeros_like(density_array)

        en_specter = specter(self.kV, angle_deg=10., filtration_materials='Al',
                             filtration_mm=6.)
        norm_specter = (en_specter[0], en_specter[1]/en_specter[1].sum())

        particles = phase_space.ct_seq(self.scan_fov, self.sdd,
                                       self.total_collimation,
                                       histories=1000, exposures=1200,
                                       batch_size=100000,
                                       energy_specter=norm_specter)
#        pdb.set_trace()
        t0 = time.clock()
        for batch, i, tot in particles:
            score_energy(batch, N, spacing, offset, material_array,
                         density_array, lut, dose)
            p = round(i * 100 / float(tot), 1)
            t1 = (time.clock() - t0) / float(i) * (tot - i)
            print(('{0}% {1}, ETA in {2}'.format(p, time.ctime(),
                                                utils.human_time(t1))))

        d = []
        for p in meas_pos:
            x, y = int(p[0]), int(p[1])
            d.append(dose[x, y, 0] / air.density * np.prod(spacing))
            d[-1] /= (float(tot) / 1000000.)

        ctdiv = d.pop(0) / 3.
        ctdiv += 2. * sum(d) / 3. / 4.
        print(ctdiv)
        self.conversion_factor_ctdiw = self.ctdi_w100 / ctdiv


class Patient(object):
    def __init__(self, name):
        self.__name = None
        self.__organ_array = None
        self.__organ_map = None
        self.__material_array = None
        self.__material_map = None
        self.__density_array = None
        self.__spacing = None
        self.__ct_array = None
        self.__exposure_modulation = None
        self.name = name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__name = name.lower()

    @property
    def ct_array(self):
        return self.__ct_array

    @ct_array.setter
    def ct_array(self, value):
        self.__ct_array = value.astype(np.int16)

    @property
    def organ_array(self):
        return self.__organ_array

    @organ_array.setter
    def organ_array(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__organ_array = value.astype(np.intc)

    @property
    def material_array(self):
        return self.__material_array

    @material_array.setter
    def material_array(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__material_array = value.astype(np.intc)

    @property
    def density_array(self):
        return self.__density_array

    @density_array.setter
    def density_array(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__density_array = value.astype(np.double)

    @property
    def organ_map(self):
        return self.__organ_map

    @organ_map.setter
    def organ_map(self, value):
        assert isinstance(value, dict)
        self.__organ_map = value

    @property
    def material_map(self):
        return self.__material_map

    @material_map.setter
    def material_map(self, value):
        assert isinstance(value, dict)
        self.__material_map = value

    @property
    def spacing(self):
        return self.__spacing

    @spacing.setter
    def spacing(self, value):
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value, dtype=np.double)
            except Exception as e:
                raise e
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 1
#        assert len(value.shape[0]) == 3
        self.__spacing = value.astype(np.double)
    @property
    def exposure_modulation(self):
        return self.__exposure_modulation
    
    @exposure_modulation.setter
    def exposure_modulation(self, value):
        arr = np.array(value).astype(np.float)
        assert len(arr.shape) == 2
        assert arr.shape[0] > 5
        self.__exposure_modulation = arr
        
    def generate_material_map_array_from_ct_array(self, specter, materials):
        if self.ct_array is None:
            return
        specter = (specter[0]/1000., specter[1]/specter[1].sum())
        
        water_key = None
        material_map = {}
        material_att = {}
        material_dens = {}
        materials.sort(key=lambda x: x.density)
        for i, mat in enumerate(materials):
            material_map[i] = mat.name
            material_dens[i] = mat.density
            #interpolationg and integrating attinuation coefficient
            material_att[i] = np.trapz(np.interp(specter[0], 
                                                 mat.attinuation['energy'], 
                                                 mat.attinuation['total']), 
                                       specter[0])
            material_att[i] *= mat.density
            if mat.name == 'water':
                water_key = i
        # getting a list of attinuation
        material_HU_list = [(key, (att / material_att[water_key] -1.)*1000.) 
                            for key, att in material_att.items()]
        material_HU_list.sort(key=lambda x: x[1])
        
        material_array = np.zeros_like(self.ct_array, dtype=np.int)
        density_array = np.zeros_like(self.ct_array, dtype=np.float)
        llim = -np.inf
        for i in range(len(material_HU_list)):
            if i == len(material_HU_list) -1:
                ulim = np.inf
            else:
                ulim = 0.5 *(material_HU_list[i][1] + material_HU_list[i+1][1])
            ind = np.nonzero((self.ct_array > llim) * (self.ct_array <= ulim))
            material_array[ind] = material_HU_list[i][0]
            density_array[ind] = material_dens[material_HU_list[i][0]]
            llim = ulim
        self.material_map = material_map
        self.material_array = material_array
        self.density_array = density_array
        
        
class Simulation(object):
    def __init__(self, name, description=None, dose=None):
        self.__description = {'name': '',
                              'protocol_name': '',
                              'patient_name': '',
                              'exposures': 1200.,
                              'histories': 1000,
                              'batch_size': 0,
                              'start': 0.,
                              'stop': 0.,
                              'start_at_exposure_no': 0,
                              'finish': False
                              }
        self.__dtype = {'name': 'a64',
                        'protocol_name': 'a64',
                        'patient_name': 'a64',
                        'exposures': np.int,
                        'histories': np.int,
                        'batch_size': np.int,
                        'start': np.float,
                        'stop': np.float,
                        'start_at_exposure_no': np.int,
                        'finish': np.bool
                        }
        self.__dose = None
        if description is not None:
            for key, value in list(description.items()):
                self.__description[key] = value
        if dose is not None:
            self.dose = dose
        self.name = name

    @property
    def dose(self):
        return self.__dose

    @dose.setter
    def dose(self, value):
        assert isinstance(value, np.ndarray)
        self.__dose = value

    @property
    def finish(self):
        return self.__description['finish']

    @finish.setter
    def finish(self, value):
        value = bool(value)
        self.__description['finish'] = value

    @property
    def name(self):
        return self.__description['name']

    @name.setter
    def name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__description['name'] = name.lower()

    @property
    def protocol_name(self):
        return self.__description['protocol_name']

    @protocol_name.setter
    def protocol_name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__description['protocol_name'] = name.lower()

    @property
    def patient_name(self):
        return self.__description['patient_name']

    @patient_name.setter
    def patient_name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__description['patient_name'] = name.lower()

    @property
    def exposures(self):
        return self.__description['exposures']

    @exposures.setter
    def exposures(self, value):
        value = int(value)
        assert value >= 0
        self.__description['exposures'] = value

    @property
    def histories(self):
        return self.__description['histories']

    @histories.setter
    def histories(self, value):
        value = int(value)
        assert value >= 0
        self.__description['histories'] = value

    @property
    def batch_size(self):
        return self.__description['batch_size']

    @batch_size.setter
    def batch_size(self, value):
        value = int(value)
        assert value >= 0
        self.__description['batch_size'] = value

    @property
    def start(self):
        return self.__description['start']

    @start.setter
    def start(self, value):
        value = float(value)
        assert value >= 0
        self.__description['start'] = value

    @property
    def stop(self):
        return self.__description['stop']

    @stop.setter
    def stop(self, value):
        value = float(value)
        assert value >= 0
        self.__description['stop'] = value

    @property
    def start_at_exposure_no(self):
        return self.__description['start_at_exposure_no']

    @start_at_exposure_no.setter
    def start_at_exposure_no(self, value):
        value = int(value)
        assert value >= 0
        self.__description['start_at_exposure_no'] = value

    @property
    def dtype(self):
        d = {'names': [], 'formats': []}
        for key, value in list(self.__dtype.items()):
            d['names'].append(key)
            d['formats'].append(value)
        return np.dtype(d)

    @property
    def values(self):
        return self.__description



def test_simulation():

    p2 = Patient('test2')
    p1 = Patient('test1')

    pt = CTProtocol('test')
    s2 = Simulation('test2')
    s1 = Simulation('test1')
    s1.patient_name = 'test1'
    s1.protocol_name = 'test2'
    s2.patient_name = 'test2'
    s2.protocol_name = 'test2'


    db_path = os.path.abspath("C://test//test.h5")
    db = Database(db_path)

    db.add_simulation(s1)
    db.add_simulation(s2)
    db.add_patient(p1, overwrite=True)
    db.add_patient(p2, overwrite=True)
    db.add_protocol(pt)

    db.validate()
#    pdb.set_trace()
#    ss = db.get_simulation('test')
#    pdb.set_trace()

def test_protocol():
    p = CTProtocol('test')

    db_path = os.path.abspath("C://test//test.h5")
    database = Database(db_path)
    database.add_protocol(p)
    pp = database.get_protocol('test')
    pp.ctdi_air100 = 8.7
    pp.ctdi_w100 = 14.9

    air = database.get_material('air')
    pmma = database.get_material('pmma')
    pp.obtain_ctdiw_conversion_factor(pmma, air)

    pp.obtain_ctdiair_conversion_factor(air)
    print(pp.conversion_factor_ctdiair, pp.conversion_factor_ctdiw)
    database.add_protocol(pp)
#    pdb.set_trace()

def test_materials():
    db_path = os.path.abspath("C://test//test.h5")
    database = Database(db_path)
    material_dir = os.path.join(os.path.dirname(sys.argv[0]), 'data', 'materials')
#    material_dir = os.path.abspath("C://GitHub//OpenDXMC//data//materials//attinuation")
    material_paths = utils.find_all_files([os.path.join(material_dir, 'attinuation')])
    dens_file = os.path.join(material_dir, 'densities.txt')
    for path in material_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        material = Material(name, att_file=path, density_file=dens_file)
        database.add_material(material)
    database.close()

    materials = database.get_all_materials()
    air = database.get_material('air')
    database.close()

def test_golem_patient():
    phantom_path = os.path.join(os.path.dirname(sys.argv[0]), 'data','phantoms')
    spacing, material_array, material_map, organ_array, organ_map = phantom_importer.read_golem(phantom_path)
    pat = Patient('golem')
    pat.spacing = spacing
    pat.material_array = material_array
    pat.material_map = material_map
    pat.organ_array = organ_array
    pat.organ_map = organ_map

    db_path = os.path.abspath("C://test//test.h5")
    db = Database(db_path)
    db.add_patient(pat)
    db.close()

def test_simulation():
    db_path = os.path.abspath("C:\test\test.h5")
    db = Database(db_path)
    # adding materials


if __name__ == '__main__':
#    test_materials()
#    test_protocol()
#    test_simulation()
    test_golem_patient()

