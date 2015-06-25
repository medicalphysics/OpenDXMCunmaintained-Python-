# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:27:26 2015

@author: erlean
"""
import numpy as np
from engine import score_energy
from tube.tungsten import specter
import phase_space
import tables as tb
import utils
import os
import sys
import time 

import pdb


def generate_attinuation_lut(materials, material_map, min_eV=None, max_eV=None):

    if min_eV is None:
        min_eV = 0.
    if max_eV is None:
        max_eV = 100.e6
    
    names = [m.name for m in materials]
    atts = {}
    
    for key, value in material_map.items():
        key = int(key)
        try:
            ind = names.index(value)
        except ValueError:
            raise ValueError('No material named '
                             '{0} in first argument. '
                             'The material_map requires {0}'.format(value))
        atts[key] = materials[ind].attinuation
        
    
    energies = np.unique(np.array([a['energy'] for a in atts.values()]).ravel())
    e_ind = (energies <= max_eV) * (energies >= min_eV)
    if not any(e_ind):
        raise ValueError('Supplied minimum or maximum energies are out of range')
    energies = energies[e_ind]
    lut = np.empty((len(atts), 5, len(energies)), dtype=np.double)
    for i, a in atts.items():
        lut[i, 0, :] = energies
        for j, key in enumerate(['total', 'rayleigh', 'photoelectric', 'compton']):
            lut[i, j+1, :] = np.interp(energies, a['energy'], a[key])
    return lut
    

class Database(object):
    def __init__(self, db_file_path):
        self.__dbpath = os.path.abspath(db_file_path)
        self.__filters = tb.Filters(complevel=7, complib='zlib')
        self.__h5 = None
        if not self.database_valid():
            raise ValueError("Database path is not writeable")

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
        except tb.NoSuchNodeError, e:
            if create:
                print where
                n = self.__h5.create_group(where, name, createparents=False)
            else:
                raise e
        else:
            pass
        return n

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
            dens_table = self.__h5.get_node(node, name='densities')
        except tb.NoSuchNodeError:
            dtype = np.dtype([('key', 'a64'), ('value', np.double)])
            dens_table = self.__h5.create_table(node, name='densities', description=dtype)

        condition = 'key == "{}"'.format(material.name)
        cond_index = dens_table.get_where_list(condition)
        for ind in cond_index:
            dens_table.remove_row(ind)
        row = dens_table.row
        row['key'] = material.name
        row['value'] = material.density
        row.append()
        dens_table.flush()
        

    def get_material(self, name):
        node = self.get_node('materials', create=True)
        att_node = self.get_node('attinuation', where=node, create=True)
        try:
            mat_table = self.__h5.get_node(att_node, name=name)
        except tb.NoSuchNodeError, e:
            raise e

        try:
            dens_table = self.__h5.get_node(node, name='densities')
        except tb.NoSuchNodeError, e:
            pass
        else:
            for row in dens_table:
                if row['key'] == name:        
                    density = row['value']
                    break
            else:
                density = None
        
        return Material(name, attinuations=mat_table.read(), density=density)
        


    def get_all_materials(self):
        materials = []
        node = self.get_node('materials', create=True)
        att_node = self.get_node('attinuation', where=node, create=True)
        for child in att_node:
            materials.append(self.get_material(child._v_name))
        return materials

    def add_patient(self, patient, overwrite=False):
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
                    'spacing']:
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
        except tb.NoSuchNodeError, e:
            raise e

        pat_obj = Patient(name)

        # getting arrays
        for var in ['organ_array', 'density_array', 'material_array',
                    'spacing']:
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

    def __del__(self):
        self.close()


class Material(object):
    def __init__(self, name, density=None, att_file=None, attinuations=None, density_file=None):
        self.name = name
        self.__density = density
        self.__atts = attinuations

        if att_file is not None:
            self.attinuation = att_file
        if density_file is not None:
            self.density_from_file(density_file)
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

    def density_from_file(self, path):
        with open(path) as f:
            s = f.read()
            l = s.lower().split()
            if self.name in l:
                self.density = l[l.index(self.name) + 1]
                
            
            
            

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
        description = {'name': '',
                              'scan_fov': 50.,
                              'sdd': 100.,
                              'detector_width': 0.6,
                              'detector_rows': 64,
                              'modulation_xy': False,
                              'modulation_z': False,
                              'al_filtration': 0.,
                              'xcare': False,
                              'ctdi_air100': 0.,
                              'ctdi_w100': 0.,
                              'kV': 120.,
                              'region': 'abdomen',
                              'conversion_factor_ctdiair': 0, # per 1000000 histories to dose
                              'conversion_factor_ctdiw': 0, # per 1000000 histories to dose
                              'is_spiral': False,
                             }
        dtype = {'name': 'a64',
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
                        'conversion_factor_ctdiair': np.float, # per 1000000 histories to dose
                        'conversion_factor_ctdiw': np.float, # per 1000000 histories to dose
                        'is_spiral': np.bool,
                        }
        super(CTProtocol, self).__setattr__('__description', description)
        super(CTProtocol, self).__setattr__('__dtype', dtype)
        
    def __setattr__(self, name, value):
        if '__description' in self.__dict__:
            if name in super(CTProtocol, self).__getattr__('__description'):
#                if isinstance(value, str):
#                    value = "".join([l for l in value.split() if len(l) > 0])
#                    value = value.lower()
    
                self.__dict__['__description'][name] = value
                return
        super(CTProtocol, self).__setattr__(name, value)
    
    def __getattr__(self, name):
        if '__description' in self.__dict__:
            if name in super(CTProtocol, self).__getattr__('__description'):
                return self.__dict__['__description'][name] 
        return super(CTProtocol, self).__getattr__(name)
            
        
#    @property
#    def name(self):
#        return self.__dict__['__description']['name'] 
#
#    @name.setter
#    def name(self, value):
#        value = str(value)
#        name = "".join([l for l in value.split() if len(l) > 0])
#        assert len(name) > 0
#        self.__dict__['__description']['name'] = name.lower()
##        self.__description['name'] = name.lower()

    def obtain_ctdiair_conversion_factor(self, material, 
                                 callback=None):
        
        spacing = np.array((1 ,1, 10), dtype=np.double)
        N = np.rint(np.array((self.sdd / spacing[0], self.sdd / spacing[1], 1),
                             dtype=np.double))
        
        offset = -N * spacing / 2.
        material_array = np.zeros(N, dtype=np.intc)
        material_map = {0: material.name}
        density_array = np.zeros(N, dtype=np.double) + material.density
        lut = generate_attinuation_lut([material], material_map, max_eV=0.5e6)
        dose = np.zeros_like(density_array, dtype=np.double)
        
        en_specter = specter(self.kV, angle_deg=10., filtration_materials='Al', filtration_mm=6.)
        norm_specter = (en_specter[0], en_specter[1]/en_specter[1].sum())        
        particles = phase_space.ct_seq(self.scan_fov, self.sdd, 
                                       self.total_collimation, 
                                       histories=10000, exposures=1200, 
                                       batch_size=1000000, 
                                       energy_specter=norm_specter)
#        pdb.set_trace()
        t0 = time.clock() 
        for batch, i, tot in particles:
            score_energy(batch, N, spacing, offset, material_array, 
                         density_array, lut, dose)
            p = round(i * 100 / float(tot), 1)
            t1 = (time.clock()- t0) / float(i) * (tot - i)
            print '{0}% {1}, ETA in {2}'.format(p, time.ctime(), utils.human_time(t1))
        
        center = np.floor(N / 2).astype(np.int)
        d = dose[center[0], center[1], center[2]] / material.density * np.prod(spacing)
        d /= float(tot) / 1000000. 
        print d
        self.__desciption['conversion_factor_ctdiair'] = self.ctdi_air100 / d
        print self.conversion_factor
        
            
        
#        lut = np.zeros((1,5,))
        

class Patient(object):
    def __init__(self, name):
        self.__name = None
        self.__organ_array = None
        self.__organ_map = None
        self.__material_array = None
        self.__material_map = None
        self.__density_array = None
        self.__spacing = None

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
#        assert isinstance(value, np.ndarray)
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
            except Exception, e:
                raise e
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 1
        assert len(value.shape[0]) == 3
        self.__spacing = value.astype(np.double)


class Simulation(object):
    def __init__(self):
        self.__exposures = 100.
        self.__histories = 1
        self.__energy = 70000.
        self.__energy_specter = None
        self.__batch_size = None
        self.__pitch = 1.
        self.__start = 0.
        self.__stop = 1.
        self.__start_at_exposure_no = 0

def test_protocol():
    p = CTProtocol('test')
    pdb.set_trace()
    p.ctdi_air100 = 8.76
    
    db_path = os.path.abspath("C://test//test.h5")
    database = Database(db_path)
    air = database.get_material('air')
    p.obtain_conversion_factor(air)
    pdb.set_trace()
    
def test_materials():
    db_path = os.path.abspath("C://test//test.h5")
    database = Database(db_path)
    material_dir = os.path.join(os.path.dirname(sys.argv[0]), 'data//materials')
#    material_dir = os.path.abspath("C://GitHub//OpenDXMC//data//materials//attinuation")
    material_paths = utils.find_all_files([os.path.join(material_dir, 'attinuation')])
    dens_file = os.path.join(material_dir, 'densities.txt')
    for path in material_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        material = Material(name, att_file=path, density_file=dens_file)
        database.add_material(material)
        database.close()

    materials = database.get_all_materials()
    


if __name__ == '__main__':
    test_materials()
    test_protocol()

