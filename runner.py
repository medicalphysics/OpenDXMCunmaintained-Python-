# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:27:26 2015

@author: erlean
"""
import numpy as np
from engine import score_energy
from tube.tungsten import specter
import tables as tb
import utils
import os
import sys

import pdb

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
        try:
            self.__h5.get_node(node, name=material.name)
        except tb.NoSuchNodeError:
            pass
        else:
            self.__h5.remove_node(node, name=material.name)
        mat_table = self.__h5.create_table(node, material.name,
                                           obj=material.attinuation)
        mat_table.flush()

    def get_material(self, name):
        node = self.get_node('materials', create=True)
        try:
            mat_table = self.__h5.get_node(node, name=name)
        except tb.NoSuchNodeError, e:
            raise e
        else:
            return Material(name, attinuations=mat_table.read())

    def get_materials(self):
        materials = []
        node = self.get_node('materials', create=True)
        for child in node:
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
    def __init__(self, name, density=None, att_file=None, attinuations=None):
        self.name = name
        self.__density = density
        self.__atts = attinuations

        if att_file is not None:
            self.attinuation = att_file

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
        self.__density = value

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
        self.__atts = array


class CTProtocol(object):
    def __init__(self, name):
        self.__name = None
        self.scan_fov = 50.
        self.sdd = 100.
        self.total_collimation = 3.89
        self.modulation_xy = False
        self.modulation_z = False
        self.al_filtration = 0.
        self.xcare = False
        self.ctdi_air100 = 0.

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


def test():
    db_path = os.path.abspath("C://test//test.h5")
    database = Database(db_path)

    material_dir = os.path.abspath("C://GitHub//OpenDXMC//data//materials//attinuation")
    material_paths = utils.find_all_files([material_dir])
#    for path in material_paths:
#        name = os.path.splitext(os.path.basename(path))[0]
#        material = Material(name, att_file=path)
#        database.add_material(material)
#        database.close()
#    pdb.set_trace()

    materials = database.get_materials()
#    pdb.set_trace()

    ##creating teat pat
    pat = Patient('eple')
    print
    pat.density_array = np.random.uniform(.5, 1.5, size=(25,25,35)).astype(np.double)
    pat.material_array = np.rint(np.random.uniform(0,2, size=(25,25,35))).astype(np.intc)
    pat.material_map = {0:'soft', 1:'lung', 2:'air'}
    database.add_patient(pat, overwrite=True)

    pdb.set_trace()
    pat1 = database.get_patient('eple')
    pdb.set_trace()


#    def specter(self):
#
#
#    def __ev_to_dose(self, ev):


if __name__ == '__main__':
    test()

