# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:27:26 2015

@author: erlean
"""
import numpy as np
from engine import score_energy
from tube.tungsten import specter
import tables as tb
from multiprocessing import Process, Queue
import os
import sys


class Database(object):
    def __init__(self, db_file_path):
        self.__dbpath = os.path.abspath(db_file_path)
        self.__filters = tb.Filters(complevel=7, complib='zlib')
        self.__h5 = None
        if not self.database_valid():
            raise ValueError("Database path is not writeable")

    def open(self, mode='a'):
        if self.__h5 is None:
            self.__h5 = tb.open_file(self.__dbfile, mode=mode,
                                     filters=self.__filters)
            return

        if not self.__h5.isopen:
            self.__h5 = tb.open_file(self.__dbfile, mode=mode,
                                     filters=self.__filters)
            return
        if self.__h5.mode != mode:
            self.__h5.close()
            self.__h5 = tb.open_file(self.__dbfile, mode=mode,
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

    def get_node(self,name, where=None, create=False):
        if where is None:
            where = self.__h5.root
        self.open()
        try:
            self.__h5.get_node(where, name)
        except tb.NoSuchNode:
            if create:
                self.__h5.create_group(where, name, createparents=True)

    def add_material(self, material):
        node = self.get_node('materials', create=True)
        self.__h5.create_table(node, material.name, obj=materials.attinuation)


    def __del__(self):
        self.close()




class Material(object):
    def __init__(self, name):
        self.__name = name
        self.__density = None
        self.__atts = None

    @property
    def name(self):
        return self.__name

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
        header = ('energy', 'rayleigh', 'compton', 'photoelectric', 'ppn', 'ppe',
                  'total', 'totalnocoherent')
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




class Simulation(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    p = "C://GitHub//OpenDXMC//phantoms//golem//lung.txt"
    m = Material('lung')
    m.attinuation = p
    import pdb
    pdb.set_trace()

