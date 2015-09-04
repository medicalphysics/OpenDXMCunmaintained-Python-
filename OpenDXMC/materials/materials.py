# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:15:30 2015

@author: erlean
"""
import numpy as np
import logging

import pdb

logger = logging.getLogger('OpenDXMC')


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
        self.__organic = organic

        if att_file is not None:
            self.attinuation = att_file
        if density_file is not None:
            self.density_from_file(density_file)
        if organic_file is not None:
            self.organic_from_file(organic_file)

    def numpy_dtype(self):
        return np.dtype({'names': ['name', 'density', 'organic'],
                         'formats': ['a64', np.double, np.bool]})
    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        if isinstance(value, bytes):
            value = str(value, encoding='utf-8')
        else:
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
