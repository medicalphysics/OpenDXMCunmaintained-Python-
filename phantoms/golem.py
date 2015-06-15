# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 11:09:12 2015

@author: erlean
"""

"""
Utility functions for parsing phantom data
"""

import os
import sys
import numpy as np
from scipy import interp as interp

HEADER = ('energy', 'rayleigh', 'compton', 'photoelectric', 'ppn', 'ppe',
          'total', 'totalnocoherent')

MAX_ENERGY = 500e3


def _find_attinuation_lut(golem_path, material_map, index_map=None):

    if index_map is None:
        index_map = dict([(k, k) for k in material_map.keys()])


    atts = {}
    for key, value in material_map.items():
        atts[index_map[key]] = _parse_tissue_attinuation(os.path.join(golem_path, value+'.txt'))

    energies = np.array([])
    for a in atts.values():
        energies = np.hstack((energies, np.array(a['energy'])))
    energies = np.unique(energies[energies < 0.5])

    lut = np.empty((len(atts), 5, energies.shape[0]), dtype=np.double)
    for i, a in atts.items():
        lut[i, 0, :] = energies[:]
#        pdb.set_trace()
        for ind, item in enumerate(['total', 'rayleigh', 'photoelectric', 'compton']):
            en = np.array(a['energy'])
            at = np.array(a[item])
            sind = np.argsort(en)
            lut[i, ind+1, :] = interp(energies, en[sind], at[sind])
#        lut[i, 1, :] = interp(energies, a['energy'], a['total'])
#        lut[i, 2, :] = interp(energies, a['energy'], a['rayleigh'])
#        lut[i, 3, :] = interp(energies, a['energy'], a['photoelectric'])
#        lut[i, 4, :] = interp(energies, a['energy'], a['compton'])
    return lut


def _parse_densities(file_path):
    data = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip().split()
            data[line[0]] = float(line[1])
    return data


def _parse_tissue_attinuation(fil):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letters += letters.lower()
    header = ('energy', 'rayleigh', 'compton', 'photoelectric', 'ppn', 'ppe',
              'total', 'totalnocoherent')
    data_list = dict([(h, list()) for h in header])
    with open(fil) as f:
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
    return data_list


def read_golem():
    """
    output:
    spacing, densityarray, lut_material, materialarray, organarray, organmap

    """
    golem_path = os.path.join(os.path.normpath(os.path.dirname(sys.argv[0])), 'phantoms', 'golem')

    # reading golem_array
    try:
        with open(os.path.join(golem_path, 'segm_golem')) as r:
            c = r.read()
    except IOError, e:
        print """Could not find segm_golem binary in folder {0}.
                 You may download the
                 golem voxel phantom from Helmholtz Zentrum research
                 center""".format(golem_path)
        raise e
    header_len = 4096
    a_raw = np.array(c[header_len:], dtype='c',
                     order='C').view(np.uint8).reshape((220, 256, 256))

    organ_array = np.swapaxes(a_raw, 0, 2)
    organ_array = np.swapaxes(organ_array, 0, 1)
    # defining spacing
    spacing = np.array([2.08, 2.08, 8.], dtype=np.double) / 10.

    # getting density
    density = _parse_densities(os.path.join(golem_path, 'densities.txt'))

    density_array = np.zeros_like(organ_array, dtype=np.double)
    material_array = np.zeros_like(organ_array, dtype=np.intc)

    material_map = {}
    material_index_map = {}
    with open(os.path.join(golem_path, 'materialmap.txt')) as r:
        i = 0
        for line in r.readlines():
            material, ind = (line.strip()).split()
            material_map[int(ind)] = material
            material_index_map[int(ind)] = i
            i += 1

    organ_map = {}
    with open(os.path.join(golem_path, 'organmap.txt')) as r:
        for line in r.readlines():
            seg_ind, organ, materials = (line.strip()).split(';')
            material = int(materials[0])
            ind = organ_array == int(seg_ind)

            density_array[ind] = density[material_map[material]]
            material_array[ind] = material_index_map[material]
            organ_map[int(seg_ind)] = organ

    lut = _find_attinuation_lut(golem_path, material_map, material_index_map)

    return spacing, density_array, lut, material_array, organ_array, organ_map


#if __name__ == '__main__':
#    spacing, density_array, lut, material_array, organ_array, organ_map = read_golem()
#
#    from matplotlib import pylab as plt
#
#    k = 110
##    pdb.set_trace()
#    plt.subplot(131)
#    plt.imshow(density_array[:,:,k])
#    plt.subplot(132)
#    plt.imshow(material_array[:,:,k])
#    plt.subplot(133)
#    plt.imshow(organ_array[:,:,k])
#    plt.show()
#    pdb.set_trace()
