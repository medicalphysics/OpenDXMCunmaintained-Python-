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
import tables as tb

import pdb
import pylab as plt

HEADER = ('energy', 'rayleigh', 'compton', 'photoelectric', 'ppn', 'ppe',
          'total', 'totalnocoherent')

MAX_ENERGY = 500e3

def save_array_to_h5(path, array, name):
    
    h5 = tb.open_file(path, mode='a')
    try:
        h5.get_node(h5.root, name)
    except tb.NoSuchNodeError:
        pass
    else:
        h5.remove_node(h5.root, name, True)
    h5.create_carray(h5.root, name=name, obj=array)
    h5.close()

def load_array_from_h5(path, name):
    try:
        h5 = tb.open_file(path, mode='r')
    except IOError, e:
        raise e
    try:
        arr_n = h5.get_node(h5.root, name)
    except tb.NoSuchNodeError:
        h5.close()
        raise ValueError('No node named {0} in database'.format(name))
    else:
        arr = arr_n.read()
        h5.close()
        return arr

def save_dict_to_h5(path, d, name, dtype=None):
    h5 = tb.open_file(path, mode='a')
    try:
        h5.get_node(h5.root, name)
    except tb.NoSuchNodeError:
        pass
    else:
        h5.remove_node(h5.root, name, True)
    # creating structured array:
    n = len(d)
    if dtype is None:
        dtype = {'names':['index', 'tissue'], 'formats': ['i4', 'a64']}
    arr = np.zeros(n, dtype=dtype)
    teller = 0
    for key, value in d.items():
        arr['tissue'][teller] = value
        arr['index'][teller] = key
        teller += 1
    h5.create_table(h5.root, name, description=arr)
    h5.close()

def load_dict_from_h5(path, name, key='index', value='tissue'):
    try:
        h5 = tb.open_file(path, mode='r')
    except IOError, e:
        raise e
    try:
        t = h5.get_node(h5.root, name)
    except tb.NoSuchNodeError:
        h5.close()
        raise ValueError('No node named {0} in database'.format(name))
    else:
        try:
            d = dict(zip(t.col(key), t.col(value)))
        except KeyError:
            h5.close()
            raise ValueError('Key {0} or value {1} is missing from table ' 
                'names {2}'.format(key, value, t.colnames))
        else:
            h5.close()
            return d
    h5.close()
        
    

def _find_attinuation_lut(golem_path, material_map):

    atts = {}
    for key, value in material_map.items():
        atts[key] = _parse_tissue_attinuation(os.path.join(golem_path,
                                                           value+'.txt'))

    energies = np.array([])
    for a in atts.values():
        energies = np.hstack((energies, np.array(a['energy'])))
    energies = np.unique(energies[energies < 0.5])

    lut = np.empty((len(atts), 5, energies.shape[0]), dtype=np.double)
    for i, a in atts.items():
        lut[i, 0, :] = energies[:]
#        pdb.set_trace()
        for ind, item in enumerate(['total', 'rayleigh', 'photoelectric',
                                    'compton']):
            en = np.array(a['energy'])
            at = np.array(a[item])
            sind = np.argsort(en)
            lut[i, ind+1, :] = interp(energies, en[sind], at[sind])
#        lut[i, 1, :] = interp(energies, a['energy'], a['total'])
#        lut[i, 2, :] = interp(energies, a['energy'], a['rayleigh'])
#        lut[i, 3, :] = interp(energies, a['energy'], a['photoelectric'])
#        lut[i, 4, :] = interp(energies, a['energy'], a['compton'])
    lut[:, 0, :] *= 1.e6
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


def read_golem(load=True, save=True):
    """
    output:
    spacing, densityarray, lut_material, materialarray, organarray, organmap

    """
    golem_path = os.path.join(os.path.normpath(os.path.dirname(sys.argv[0])),
                              'phantoms', 'golem')
    golem_h5 = os.path.join(golem_path, 'golem.h5')                       
    # We will read a saved golem if allowed by load keyword
    
    if load == True:
        obj = []
        for name in ['spacing', 'density_array', 'lut', 'material_array', 
                     'organ_array']:
            try:
                obj.append(load_array_from_h5(golem_h5, name))
            except ValueError, e:
                print e
                break
            except IOError, e:
                print e
                break
        else:
            try:
                obj.append(load_dict_from_h5(golem_h5, 'organ_map', 
                                             key='index', 
                                             value='tissue'))
                obj.insert(4, load_dict_from_h5(golem_h5, 'material_map', 
                                             key='index', 
                                             value='tissue'))   
            except ValueError, e:
                print e
            except IOError, e:
                print e
            else:    
                return obj

#    golem_path = os.path.join(os.path.normpath(os.path.dirname(sys.argv[0])), 'golem')

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
    density_map = _parse_densities(os.path.join(golem_path, 'densities.txt'))

    density_array = np.zeros_like(organ_array, dtype=np.double)
    material_array = np.zeros_like(organ_array, dtype=np.intc)
    organ_map = {}
    material_map = {}

    with open(os.path.join(golem_path, 'materialmap.txt')) as r:
        for line in r.readlines():
            material, ind = (line.strip()).split()
            material_map[int(ind)] = material

    with open(os.path.join(golem_path, 'organmap.txt')) as r:
        for line in r.readlines():
            seg_ind, organ, materials = (line.strip()).split(';')
            material = int(materials[0])
            ind = organ_array == int(seg_ind)

            density_array[ind] = density_map[material_map[material]]
            material_array[ind] = material
            organ_map[int(seg_ind)] = organ

    material_index = np.unique(material_array)
    material_array_red = np.zeros_like(organ_array, dtype=np.intc)
    material_map_red = {}
    for i in range(material_index.shape[0]):
        material_map_red[i] = material_map[material_index[i]]
        ind = material_array == material_index[i]
        material_array_red[ind] = i

    lut = _find_attinuation_lut(golem_path, material_map_red)
        
    if save == True:            
        saveobj = {'spacing': spacing,
                   'density_array': density_array,
                   'lut': lut,
                   'material_array': material_array_red,
                   'material_map': material_map_red,
                   'organ_array': organ_array,
                   'organ_map': organ_map,
                   }
        try:
            for key, value in saveobj.items():
                if isinstance(value, np.ndarray):
                    save_array_to_h5(golem_h5, value, key)
                elif isinstance(value, dict):
                    save_dict_to_h5(golem_h5, value, key)        
        except ValueError, e:
            print 'INFO: Error in saving phantom', e
        except IOError, e:
            print ('INFO: Error in opening or creating database '
                'file {0}'.format(golem_h5)), e

    return spacing, density_array, lut, material_array_red, material_map_red,  organ_array, organ_map


if __name__ == '__main__':
    spacing, density_array, lut, material_array, organ_array, organ_map = read_golem()

    from matplotlib import pylab as plt

    k = 110
#    pdb.set_trace()
    plt.subplot(131)
    plt.imshow(density_array[:,:,k])
    plt.subplot(132)
    plt.imshow(material_array[:,:,k])
    plt.subplot(133)
    plt.imshow(organ_array[:,:,k])
    plt.show()
    pdb.set_trace()
