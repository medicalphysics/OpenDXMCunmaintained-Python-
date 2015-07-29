# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:42:38 2015

@author: erlean
"""
import os
import sys
import numpy as np

import pdb


def test():
    path = "C://GitHub//OpenDXMC//data//phantoms//golem//segm_golem"
    arr = np.fromfile(path, dtype=np.uint8)
    arr = arr[4096:]
    print(arr[0:10])
    pdb.set_trace()
    

def read_golem(path=None):
    """
    output:
    spacing, densityarray, lut_material, materialarray, organarray, organmap

    """

    if path:
        golem_path = os.path.join(path, 'golem')
    else:
        golem_path = os.path.join(os.path.normpath(os.path.dirname(sys.argv[0])),
                              'phantoms', 'golem')

#    golem_path = os.path.join(os.path.normpath(os.path.dirname(sys.argv[0])), 'golem')

    # reading golem_array
    try:
        c = np.fromfile(os.path.join(golem_path, 'segm_golem'), dtype=np.uint8)
#        with open(os.path.join(golem_path, 'segm_golem'), "rb") as r:
#            c = r.read().decode('ascii')
    except IOError:
        err_msg =  """Could not find segm_golem binary in folder {0}.
                      You may download the
                      golem voxel phantom from Helmholtz Zentrum research
                      center""".format(golem_path)
        raise IOError(err_msg)
    header_len = 4096
    a_raw = c[header_len:].reshape((220, 256, 256))

    organ_array = np.swapaxes(a_raw, 0, 2)
    organ_array = np.swapaxes(organ_array, 0, 1)
    # defining spacing
    spacing = np.array([2.08, 2.08, 8.], dtype=np.double) / 10.

    # getting density

    material_array = np.zeros_like(organ_array, dtype=np.intc)
    organ_map = {}
    material_map = {}

    with open(os.path.join(golem_path, 'organmap.txt')) as r:
        for line in r.readlines():
            seg_ind, organ, materials = (line.strip()).split(';')
            material = int(materials[0])
            ind = organ_array == int(seg_ind)

            organ_map[int(seg_ind)] = organ
            material_map[int(material)] = ""
            material_array[ind] = material
            organ_map[int(seg_ind)] = organ
        
    with open(os.path.join(golem_path, 'materialmap.txt')) as r:
        for line in r.readlines():
            material, ind_s = (line.strip()).split(';')
            ind = int(ind_s)
            material_map[ind] = material

    return spacing, material_array, material_map,  organ_array, organ_map

if __name__ == '__main__':
    test()