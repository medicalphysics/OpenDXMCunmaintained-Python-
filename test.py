# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:17:40 2015

@author: erlean
"""
import logging
logger = logging.getLogger('OpenDXMC')
logger.addHandler(logging.StreamHandler())
logger.setLevel(10)
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from opendxmc.database import import_ct_series
from opendxmc.runner import obtain_ctdiw_conversion_factor
from opendxmc.runner import obtain_ctdiair_conversion_factor
from opendxmc.database.import_phantoms import read_phantoms
from opendxmc.database.import_materials import get_stored_materials


import pdb

def test_obtain_ctdiw_conversion_factor():
    
    simulation = {'scan_fov':50., 'sdd':100, 'exposures':360, 
                  'histories':100000, 'batch_size':1, 'kV':120, 
                  'al_filtration':5, 'detector_rows':64,
                  'detector_width':0.06, 'ctdi_w100':15., 'name':'test', 'ctdi_air100':15.}
    materials = get_stored_materials()
    for mat in materials:
        if mat.name == 'pmma':
            pmma = mat
        elif mat.name =='air':
            air = mat
    
#    im = obtain_ctdiw_conversion_factor(simulation, pmma, air)
    corr_arr = []
    for i in range(9):
        im = obtain_ctdiw_conversion_factor(simulation, pmma, air)
        corr_arr.append(simulation['conversion_factor_ctdiw'])  
#        im = obtain_ctdiair_conversion_factor(simulation, air)
#        corr_arr.append(simulation['conversion_factor_ctdiair'])  
        plt.subplot(3, 3, i+1)
        plt.imshow(np.mean(im[:,:,:], axis=-1))
        
#        break
    print('mean', sum(corr_arr)/len(corr_arr), 'std', np.std(corr_arr))
    plt.show()
    plt.plot(corr_arr, 'o')
    plt.plot(corr_arr)
    plt.show()
    pdb.set_trace()
    
if __name__ == '__main__':
    test_obtain_ctdiw_conversion_factor()
