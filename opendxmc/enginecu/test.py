# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:56:58 2015

@author: ander
"""

import ctypes as ct
import numpy as np
import os
import sys
import matplotlib.pylab as plt
import pdb
import time


def get_kernel():
    dll_path = os.path.join(os.path.dirname(sys.argv[0]), 'enginelib64.dll')
    print(dll_path)
    os.chdir(os.path.dirname(sys.argv[0]))
    dll = ct.CDLL('enginelib64.dll')
    print(dll)
    
    setup = dll.setup_simulation
    setup.argtypes = [ct.POINTER(ct.c_int), 
                      ct.POINTER(ct.c_double), 
                      ct.POINTER(ct.c_double),
                      ct.POINTER(ct.c_int),
                      ct.POINTER(ct.c_double),
                      ct.POINTER(ct.c_int),
                      ct.POINTER(ct.c_double), 
                      ct.POINTER(ct.c_double)]
    setup.restype = ct.c_void_p
    
    run = dll.run_simulation
    run.argtypes = [ct.POINTER(ct.c_double), ct.c_size_t, ct.c_void_p]
    
    cleanup = dll.cleanup_simulation
    cleanup.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)]
    
    #info = dll.device_info
    
    return setup, run, cleanup



def setup():
    
    material_map = np.zeros((128, 128, 128), dtype="int")
    density_map = np.ones((128, 128, 128), dtype="float64")
    
    energy_imparted = np.zeros((128, 128, 128), dtype="float64")

    
    # lut    
    lut = np.zeros((2, 5, 5), dtype="float64")
    lut[0, 0, :] = np.array([1000, 10000, 50000, 69000, 100000])
    lut[0, 2, :] = np.array([.34, .0246, 0.00101, .00005, .000276])
    lut[0, 3, :] = np.array([6.8, .00277, .000011, .000003, .000000987])
    lut[0, 4, :] = np.array([.05, .385, .3344, .317, .29])
    for i in range(lut.shape[2]):
        lut[0, 1, i] = lut[0, 2:, i].sum()
     
    shape = np.array(material_map.shape, dtype="int")
    spacing = np.array([.1, .1, .1], dtype="float64")
    offset = (-shape*spacing / 2. ).astype("float64")
    lut_shape = np.array(lut.shape, dtype="int")
        
    # test_setup
    kernel_setup, kernel_run, kernel_cleanup = get_kernel()
    
   
#    pdb.set_trace()

    timer = time.clock()

#    pdb.set_trace()
    void_ptr = kernel_setup(
                 shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 spacing.ctypes.data_as(ct.POINTER(ct.c_double)),
                 offset.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 material_map.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 density_map.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 lut_shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 lut.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))
                 )
    
    #void_ptr_c = ct.c_void_p(void_ptr)
    print('Setup time', time.clock()-timer)
    
    n_particles = 512*512
    particles = np.zeros((n_particles, 8), dtype="float64")
    particles[:, 0] = -1000
    particles[:, 1] = 0
    particles[:, 2] = 0
    particles[:, 3] = 1
    particles[:, 4] = 0
    particles[:, 5] = 0
    particles[:, 6] = 70000
    particles[:, 7] = 1
    
    
    
    teller = 0
    timer_tot = time.clock()
    while teller*n_particles < 10**6:
        timer = time.clock()
        kernel_run(particles.ctypes.data_as(ct.POINTER(ct.c_double)), 
                   ct.c_size_t(n_particles), 
                   void_ptr)
        print('histories per second', n_particles / (time.clock()-timer), time.clock()-timer)
        teller += 1
    print("total time for {} histories is {}".format((teller)*n_particles, time.clock()-timer_tot))
        
                   

    timer = time.clock()
    kernel_cleanup(void_ptr, 
                   shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                   energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))) 
    print('cleanup time', time.clock()-timer)
    print(shape)
    print(energy_imparted.max())
    a=energy_imparted
    plt.imshow(np.clip(energy_imparted, 0,1000000).max(axis=1))
    plt.show()
    
    #print(lut)    
    
if __name__ == '__main__':
    setup()

    
    

