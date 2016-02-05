# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:39:41 2015

@author: ander
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:56:58 2015

@author: ander
"""

import ctypes as ct
import numpy as np
import os
import sys



def get_kernel():
    dll_path = os.path.abspath(os.path.dirname(__file__))
#    os.chdir(dll_path)
    if sys.maxsize > 2**32:
        dll = ct.CDLL(os.path.join(dll_path, 'enginelib64.dll'))
    else:
        dll = ct.CDLL(os.path.join(dll_path, 'enginelib32.dll'))
    
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
    
    source = dll.setup_source
    source.argtypes = [ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_double),
                       ct.POINTER(ct.c_double),
                       ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_double), 
                       ct.POINTER(ct.c_int)]
    source.restype = ct.c_void_p
    
    run = dll.run_simulation
    run.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
    
    cleanup = dll.cleanup_simulation
    cleanup.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)]
   
    cleanup_source = dll.cleanup_source
    cleanup_source.argtypes = [ct.c_void_p]
    #info = dll.device_info
    
    return setup, source, run, cleanup, cleanup_source

class Engine(object):
    def __init__(self):
        self.c_simsetup, self.c_sourcesetup, self.crun, self.c_simcleanup, self.c_sourcecleanup = get_kernel()

    def setup_simulation(self, shape, spacing, offset, material_map, density_map, lut_shape, lut, energy_imparted):
        return self.c_simsetup(
                 shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 spacing.ctypes.data_as(ct.POINTER(ct.c_double)),
                 offset.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 material_map.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 density_map.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 lut_shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 lut.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))
                 ) 
                 
    def setup_source(self, source_position, source_direction, scan_axis, sdd, fov, collimation, weight, specter_cpd, specter_energy):
        n_specter = np.array(specter_cpd.shape, dtype='int')
        return self.c_sourcesetup(
                    source_position.ctypes.data_as(ct.POINTER(ct.c_double)),
                    source_direction.ctypes.data_as(ct.POINTER(ct.c_double)),
                    scan_axis.ctypes.data_as(ct.POINTER(ct.c_double)),
                    sdd.ctypes.data_as(ct.POINTER(ct.c_double)),
                    fov.ctypes.data_as(ct.POINTER(ct.c_double)),
                    collimation.ctypes.data_as(ct.POINTER(ct.c_double)),
                    weight.ctypes.data_as(ct.POINTER(ct.c_double)),
                    specter_cpd.ctypes.data_as(ct.POINTER(ct.c_double)),
                    specter_energy.ctypes.data_as(ct.POINTER(ct.c_double)),
                    n_specter.ctypes.data_as(ct.POINTER(ct.c_int))
                    )
                    
    def run(self, source_ptr, n_particles, sim_ptr):

#        try:
        self.crun(source_ptr, 
                   ct.c_size_t(n_particles), 
                   sim_ptr)
#        except OSError as err:
#            print(err)
#            import pdb
#            pdb.set_trace()
          
    def cleanup(self, simulation=None, energy_imparted=None, source=None):
        if simulation:
            if energy_imparted is None:
                raise ValueError("If simulation is specified energy_imparted must also be specified.")
            shape = np.array(energy_imparted.shape, dtype='int')
            self.c_simcleanup(simulation, 
                          shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                          energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double)))
        if source:
            self.c_sourcecleanup(source)
            
#def test():
#    import matplotlib.pylab as plt
#    res = 64
#    material_map = np.zeros((res, res, res), dtype="int")
#    density_map = np.ones((res, res, res), dtype="float64")
#    
#    energy_imparted = np.zeros((res, res, res), dtype="float64")
#    
#    # lut    
#    lut = np.zeros((2, 5, 5), dtype="float64")
#    lut[0, 0, :] = np.array([1000, 10000, 50000, 69000, 100000])
#    lut[0, 2, :] = np.array([.34, .0246, 0.00101, .00005, .000276])
#    lut[0, 3, :] = np.array([6.8, .00277, .000011, .000003, .000000987])
#    lut[0, 4, :] = np.array([.05, .385, .3344, .317, .29])
#    for i in range(lut.shape[2]):
#        lut[0, 1, i] = lut[0, 2:, i].sum()
#     
#    shape = np.array(material_map.shape, dtype="int")
#    spacing = np.array([.1, .1, .1], dtype="float64")
#    offset = (-shape*spacing / 2. ).astype("float64")
#    lut_shape = np.array(lut.shape, dtype="int")
#     
#         
#     
#    s_pos = np.array([100, 0, 0], dtype='float64')
#    s_dir = np.array([-1, 0, 0], dtype='float64')
#    axis = np.array([0, 0, 1], dtype='float64')
#    fov = np.array([50,], dtype='float64')
#    sdd = np.array([100,], dtype='float64')
#    collimation = np.array([4,], dtype='float64')
#    weight = np.array([1,], dtype='float64')
#    specter_cpd = np.cumsum(np.array([1, 1, 1], dtype='float64'))
#    specter_cpd /= specter_cpd.max()
#    specter_energy = np.array([60, 70, 80], dtype='float64')
#    n_specter = np.array([3], dtype='int')
#    
#    # test_setup
#    sim_setup, source_setup, kernel_run, sim_cleanup, source_cleanup = get_kernel()
#    
#    
#   
##    pdb.set_trace()
#
#    timer = time.clock()
#
##    pdb.set_trace()
#    sim_ptr = sim_setup(
#                 shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
#                 spacing.ctypes.data_as(ct.POINTER(ct.c_double)),
#                 offset.ctypes.data_as(ct.POINTER(ct.c_double)), 
#                 material_map.ctypes.data_as(ct.POINTER(ct.c_int)), 
#                 density_map.ctypes.data_as(ct.POINTER(ct.c_double)), 
#                 lut_shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
#                 lut.ctypes.data_as(ct.POINTER(ct.c_double)), 
#                 energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))
#                 )
#
#    source_ptr = source_setup(
#                    s_pos.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    s_dir.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    axis.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    sdd.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    fov.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    collimation.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    weight.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    specter_cpd.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    specter_energy.ctypes.data_as(ct.POINTER(ct.c_double)),
#                    n_specter.ctypes.data_as(ct.POINTER(ct.c_int))
#                    )
#
#    
#    #void_ptr_c = ct.c_void_p(void_ptr)
#    print('Setup time', time.clock()-timer)
#    
#    n_particles = 5000000
#            
#    teller = 0
#    timer_tot = time.clock()
#    while True:
#        
#        timer = time.clock()
#        kernel_run(source_ptr, ct.c_size_t(n_particles), sim_ptr)
#        print('histories per second', n_particles / (time.clock()-timer), time.clock()-timer)
#        teller += 1
#        if teller > 5:
#            break
#    print("total time for {} histories is {}".format((teller)*n_particles, time.clock()-timer_tot))
#        
#                   
#
#    timer = time.clock()
#    sim_cleanup(sim_ptr, 
#                shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
#                energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))) 
#    source_cleanup(source_ptr)
#    print('cleanup time', time.clock()-timer)
#    print(shape)
#    print('max value', energy_imparted[:, : , res//2].max())
#    cut = energy_imparted[:, : , res//2].max()/10000.
#    cut = 100000
#    print('cut', cut)
#    plt.subplot(121)
#    plt.imshow(np.clip(energy_imparted[:, : , res//2], 0,cut))
#    plt.subplot(122)
#    plt.imshow(energy_imparted[:, : , res//2])
#    plt.show()           
#if __name__ == '__main__':
#    test()

    
    

