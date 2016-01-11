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
from mpl_toolkits.mplot3d import Axes3D
import pdb
import time
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
                     
def ct_phase_space3(particle_pos, particle_direction, scan_axis, sdd, fov, collimation, specter_pd, specter_energy, particles, n_particles):
    
    
    particles[:, :3] = np.outer(np.ones(n_particles), particle_pos)
    
    v_rot = np.cross(scan_axis, particle_direction)
    
    v_rot_lenght = np.random.uniform(low=-fov*2/sdd, high=fov*2/sdd, size=n_particles)
    v_z_lenght = np.random.uniform(low=-collimation/(2*fov), high=collimation/(2*fov), size=n_particles)
    
    particles[:, 3:6] = np.outer(np.ones(n_particles), particle_direction)
    particles[:, 3:6] += np.outer(v_rot_lenght, v_rot)
    particles[:, 3:6] += np.outer(v_z_lenght, scan_axis)
    particles[:, 3:6] /= np.outer((1 + v_rot_lenght**2 + v_z_lenght**2)**.5, np.ones(3))
    
    particles[:, 6] = np.random.choice(specter_energy, size=n_particles, p=specter_pd)
    particles[:, 7] = 1
    
        
        
        
    
def ct_phase_space2(particle_pos, particle_direction, scan_axis, sdd, fov, collimation, specter_cpd, specter_energy, particles, n_particles):
    
    
    for i in range(n_particles):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        r3 = np.random.uniform() - 0.00001
        v_rot = np.cross(scan_axis, particle_direction)
        
        v_rot_lenght = fov * 2 / sdd * (r1 - 0.5) * 2
        v_z_lenght = collimation/(2*fov) * (r2 - 0.5) * 2
        v_rot *= v_rot_lenght
        
        v_z = scan_axis * v_z_lenght
        particles[i, 3:6] = particle_direction + v_rot + v_z
        particles[i, 3:6] /= np.sum(particles[i, 3:6]**2)**.5
        print(particles[i, 3:6], np.sum(particles[i, 3:6]**2)**.5)
        particles[i, 0:3] = particle_pos 
#        particles[i, 3:6] = particle_direction
                
        j = 0
        while True:
            if r3 < specter_cpd[j]:
                particles[i, 6] = specter_energy[j]
                break
            j+=1
        
        particles[i, 7] = 1

def ct_phase_space(particle_pos, particle_direction, sdd, fov, collimation, specter_cpd, specter_energy, particles, n_particles):
    
    
    for i in range(n_particles):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        r3 = np.random.uniform() - 0.00001
        v_rot = np.array([particle_direction[1], particle_direction[0], particle_direction[2]])
        v_rot_lenght = fov * 2 / sdd * (r1 - 0.5) * 2
        v_rot *= v_rot_lenght
        
        particles[i, 3] =  v_rot[0] + particle_direction[0]
        particles[i, 4] =  v_rot[1] + particle_direction[1]

        v_z_lenght = collimation/(2*fov) * (r2 - 0.5) * 2

        particles[i, 5] +=  v_z_lenght
        
        particles[i, 3:6] /= np.sum(particles[i, 3:6]**2)**.5
        print(particles[i, 3:6], np.sum(particles[i, 3:6]**2)**.5)
        particles[i, 0:3] = particle_pos 
#        particles[i, 3:6] = particle_direction
                
        j = 0
        while True:
            if r3 < specter_cpd[j]:
                particles[i, 6] = specter_energy[j]
                break
            j+=1
        
        particles[i, 7] = 1
        
            

def test_phase_space():

    t_ang = np.pi/2    
    z_pos = 1
    plane_vx = np.array([1, 0, 0]) 
    plane_vy = np.array([0, 1, 0])
    plane_vz = np.cross(plane_vx, plane_vy) 
    fov = 5.
    collimation = .4
    sdd = 10
    

    
    fig =plt.figure()
    ax = fig.add_subplot(111,projection='3d')
        
    isosenter = np.array([0, 0, 0])
    

    n_exposures = 32    
    teller =0
    a_pos = np.zeros((3, n_exposures)) 
    for t_ang, t_pos in zip(np.linspace(0, 6*np.pi, n_exposures),np.linspace(0, 1, n_exposures)):
        pos = (sdd / 2 * np.sin(t_ang)*plane_vx+ sdd/2*np.cos(t_ang)*plane_vy +plane_vz*t_pos) + isosenter
        direction = -(np.sin(t_ang) * plane_vx + np.cos(t_ang) * plane_vy)
        a_pos[:,teller] = pos
        n_particles = 100
        particles = np.zeros((n_particles, 8))
        
        specter = np.arange(70)
        specter_energy = np.arange(70)
        specter_cpd = np.cumsum(specter/specter.sum())
        specter_pd = specter/specter.sum()        
        
#        ct_phase_space2(pos, direction, plane_vz ,sdd, fov, collimation, specter_cpd, specter_energy, particles, n_particles)
        ct_phase_space3(pos, direction, plane_vz ,sdd, fov, collimation, specter_pd, specter_energy, particles, n_particles)
        
        
    
        
        teller += 1
        
        for i in range(n_particles):
            ax.plot([particles[i, 0], particles[i, 0] + particles[i, 3]],
                    [particles[i, 1], particles[i, 1] + particles[i, 4]],
                    zs=[particles[i, 2], particles[i, 2] + particles[i, 5]])
#    plt.subplot(122)
#    for i in range(n_particles):
#        plt.plot([particles[i, 0], particles[i, 0] + particles[i, 3]],[particles[i, 2], particles[i, 2] + particles[i, 5]])
    ax.plot(a_pos[0,:], a_pos[1,:], a_pos[2,:])
#    ax.set_xlim([-fov, fov])
#    ax.set_ylim([-fov, fov])
#    ax.set_zlim([0, 2])
    plt.show()
    


def get_kernel():
    dll_path = os.path.abspath(os.path.dirname(__file__))
    
    os.chdir(dll_path)
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
    
    source = dll.setup_source
    source.argtypes = [ct.POINTER(ct.c_double), 
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
    
    
    
def setup():
    res = 64
    material_map = np.zeros((res, res, res), dtype="int")
    density_map = np.ones((res, res, res), dtype="float64")
    
    energy_imparted = np.zeros((res, res, res), dtype="float64")
    
    # lut    
    lut = np.zeros((2, 5, 5), dtype="float64")
    lut[0, 0, :] = np.array([1000, 10000, 50000, 69000, 100000])
    lut[0, 2, :] = np.array([.34, .0246, 0.00101, .00005, .000276])
    lut[0, 3, :] = np.array([6.8, .00277, .000011, .000003, .000000987])
    lut[0, 4, :] = np.array([.05, .385, .3344, .317, .29])
    for i in range(lut.shape[2]):
        lut[0, 1, i] = lut[0, 2:, i].sum()
     
    shape = np.array(material_map.shape, dtype="int")
    spacing = np.array([.01, 1, 1], dtype="float64")
    offset = (-shape*spacing / 2. ).astype("float64")
    lut_shape = np.array(lut.shape, dtype="int")
     
    s_pos = np.array([50, 0, 0], dtype='float64')
    s_dir = np.array([-1, 0, 0], dtype='float64')
    axis = np.array([0, 0, 1], dtype='float64')
    fov = np.array([50,], dtype='float64')
    sdd = np.array([100,], dtype='float64')
    collimation = np.array([4,], dtype='float64')
    specter_cpd = np.cumsum(np.array([1, 1, 1], dtype='float64'))
    specter_cpd /= specter_cpd.max()
    specter_energy = np.array([60, 70, 80], dtype='float64')
    n_specter = np.array([3], dtype='int')
    
    # test_setup
    sim_setup, source_setup, kernel_run, sim_cleanup, source_cleanup = get_kernel()
    
    
   
#    pdb.set_trace()

    timer = time.clock()

#    pdb.set_trace()
    sim_ptr = sim_setup(
                 shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 spacing.ctypes.data_as(ct.POINTER(ct.c_double)),
                 offset.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 material_map.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 density_map.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 lut_shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                 lut.ctypes.data_as(ct.POINTER(ct.c_double)), 
                 energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))
                 )

    source_ptr = source_setup(
                    s_pos.ctypes.data_as(ct.POINTER(ct.c_double)),
                    s_dir.ctypes.data_as(ct.POINTER(ct.c_double)),
                    axis.ctypes.data_as(ct.POINTER(ct.c_double)),
                    sdd.ctypes.data_as(ct.POINTER(ct.c_double)),
                    fov.ctypes.data_as(ct.POINTER(ct.c_double)),
                    collimation.ctypes.data_as(ct.POINTER(ct.c_double)),
                    specter_cpd.ctypes.data_as(ct.POINTER(ct.c_double)),
                    specter_energy.ctypes.data_as(ct.POINTER(ct.c_double)),
                    n_specter.ctypes.data_as(ct.POINTER(ct.c_int))
                    )

    
    #void_ptr_c = ct.c_void_p(void_ptr)
    print('Setup time', time.clock()-timer)
    
    n_particles = 5000000
            
    teller = 0
    timer_tot = time.clock()
    while True:
        timer = time.clock()
        kernel_run(source_ptr, ct.c_size_t(n_particles), sim_ptr)
        print('histories per second', n_particles / (time.clock()-timer), time.clock()-timer)
        teller += 1
        if teller > 5:
            break
    print("total time for {} histories is {}".format((teller)*n_particles, time.clock()-timer_tot))
        
                   

    timer = time.clock()
    sim_cleanup(sim_ptr, 
                shape.ctypes.data_as(ct.POINTER(ct.c_int)), 
                energy_imparted.ctypes.data_as(ct.POINTER(ct.c_double))) 
    source_cleanup(source_ptr)
    print('cleanup time', time.clock()-timer)
    print(shape)
    print(energy_imparted.max())
    a=energy_imparted
    plt.subplot(121)
    plt.imshow(np.clip(energy_imparted, 0,1000000).max(axis=-1))
    plt.subplot(122)
    plt.imshow(energy_imparted.max(axis=-1))
    plt.show()
    
    #print(lut)    
    
if __name__ == '__main__':
#    test_phase_space()
    setup()

    
    

