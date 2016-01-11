# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:26:24 2015

@author: ander
"""

import numpy as np
import scipy.interpolate
from opendxmc.tube.tungsten import specter
import logging
logger = logging.getLogger('OpenDXMC')

SHUFFLE_SEED = 100



def half_shuffle(arr):
    """
    Shuffles an array in a predictable manner
    """
    assert len(arr.shape) == 1
    n = arr.shape[0]
    shuf = np.zeros_like(arr)
    d = n / 2
    shuf[::2] = arr[d:]
    shuf[1::2] = arr[:d][::-1]
    return shuf


def ct_source_space(simulation, exposure_modulation=None, batch_size=None):
    arglist = ['scan_fov', 'sdd']
    kwarglist = ['start', 'stop', 'exposures', 'histories',
                 'start_at_exposure_no', 'batch_size',
                 ]

    args = [simulation.get(a) for a in arglist]
    args.append(simulation.get('detector_rows') * simulation.get('detector_width'))
    kwargs = {'exposure_modulation': exposure_modulation}
    for a in kwarglist:
        kwargs[a] = simulation.get(a)
    kwargs['rotation_center'] = simulation.get('data_center')
    kwargs['rotation_plane_cosines'] = simulation.get('image_orientation')

#    'rotation_center', kwargs['rotation_plane_cosines']

    if simulation.get('is_spiral'):
        kwargs['pitch'] = simulation.get('pitch')
        phase_func = ct_spiral
    else:
        kwargs['step'] = simulation.get('step')
        phase_func = ct_seq

    s = specter(simulation.get('kV'), filtration_materials='Al',
                filtration_mm=simulation.get('al_filtration'))
    kwargs['energy_specter'] = s

    return phase_func(*args, **kwargs)

def world_image_matrix(orientation):
    iop = np.array(orientation, dtype=np.float).reshape(2, 3).T
    s_norm = np.cross(*iop.T[:])
    R = np.eye(3)
    R[:, :2] = np.fliplr(iop)
    R[:, 2] = s_norm
    return np.linalg.inv(R)

def rotation_z_matrix(alpha):
    return np.array([[np.cos(alpha), - np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]], dtype=np.double)

def ct_spiral(scan_fov, sdd, total_collimation, pitch=1,
              start=0, stop=1, exposures=100, histories=1,
              energy=70000., energy_specter=None,
              batch_size=0, rotation_center=None,
              rotation_plane_cosines = None,
              exposure_modulation=None, start_at_exposure_no=0):
    """Generate CT phase space, return a iterator.

    INPUT:
        scan_fov : float
            scanner field of view in cm
        sdd : float
            source detector distance in cm
        total_collimation : float
            detector collimation or width in cm
        pitch : float
            pitch value of the spiral
        start : float
            start z position for spiral in cm
        stop : float
            stop z position for spiral in cm
        exposures : int
            number of exposures per rotation
        histories : int
            number of photon histories per exposure
        energy : float
            monochromatic photon energy in eV (ignored if energy_specter
            is applied)
        energy_specter : [(N,), (N,)]
            [ndarray(energy), ndarray(intensity)] list/tuple of
            two ndarrays of lenght one, with energy and intensity of specter
        batch_size : int
            number of exposures per batch, if less than histories it is set to
            histories.
        modulation_xy : [(N,), (N,)] (NOT IMPLEMENTED)
            tube current XY modulation, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        exposure_modulation : [(N,), (N,)]
            tube current modulation z axis, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        start_at_exposure_no: int
            Starting at this exposure number, used for resuming a simulation
    OUTPUT:
        Iterator returning ndarrays of shape (8, batch_size),
        one row is equal to photon (start_x, start_y, star_z, direction_x,
        direction_y, direction_z, energy, weight)
    """
    logger.debug('Generating CT spiral phase space')
    if rotation_center is None:
        rotation_center = np.zeros(3, dtype=np.double)
    rotation_center[2] = 0  # we start spiral at start not center
    if rotation_plane_cosines is None:
        rotation_plane_cosines = np.array([1, 0, 0, 0, 1, 0], dtype=np.double)
    print('CT phase space start', start, stop)
    # total number of exposures + one total rotation
    exposures = int(exposures)
    e = int((abs(start - stop) / (pitch * total_collimation) + 1) * exposures)

    if start < stop:
        d_col = total_collimation / 2.
    else:
        d_col = -total_collimation / 2.
    # positions along z for each exposure
    t = np.linspace(start-d_col, stop + d_col, e)
#    # we shuffle the positions to take generate conservative ETA estimates
    t = half_shuffle(t)
#    print('whole t', t)
    # angle for each z position , i.e the x, y coordinates
    ang = t / (pitch * total_collimation) * np.pi * 2.

    # rotation matrix along z-axis for an angle x


    if batch_size is None:
        batch_size = 1
    if batch_size < 1:
        batch_size = 1
    batch_size *= histories

    assert batch_size % histories == 0

    if energy_specter is None:
        energy_specter = [np.array([energy], dtype=np.double),
                          np.array([1.0], dtype=np.double)]
    energy_specter = (energy_specter[0],
                      energy_specter[1] / energy_specter[1].sum())

    specter_cpd = np.cumsum(energy_specter[1])
    specter_cpd /= specter_cpd.max()
    
    specter_energy = energy_specter[0]

#    if modulation_xy is None:
#        mod_xy = lambda x: 1.0
#    else:
#        mod_xy = scipy.interpolate.interp1d(modulation_xy[0], modulation_xy[1],
#                                            copy=False, bounds_error=False,
#                                            fill_value=1.0)

    if exposure_modulation is None:
        mod_z = lambda x: 1.0
    else:
        if np.abs(np.mean(exposure_modulation[:, 1])) > 0.000001:
            exposure_modulation[:, 1] /= np.mean(exposure_modulation[:, 1])

            mod_z = scipy.interpolate.interp1d(exposure_modulation[:, 0],
                                               exposure_modulation[:, 1],
                                               copy=True, bounds_error=False,
                                               fill_value=1.0, kind='nearest')
        else:
            mod_z = lambda x: 1.0

 
    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center[[1, 0, 2]])
    for i in range(start_at_exposure_no, e):
        R = np.dot(M, rotation_z_matrix(ang[i]))

        position = np.dot(R, np.array([-sdd/2., 0, t[i]], dtype='float64')) + rotation_center_image
        direction = np.dot(R, np.array([1., 0, 0], dtype='float64'))
        scan_axis = np.dot(R, np.array([0, 0, 1], dtype='float64'))        
        ret = (position, direction, scan_axis, 
               np.array([sdd], dtype='float64'), 
               np.array([scan_fov], dtype='float64'), 
               np.array([total_collimation], dtype='float64'),
               specter_cpd.astype('float64'), specter_energy.astype('float64'))
        yield ret, i, e

#        ind_b = teller * histories
#        ind_s = (teller + 1) * histories
#
#        ret[0, ind_b:ind_s] = -sdd/2.
#        ret[1, ind_b:ind_s] = 0
#        ret[2, ind_b:ind_s] = t[i]
##        print('t', t[i])
#        ret[0:3, ind_b:ind_s] = np.dot(R, ret[0:3, ind_b:ind_s])
#        for j in range(2):
#            ret[j, ind_b:ind_s] += rotation_center_image[j]
#
#        ret[3, ind_b:ind_s] = sdd / 2.
#        ret[4, ind_b:ind_s] = scan_fov /2 * np.random.uniform(-1., 1., histories)
#        ret[5, ind_b:ind_s] = d_col * np.random.uniform(-1., 1.,
#                                                        histories)
#        ret[3:6, ind_b:ind_s] = np.dot(R, ret[3:6, ind_b:ind_s])
#
#        lenght = np.sqrt(np.sum(ret[3:6, ind_b:ind_s]**2, axis=0))
#        ret[3:6, ind_b:ind_s] /= lenght
#
#        ret[7, ind_b:ind_s] = mod_z(t[i])  # * mod_xy(t[i])
#
#        if ind_s == batch_size:
#            ret[6, :] = np.random.choice(energy_specter[0],
#                                         batch_size,
#                                         p=energy_specter[1])
##            print('phase space pos', ret[2, :])
#            yield ret, i, e
#            teller = 0
#        else:
#            teller += 1
#    if teller > 0:
#        teller -= 1
#    if teller > 0:
#        ret[6, :] = np.random.choice(energy_specter[0],
#                                     batch_size,
#                                     p=energy_specter[1])
#        yield ret[:, :teller * histories], i, e


def ct_seq(scan_fov, sdd, total_collimation, step=1,
              start=0, stop=1, exposures=100, histories=1,
              energy=70000., energy_specter=None,
              batch_size=0, rotation_center=None,
              rotation_plane_cosines = None,
              exposure_modulation=None, start_at_exposure_no=0):
    """Generate CT phase space, return a iterator.

    INPUT:
        scan_fov : float
            scanner field of view in cm
        sdd : float
            source detector distance in cm
        total_collimation : float
            detector collimation or width in cm
        pitch : float
            pitch value of the spiral
        start : float
            start z position for spiral in cm
        stop : float
            stop z position for spiral in cm
        exposures : int
            number of exposures per rotation
        histories : int
            number of photon histories per exposure
        energy : float
            monochromatic photon energy in eV (ignored if energy_specter
            is applied)
        energy_specter : [(N,), (N,)]
            [ndarray(energy), ndarray(intensity)] list/tuple of
            two ndarrays of lenght one, with energy and intensity of specter
        batch_size : int
            number of exposures per batch, if less than histories it is set to
            histories.
        modulation_xy : [(N,), (N,)] (NOT IMPLEMENTED)
            tube current XY modulation, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        exposure_modulation : [(N,), (N,)]
            tube current modulation z axis, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        start_at_exposure_no: int
            Starting at this exposure number, used for resuming a simulation
    OUTPUT:
        Iterator returning ndarrays of shape (8, batch_size),
        one row is equal to photon (start_x, start_y, star_z, direction_x,
        direction_y, direction_z, energy, weight)
    """
    logger.debug('Generating CT seq phase space')
    if rotation_center is None:
        rotation_center = np.zeros(3, dtype=np.double)
    rotation_center[2] = 0  # we start spiral at start not center
    if rotation_plane_cosines is None:
        rotation_plane_cosines = np.array([1, 0, 0, 0, 1, 0], dtype=np.double)

    # total number of exposures + one total rotation
    exposures = int(exposures)
    N = int(np.ceil(abs(start - stop) / step))
    if N <= 0:
        N = 1
    e = N * exposures

    d_col = total_collimation / 2.

    if start > stop:
        start, stop = stop, start
    # positions along z for each exposure
    t = np.empty((e,))
    ang = np.empty((e,))
    for i in range(N):
        t[exposures*i: exposures*(i+1)] = start + step * i
        ang[exposures*i: exposures*(i+1)] = np.linspace(0, 2*np.pi, exposures)

    t = half_shuffle(t)
    ang = half_shuffle(ang)

    if batch_size is None:
        batch_size = 1
    if batch_size < 1:
        batch_size = 1
    batch_size *= histories

    assert batch_size % histories == 0

    if energy_specter is None:
        energy_specter = [np.array([energy], dtype=np.double),
                          np.array([1.0], dtype=np.double)]
    energy_specter = (energy_specter[0],
                      energy_specter[1] / energy_specter[1].sum())
    specter_cpd = np.cumsum(energy_specter[1])
    specter_cpd /= specter_cpd.max()
    specter_energy = energy_specter[0]

#    if modulation_xy is None:
#        mod_xy = lambda x: 1.0
#    else:
#        mod_xy = scipy.interpolate.interp1d(modulation_xy[0], modulation_xy[1],
#                                            copy=False, bounds_error=False,
#                                            fill_value=1.0)

    if exposure_modulation is None:
        mod_z = lambda x: 1.0
    else:
        if np.abs(np.mean(exposure_modulation[:, 1])) > 0.000001:
            exposure_modulation[:, 1] /= np.mean(exposure_modulation[:, 1])

            mod_z = scipy.interpolate.interp1d(exposure_modulation[:, 0],
                                               exposure_modulation[:, 1],
                                               copy=True, bounds_error=False,
                                               fill_value=1.0, kind='nearest')
        else:
            mod_z = lambda x: 1.0

    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center[[1, 0, 2]])
    for i in range(start_at_exposure_no, e):
        R = np.dot(M, rotation_z_matrix(ang[i]))

        position = np.dot(R, np.array([-sdd/2., 0, t[i]], dtype='float64')) + rotation_center_image
        direction = np.dot(R, np.array([1., 0, 0], dtype='float64'))
        scan_axis = np.dot(R, np.array([0, 0, 1], dtype='float64'))        
        ret = (position, direction, scan_axis, 
               np.array([sdd], dtype='float64'), 
               np.array([scan_fov], dtype='float64'), 
               np.array([total_collimation], dtype='float64'),
               specter_cpd.astype('float64'), specter_energy.astype('float64'))
        yield ret, i, e

