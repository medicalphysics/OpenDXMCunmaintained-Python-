# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:47:01 2015

@author: erlean
"""

import numpy as np
import scipy.interpolate
from opendxmc.tube.tungsten import specter
import logging
logger = logging.getLogger('OpenDXMC')
import pdb
import random
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


def ct_phase_space(simulation, batch_size=None):
    arglist = ['scan_fov', 'sdd', 'total_collimation']
    kwarglist = ['start', 'stop', 'exposures', 'histories',
                 'exposure_modulation', 'start_at_exposure_no', 'batch_size',
                 ]

    args = [getattr(simulation, a) for a in arglist]
    kwargs = {}
    for a in kwarglist:
        kwargs[a] = getattr(simulation, a)
    kwargs['rotation_center'] = simulation.data_center
    kwargs['rotation_plane_cosines'] = simulation.image_orientation
#    'rotation_center', kwargs['rotation_plane_cosines']

    if simulation.is_spiral:
        kwargs['pitch'] = simulation.pitch
        phase_func = ct_spiral
    else:
        kwargs['step'] = simulation.step
        phase_func = ct_seq

    s = specter(simulation.kV, filtration_materials='Al',
                filtration_mm=simulation.al_filtration)
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

    teller = 0
    ret = np.zeros((8, batch_size), dtype=np.double)
    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center)
    for i in range(start_at_exposure_no, e):
        R = np.dot(M, rotation_z_matrix(ang[i]))
#        R = rotation_z_matrix(ang[i])
        ind_b = teller * histories
        ind_s = (teller + 1) * histories

        ret[0, ind_b:ind_s] = -sdd/2.
        ret[1, ind_b:ind_s] = 0
        ret[2, ind_b:ind_s] = t[i]
#        print('t', t[i])
        ret[0:3, ind_b:ind_s] = np.dot(R, ret[0:3, ind_b:ind_s])
        for j in range(2):
            ret[j, ind_b:ind_s] += rotation_center_image[j]

        ret[3, ind_b:ind_s] = sdd / 2.
        ret[4, ind_b:ind_s] = scan_fov /2 * np.random.uniform(-1., 1., histories)
        ret[5, ind_b:ind_s] = d_col * np.random.uniform(-1., 1.,
                                                        histories)
        ret[3:6, ind_b:ind_s] = np.dot(R, ret[3:6, ind_b:ind_s])

        lenght = np.sqrt(np.sum(ret[3:6, ind_b:ind_s]**2, axis=0))
        ret[3:6, ind_b:ind_s] /= lenght

        ret[7, ind_b:ind_s] = mod_z(t[i])  # * mod_xy(t[i])

        if ind_s == batch_size:
            ret[6, :] = np.random.choice(energy_specter[0],
                                         batch_size,
                                         p=energy_specter[1])
#            print('phase space pos', ret[2, :])
            yield ret, i, e
            teller = 0
        else:
            teller += 1
    if teller > 0:
        teller -= 1
    if teller > 0:
        ret[6, :] = np.random.choice(energy_specter[0],
                                     batch_size,
                                     p=energy_specter[1])
        yield ret[:, :teller * histories], i, e


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
    N = int(np.floor(abs(start - stop) / step))
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

    teller = 0
    ret = np.zeros((8, batch_size), dtype=np.double)
    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center)
    for i in range(start_at_exposure_no, e):
        R = np.dot(M, rotation_z_matrix(ang[i]))
#        R = rotation_z_matrix(ang[i])
        ind_b = teller * histories
        ind_s = (teller + 1) * histories

        ret[0, ind_b:ind_s] = -sdd/2.
        ret[1, ind_b:ind_s] = 0
        ret[2, ind_b:ind_s] = t[i]
#        print('t', t[i])
        ret[0:3, ind_b:ind_s] = np.dot(R, ret[0:3, ind_b:ind_s])
        for j in range(2):
            ret[j, ind_b:ind_s] += rotation_center_image[j]

        ret[3, ind_b:ind_s] = sdd / 2.
        ret[4, ind_b:ind_s] = scan_fov /2 * np.random.uniform(-1., 1., histories)
        ret[5, ind_b:ind_s] = d_col * np.random.uniform(-1., 1.,
                                                        histories)
        ret[3:6, ind_b:ind_s] = np.dot(R, ret[3:6, ind_b:ind_s])

        lenght = np.sqrt(np.sum(ret[3:6, ind_b:ind_s]**2, axis=0))
        ret[3:6, ind_b:ind_s] /= lenght

        ret[7, ind_b:ind_s] = mod_z(t[i])  # * mod_xy(t[i])

        if ind_s == batch_size:
            ret[6, :] = np.random.choice(energy_specter[0],
                                         batch_size,
                                         p=energy_specter[1])
#            print('phase space pos', ret[2, :])
            yield ret, i, e
            teller = 0
        else:
            teller += 1
    if teller > 0:
        teller -= 1
    if teller > 0:
        ret[6, :] = np.random.choice(energy_specter[0],
                                     batch_size,
                                     p=energy_specter[1])
        yield ret[:, :teller * histories], i, e


#def ct_seq(scan_fov, sdd, total_collimation,
#           start=0, stop=0, step=0,  exposures=100, histories=1,
#           energy=70000., energy_specter=None,
#           batch_size=None, exposure_modulation=None, start_at_exposure_no=0):
#    """Generate CT sequential phase space, returns an iterator.
#
#    INPUT:
#        scan_fov : float
#            scanner field of view in cm
#        sdd : float
#            source detector distance in cm
#        total_collimation : float
#            detector collimation or width in cm
#        start : float
#            start z position for spiral in cm
#        stop : float
#            stop z position for spiral in cm
#        step : float > 0
#            distance between each slice
#        exposures : int
#            number of exposures per rotation
#        histories : int
#            number of photon histories per exposure
#        energy : float
#            monochromatic photon energy in eV (ignored if energy_specter
#            is applied)
#        energy_specter : [(N,), (N,)]
#            [ndarray(energy), ndarray(intensity)] list/tuple of
#            two ndarrays of lenght one, with energy [eV] and intensity of specter
#        batch_size : int
#            number of exposures per batch
#        exposure_modulation : [(N,), (N,)]
#            tube current modulation z axis, list/tuple of
#            (ndarray(position), ndarray(scale_factors))
#        start_at_exposure_no: int
#            Starting at this exposure number, used for resuming a simulation
#    OUTPUT:
#        Iterator returning ndarrays of shape (8, batch_size),
#        one row is equal to photon (start_x, start_y, star_z, direction_x,
#        direction_y, direction_z, energy, weight)
#    """
#    logger.debug('Generating CT sequential phase space')
#    if start < stop:
#        d_col = total_collimation / 2.
#    else:
#        d_col = -total_collimation / 2.
#    # total number of exposures + one total rotation
#    if step <= 0:
#        step = total_collimation
#    exposures = int(exposures)
#    s = int(np.ceil(np.abs(start - stop) / float(step)))
#    if s == 0:
#        s = 1
#
#    e = s * exposures
#    t = np.zeros(e)
#    if start < stop:
#        dstep = step
#    else:
#        dstep =- step
#    for i in range(s):
#        t[i*exposures:(i+1)*exposures] = start + i*dstep
#
##    # we randomize the positions not make the progress bar jump
##    t = np.random.permutation(t)
#    # angle for each z position , i.e the x, y coordinates
#    ang = np.repeat(np.linspace(0, 2*np.pi, exposures), s)
#
#    # rotation matrix along z-axis for an angle x
#    rot = lambda x: np.array([[np.cos(x), - np.sin(x), 0],
#                              [np.sin(x), np.cos(x), 0],
#                              [0, 0, 1]], dtype=np.double)
#
#    if batch_size is None:
#        batch_size = 1
#    if batch_size < 1:
#        batch_size =1
#    batch_size *= histories
#
#
#    if energy_specter is None:
#        energy_specter = [np.array([energy], dtype=np.double),
#                          np.array([1.0], dtype=np.double)]
#    energy_specter = (energy_specter[0],
#                      energy_specter[1] / energy_specter[1].sum())
#
##    if modulation_xy is None:
##        mod_xy = lambda x: 1.0
##    else:
##        mod_xy = scipy.interpolate.interp1d(modulation_xy[0], modulation_xy[1],
##                                            copy=False, bounds_error=False,
##                                            fill_value=1.0)
#
#    if exposure_modulation is None:
#        mod_z = lambda x: 1.0
#    else:
#        exposure_modulation[:, 1] /= np.mean(exposure_modulation[:, 1])
#        mod_z = scipy.interpolate.interp1d(exposure_modulation[:, 0],
#                                           exposure_modulation[:, 1],
#                                           copy=True, bounds_error=False,
#                                           fill_value=1.0, kind='nearest')
#
#    teller = 0
#    ret = np.zeros((8, batch_size), dtype=np.double)
#
#    for i in range(start_at_exposure_no, e):
#        R = rot(ang[i])
##        pdb.set_trace()
#        ind_b = teller * histories
#        ind_s = (teller + 1) * histories
#
#        ret[1, ind_b:ind_s] = 0
#        ret[0, ind_b:ind_s] = -sdd/2.
#        ret[2, ind_b:ind_s] = t[i]
#        ret[0:3, ind_b:ind_s] = np.dot(R, ret[0:3, ind_b:ind_s])
#
#        ret[3, ind_b:ind_s] = sdd / 2.
#        ret[4, ind_b:ind_s] = scan_fov * np.random.uniform(-1., 1., histories)
#        ret[5, ind_b:ind_s] = t[i] + d_col * np.random.uniform(-1., 1.,
#                                                               histories)
#        ret[3:6, ind_b:ind_s] = np.dot(R, ret[3:6, ind_b:ind_s])
#        lenght = np.sqrt(np.sum(ret[3:6, ind_b:ind_s]**2, axis=0))
#        ret[3:6, ind_b:ind_s] /= lenght
#
#        ret[7, ind_b:ind_s] = mod_z(t[i])
#
#        if ind_s == batch_size:
#            ret[6, :] = np.random.choice(energy_specter[0],
#                                         batch_size,
#                                         p=energy_specter[1])
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


