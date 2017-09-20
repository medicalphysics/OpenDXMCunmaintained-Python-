# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:26:24 2015

@author: ander
"""

import numpy as np
import scipy.interpolate
import itertools
from opendxmc.tube.tungsten import specter, attinuation
import logging
logger = logging.getLogger('OpenDXMC')


def half_shuffle(arr):
    """
    Shuffles an array in a predictable manner
    """
    assert len(arr.shape) == 1
    
    h = arr.shape[0] // 2
    shuf = np.empty_like(arr)
    shuf[::2] = arr[h:]
    shuf[1::2] = arr[:h][::-1]
    return shuf


def bowtie_path_lenght(angles, radius, distance):
    angle_max = np.arcsin(radius / (radius+distance))
    angle_max_ind = np.abs(angles ) > angle_max

    theta = np.arctan(distance*np.tan(angles)/radius)
    c = radius*(1-np.cos(theta))
    c[angle_max_ind] = radius
    return c / np.cos(angles)


def ct_source_space(simulation, exposure_modulation=None):
    if simulation.get('use_tube_B'):
        wA = simulation.get('tube_weight_A')
        wB = simulation.get('tube_weight_A')
        wS = wA + wB
        
        return itertools.chain(
            ct_source_space_single(simulation, exposure_modulation, tube='A', weight=2*wA/wS),
            ct_source_space_single(simulation, exposure_modulation, tube='B', weight=2*wB/wS),
        )
    else :
        return ct_source_space_single(simulation, exposure_modulation, tube='A')
    

def ct_source_space_single(simulation, exposure_modulation=None, tube='A', weight=1.0):

    
    arglist = ['scan_fov', 'sdd']
    kwarglist = ['start', 'stop', 'exposures', 'histories',
                 'start_at_exposure_no',
                 'bowtie_distance', 'bowtie_radius']

    args = [simulation.get(a) for a in arglist]
    args.append(simulation.get('detector_rows') * simulation.get('detector_width'))
    kwargs = {'exposure_modulation': exposure_modulation}
    for a in kwarglist:
        kwargs[a] = simulation.get(a)
    kwargs['rotation_center'] = simulation.get('data_center')
    kwargs['rotation_plane_cosines'] = simulation.get('image_orientation')
    kwargs['tube_start_angle'] = simulation.get('tube_start_angle_'+tube)
    kwargs['weight'] = weight

    if simulation.get('is_spiral'):
        kwargs['pitch'] = simulation.get('pitch')
        phase_func = ct_spiral
    else:
        kwargs['step'] = simulation.get('step')
        phase_func = ct_seq

    s = specter(simulation.get('kV_'+tube), angle_deg=simulation['anode_angle'], filtration_materials='Al',
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
              tube_start_angle=0.,
              energy=70000., energy_specter=None,
              rotation_center=None,
              rotation_plane_cosines=None,
              exposure_modulation=None, start_at_exposure_no=0,
              bowtie_radius=1, bowtie_distance=0,
              weight=1.0):
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
        modulation_xy : [(N,), (N,)] (NOT IMPLEMENTED)
            tube current XY modulation, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        exposure_modulation : [(N,), (N,)]
            tube current modulation z axis, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        start_at_exposure_no: int
            Starting at this exposure number, used for resuming a simulation
        weight: float
            weight of phasespace 
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
#    print('CT phase space start', start, stop)
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

    # angle for each z position , i.e the x, y coordinates
    ang = np.mod(t / (pitch * total_collimation) * np.pi * 2. + np.deg2rad(tube_start_angle), 2*np.pi)
    logger.info('Tube start angle: {}, first angle {}, second angle {}'.format(tube_start_angle, np.rad2deg(ang[0]), np.rad2deg(ang[1])))
    # rotation matrix along z-axis for an angle x

    if energy_specter is None:
        energy_specter = [np.array([energy], dtype=np.double),
                          np.array([1.0], dtype=np.double)]
    energy_specter = (energy_specter[0],
                      energy_specter[1] / energy_specter[1].sum())

    specter_cpd = np.cumsum(energy_specter[1]).astype('float64')
    specter_cpd /= specter_cpd.max()

    specter_energy = energy_specter[0].astype('float64')

#    if modulation_xy is None:
#        mod_xy = lambda x: 1.0
#    else:
#        mod_xy = scipy.interpolate.interp1d(modulation_xy[0], modulation_xy[1],
#                                            copy=False, bounds_error=False,
#                                            fill_value=1.0)
    
    if exposure_modulation is None:
        mod_z = lambda x: weight
    else:
        if np.abs(np.mean(exposure_modulation[:, 1])) > 0.000001:
            modulator_array = exposure_modulation[:, 1]
            modulator_array /= np.mean(modulator_array)
            modulator_array *= weight
            mod_z = scipy.interpolate.interp1d(exposure_modulation[:, 0],
                                               modulator_array,
                                               copy=True, bounds_error=False,
                                               fill_value=weight, kind='nearest')
        else:
            mod_z = lambda x: weight
    fov_arr=np.array([scan_fov], dtype='float64')
    collimation_arr=np.array([total_collimation], dtype='float64')
    rot_fan_angle = np.array([np.arctan(fov_arr[0]/sdd) * 2],dtype='float64')
    scan_fan_angle = np.array([np.arctan(collimation_arr[0] *.5 / sdd) * 2], dtype='float64')

    bowtie_angle = np.linspace(-rot_fan_angle[0]/2, rot_fan_angle[0]/2, 101, dtype='float64')
    bowtie_lenghts= bowtie_path_lenght(bowtie_angle, bowtie_radius, bowtie_distance)
    bowtie_weights = np.empty_like(bowtie_angle, dtype='float64')
    bowtie_att = attinuation(specter_energy/1000, name='aluminum', density=True).astype('float64')
    for i in range(bowtie_lenghts.shape[0]):
        bowtie_weights[i] = np.sum(energy_specter[1]*np.exp(-bowtie_att*bowtie_lenghts[i]))


    n_bowtie = np.array(bowtie_weights.shape, dtype='int')
    n_specter = np.array(specter_energy.shape, dtype='int')


    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center[[1, 0, 2]])
    for i in range(start_at_exposure_no, e):
        R = np.dot(M, rotation_z_matrix(ang[i]))

        position = np.dot(R, np.array([-sdd/2., 0, t[i]], dtype='float64')) + rotation_center_image
        direction = np.dot(R, np.array([1., 0, 0], dtype='float64'))
        scan_axis = np.dot(R, np.array([0, 0, 1], dtype='float64'))
        ret = (position, direction, scan_axis,
               scan_fan_angle,
               rot_fan_angle,
               np.array([mod_z(t[i])], dtype='float64'),
               specter_cpd, specter_energy, n_specter,
               bowtie_weights, bowtie_angle, n_bowtie)
#        ret = (position, direction, scan_axis,
#               np.array([sdd], dtype='float64'),
#               np.array([scan_fov], dtype='float64'),
#               np.array([total_collimation], dtype='float64'),
#               np.array([mod_z(t[i])], dtype='float64'),
#               specter_cpd.astype('float64'), specter_energy.astype('float64'))
#        print('Weight: {}'.format(ret[5]), exposure_modulation[0, 0], t[i] ,exposure_modulation[-1, 0] )
        yield ret, i, e




def ct_seq(scan_fov, sdd, total_collimation, step=1,
              start=0, stop=1, exposures=100, histories=1,
              tube_start_angle=0.,
              energy=70000., energy_specter=None,
              rotation_center=None,
              rotation_plane_cosines = None,
              bowtie_radius=1, bowtie_distance=0,
              exposure_modulation=None, start_at_exposure_no=0,
              weight=1.0):
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
        ang[exposures*i: exposures*(i+1)] = np.linspace(0, 2*np.pi, exposures) + np.deg2rad(tube_start_angle)

    t = half_shuffle(t)
    ang = half_shuffle(ang)

    if energy_specter is None:
        energy_specter = [np.array([energy], dtype=np.double),
                          np.array([1.0], dtype=np.double)]
    energy_specter = (energy_specter[0],
                      energy_specter[1] / energy_specter[1].sum())
    specter_cpd = np.cumsum(energy_specter[1]).astype('float64')
    specter_cpd /= specter_cpd.max()
    specter_energy = energy_specter[0].astype('float64')

#    if modulation_xy is None:
#        mod_xy = lambda x: 1.0
#    else:
#        mod_xy = scipy.interpolate.interp1d(modulation_xy[0], modulation_xy[1],
#                                            copy=False, bounds_error=False,
#                                            fill_value=1.0)

    fov_arr=np.array([scan_fov], dtype='float64')
    collimation_arr=np.array([total_collimation], dtype='float64')
    rot_fan_angle = np.array([np.arctan(fov_arr[0]/sdd) * 2],dtype='float64')
    scan_fan_angle = np.array([np.arctan(collimation_arr[0] *.5 / sdd) * 2], dtype='float64')

    bowtie_angle = np.linspace(-rot_fan_angle[0]/2, rot_fan_angle[0]/2, 101, dtype='float64')
    bowtie_lenghts= bowtie_path_lenght(bowtie_angle, bowtie_radius, bowtie_distance)
    bowtie_weights = np.empty_like(bowtie_angle, dtype='float64')
    bowtie_att = attinuation(specter_energy/1000, name='aluminum', density=True).astype('float64')
    for i in range(bowtie_lenghts.shape[0]):
        bowtie_weights[i] = np.sum(energy_specter[1]*np.exp(-bowtie_att*bowtie_lenghts[i]))

    n_bowtie = np.array(bowtie_weights.shape, dtype='int')
    n_specter = np.array(specter_energy.shape, dtype='int')


    if exposure_modulation is None:
        mod_z = lambda x: weight
    else:
        if np.abs(np.mean(exposure_modulation[:, 1])) > 0.000001:
            modulator_array = exposure_modulation[:, 1]
            modulator_array /= np.mean(modulator_array)
            modulator_array *= weight
            mod_z = scipy.interpolate.interp1d(exposure_modulation[:, 0],
                                               modulator_array,
                                               copy=True, bounds_error=False,
                                               fill_value=weight, kind='nearest')
        else:
            mod_z = lambda x: weight

    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center[[1, 0, 2]])
    for i in range(start_at_exposure_no, e):
        R = np.dot(M, rotation_z_matrix(ang[i]))

        position = np.dot(R, np.array([-sdd/2., 0, t[i]], dtype='float64')) + rotation_center_image
        direction = np.dot(R, np.array([1., 0, 0], dtype='float64'))
        scan_axis = np.dot(R, np.array([0, 0, 1], dtype='float64'))

        ret = (position, direction, scan_axis,
               scan_fan_angle,
               rot_fan_angle,
               np.array([mod_z(t[i])], dtype='float64'),
               specter_cpd, specter_energy, n_specter,
               bowtie_weights, bowtie_angle, n_bowtie)

#        ret = (position, direction, scan_axis,
#               np.array([sdd], dtype='float64'),
#               np.array([scan_fov], dtype='float64'),
#               np.array([total_collimation], dtype='float64'),
#               np.array([mod_z(t[i])], dtype='float64'),
#               specter_cpd.astype('float64'), specter_energy.astype('float64'))
        yield ret, i, e

