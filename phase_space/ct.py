# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:47:01 2015

@author: erlean
"""

import numpy as np
import scipy.interpolate


def spiral(scan_fov, sdd, total_collimation, pitch=1,
           start=0, stop=1, exposures=100, histories=1,
           energy=70000., energy_specter=None,
           batch_size=None, modulation_xy=None,
           modulation_z=None):
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
            number of histories per batch, must be greater than
            histories
        modulation_xy : [(N,), (N,)]
            tube current XY modulation, list/tuple of
            (ndarray(position), ndarray(scale_factors))
        modulation_z : [(N,), (N,)]
            tube current modulation z axis, list/tuple of
            (ndarray(position), ndarray(scale_factors))

    OUTPUT:
        Iterator returning ndarrays of shape (8, batch_size),
        one row is equal to photon (start_x, start_y, star_z, direction_x,
        direction_y, direction_z, energy, weight)
    """

    # total number of exposures + one total rotation
    exposures = int(exposures)
    e = int((abs(start - stop) / total_collimation + 1) * exposures)
    if start < stop:
        d_col = total_collimation / 2.
    else:
        d_col = -total_collimation / 2.
    # positions along z for each exposure
    t = np.linspace(start-d_col, stop + d_col, e)
    # we randomize the positions not make the progress bar jump
    t = np.random.permutation(t)
    # angle for each z position , i.e the x, y coordinates
    ang = t / (pitch * total_collimation) * np.pi * 2.

    # rotation matrix along z-axis for an angle x
    rot = lambda x: np.array([[np.cos(x), -np.sin(x), 0],
                              [np.sin(x), np.cos(x), 0],
                              [0, 0, 1]], dtype=np.double)

    if batch_size is None:
        batch_size = histories
    if batch_size < histories:
        batch_size = histories
    batch_size -= (batch_size % histories)
    assert batch_size % histories == 0

    if energy_specter is None:
        energy_specter = [np.array([energy], dtype=np.double),
                          np.array([1.0], dtype=np.double)]
    energy_specter = (energy_specter[0],
                      energy_specter[1] / energy_specter[1].sum())

    if modulation_xy is None:
        mod_xy = lambda x: 1.0
    else:
        mod_xy = scipy.interpolate.interp1d(modulation_xy[0], modulation_xy[1],
                                            copy=False, bounds_error=False,
                                            fill_value=1.0)

    if modulation_z is None:
        mod_z = lambda x: 1.0
    else:
        mod_z = scipy.interpolate.interp1d(modulation_z[0], modulation_z[1],
                                           copy=False, bounds_error=False,
                                           fill_value=1.0)

    teller = 0
    ret = np.zeros((8, batch_size), dtype=np.double)

    for i in xrange(e):
        R = rot(ang[i])
#        pdb.set_trace()
        ind_b = teller * histories
        ind_s = (teller + 1) * histories

        ret[1, ind_b:ind_s] = 0
        ret[0, ind_b:ind_s] = -sdd/2.
        ret[2, ind_b:ind_s] = t[i]
        ret[0:3, ind_b:ind_s] = np.dot(R, ret[0:3, ind_b:ind_s])

        ret[3, ind_b:ind_s] = sdd / 2.
        ret[4, ind_b:ind_s] = scan_fov * np.random.uniform(-1., 1., histories)
        ret[5, ind_b:ind_s] = t[i] + d_col * np.random.uniform(-1., 1.,
                                                               histories)
        ret[3:6, ind_b:ind_s] = np.dot(R, ret[3:6, ind_b:ind_s])
        lenght = np.sqrt(np.sum(ret[3:6, ind_b:ind_s]**2, axis=0))
        ret[3:6, ind_b:ind_s] /= lenght

        ret[7, ind_b:ind_s] = mod_z(t[i]) * mod_xy(t[i])

        if ind_s == batch_size:
            ret[6, :] = np.random.choice(energy_specter[0],
                                         batch_size,
                                         p=energy_specter[1])
            yield ret, i*histories, e*histories
            teller = 0
        else:
            teller += 1
    teller -= 1
    if teller > 0:
        ret[6, :] = np.random.choice(energy_specter[0],
                                     batch_size,
                                     p=energy_specter[1])
        yield ret[:, :teller * histories], i*histories, e*histories

