# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:30:02 2015

@author: erlean
"""
import numpy as np
from opendxmc.engine import score_energy
from opendxmc.tube.tungsten import specter as tungsten_specter
from opendxmc.runner.phase_space import ct_phase_space
import time
from opendxmc.utils import human_time


from matplotlib import pylab as plt
import pdb
import logging
logger = logging.getLogger('OpenDXMC')


def log_elapsed_time(time_start, elapsed_exposures, total_exposures, n_histories=None):
    time_delta = time.clock() - time_start
    p = np.round(float(elapsed_exposures) / float(total_exposures) * 100., 1)
    eta = time_delta * (total_exposures / float(elapsed_exposures) - 1.)
    if elapsed_exposures == total_exposures:
        if n_histories is not None:
            logger.info('{0}: [{1}%] Finished {2} histories in {3}'.format(time.ctime(), p, n_histories*total_exposures, human_time(time_delta)))
        else:
            logger.info('{0}: [{1}%] Finished {2} exposures in {3}'.format(time.ctime(), p, total_exposures, human_time(time_delta)))
    else:
        logger.info('{0}: [{1}%] estimated ETA is {2}, finished exposure {3} of {4}'.format(time.ctime(), p, human_time(eta), elapsed_exposures, total_exposures))


def recarray_to_dict(arr, key='key', value='value', value_is_string=False):
    assert key in arr.dtype.names
    assert value in arr.dtype.names
    if value_is_string:
        return {arr[key][i]: str(arr[value][i], encoding='utf-8') for i in range(arr.shape[0])}
    return {arr[key][i]: arr[value][i] for i in range(arr.shape[0])}


def generate_attinuation_lut(materials, material_map, min_eV=None,
                             max_eV=None, ignore_air=False):

    if isinstance(material_map, np.recarray):
        material_map = recarray_to_dict(material_map)
    if min_eV is None:
        min_eV = 0.
    if max_eV is None:
        max_eV = 500.e3

    names = [m.name for m in materials]
    atts = {}

    for key, value in list(material_map.items()):
        key = int(key)
        try:
            ind = names.index(value)
        except ValueError:
            raise ValueError('No material named '
                             '{0} in first argument. '
                             'The material_map requires {0}'.format(value))
        atts[key] = materials[ind].attinuation
        if value == 'air':
            air_key = key

    energies = np.unique(np.hstack([a['energy'] for a in list(atts.values())]))
    e_ind = (energies <= max_eV) * (energies >= min_eV)
    if not any(e_ind):
        raise ValueError('Supplied minimum or maximum energies '
                         'are out of range')
    energies = energies[e_ind]
    lut = np.empty((len(atts), 5, len(energies)), dtype=np.double)
    for i, a in list(atts.items()):
        lut[i, 0, :] = energies
        if ignore_air and air_key == i:
            lut[i, 1:, :] = 0
        else:
            for j, key in enumerate(['total', 'rayleigh', 'photoelectric',
                                     'compton']):
                lut[i, j+1, :] = np.interp(energies, a['energy'], a[key])
    return lut


def prepare_geometry_from_ct_array(ctarray, specter, materials):
        """genereate material and density arrays and material map from
           a list of materials to use
           INPUT:
               specter for this study
               list of materials
           OUTPUT :
               material_map, material_array, density_array
        """
        if ctarray is None:
            return
        specter = (specter[0], specter[1]/specter[1].sum())

        water_key = None
        material_map = {}
        material_att = {}
        material_dens = {}
        materials.sort(key=lambda x: x.density)
        for i, mat in enumerate(materials):
            material_map[i] = mat.name
            material_dens[i] = float(mat.density)
            # interpolationg and integrating attinuation coefficient
            material_att[i] = np.trapz(np.interp(specter[0],
                                                 mat.attinuation['energy'],
                                                 mat.attinuation['total']),
                                       specter[0])
            material_att[i] *= mat.density
            if mat.name == 'water':
                water_key = i
        assert water_key is not None  # we need to include water in materials

        # getting a list of attinuation
        material_HU_list = [(key, (att / material_att[water_key] - 1.)*1000.)
                            for key, att in material_att.items()]
        material_HU_list.sort(key=lambda x: x[1])
        HU_bins = (np.array(material_HU_list)[:-1,1] + np.array(material_HU_list)[1:, 1]) / 2.

        material_array = np.digitize(ctarray.ravel(), HU_bins).reshape(ctarray.shape).astype(np.int)
#        plt.imshow(material_array[:,:,20])
#        plt.show(block=True)
#        pdb.set_trace()
        density_array = np.asarray(material_array, dtype=np.float)
        np.choose(material_array,
                  [material_dens[i] for i in range(len(material_dens))],
                  out=density_array)
        return material_map, material_array, density_array
        #cleaning up


#
#
#        material_array = np.zeros_like(ctarray, dtype=np.int)
#        density_array = np.zeros_like(ctarray, dtype=np.float)
#        llim = -5000
#
#        for i in range(len(material_HU_list)):
#            if i == len(material_HU_list) - 1:
#                ulim = 50000
#            else:
#                ulim = int(0.5 * (material_HU_list[i][1] + material_HU_list[i + 1][1]))
#
#            for snitt in range(ctarray.shape[2]):
#                for row in range(ctarray.shape[1]):
#                    ind = np.nonzero((ctarray[:, row, snitt] > llim) * (ctarray[:, row, snitt] <= ulim))
#                    material_array[ind, row, snitt] = material_HU_list[i][0]
#                    density_array[ind, row, snitt] = material_dens[material_HU_list[i][0]]
#            llim = ulim
#
#        return material_map, material_array, density_array

def ct_runner_validate_simulation(simulation, materials, ignore_air=False):
    """
    validating and performs a ct mc simulation
    """

    if simulation.ctarray is None:
        logger.warning('CT study {} has no CT images. Simulation not started'.format(simulation.name))
        raise ValueError('CT study {} must have CT images to run a simulation'.format(simulation.name))

    # testing for required attributes
    for att in ['material_map', 'material_array', 'density_array']:
        if getattr(simulation, att) is None:
            logger.info('CT study {0} do not have a {1}. Recalculating from CT array.'.format(simulation.name, att))
            specter = tungsten_specter(simulation.kV,
                                       filtration_materials='al',
                                       filtration_mm=simulation.al_filtration)
            vals = prepare_geometry_from_ct_array(simulation.ctarray,
                                                  specter,
                                                  materials)
            simulation.material_map = vals[0]
            simulation.material = vals[1]
            simulation.density = vals[2]
            if simulation.energy_imparted is not None:
                logger.warning('Erasing dosematrix for simulation {} due to recalculating of tissue materials'.format(simulation.name))
                simulation.energy_imparted = None
            break
    else:
        # Testing for required materials if the simulation have a material_map
        material_names = [m.name for m in materials]
        for m_name in simulation.material_map['values']:
            if m_name not in material_names:
                raise ValueError('Provided materials are not in ct study')

    # test for correct material geometry and mapping
    materials_ind = list(np.unique(simulation.material_array))
    materials_key = list(recarray_to_dict(simulation.material_map, value_is_string=True).keys())
    for ind, key in zip(materials_ind, materials_key):
        try:
            assert key == ind
        except AssertionError:
            pdb.set_trace()


def ct_runner(simulation, materials, ignore_air=True):
    ct_runner_validate_simulation(simulation, materials, ignore_air)

    phase_space = ct_phase_space(simulation)
    material_array = simulation.material
    density_array = simulation.density
    spacing = simulation.spacing
    material_map = recarray_to_dict(simulation.material_map, value_is_string=True)
    n_histories = simulation.histories
#    del simulation

#    plt.imshow(material_array[:,:,20])
#    plt.show(block=True)

    N = np.array(material_array.shape, dtype=np.double)

    offset = - N * spacing / 2.
    offset[-1] = simulation.start
    logger.warning('Offset not properly corrected for, currently using center of reconstruction FOV as isocenter.')

    lut = generate_attinuation_lut(materials, material_map,
                                   max_eV=500.e3, ignore_air=ignore_air)

    energy_imparted = np.zeros_like(density_array, dtype=np.double)
    tot_histories = simulation.histories * simulation.exposures
    if tot_histories > 1e6:
        coffe_msg = ', go get coffe!'
    else:
        coffe_msg = ''
    logger.info('{0}: Starting simulation with {1} histories per rotation{2}'.format(time.ctime(), tot_histories, coffe_msg))
    time_start = time.clock()
    for p, e, n in phase_space:
#        pdb.set_trace()
        score_energy(p, N, spacing, offset, material_array, density_array, lut,
                     energy_imparted)
        log_elapsed_time(time_start, e+1, n, n_histories=n_histories)
    simulation.energy_imparted = energy_imparted
    dose = energy_imparted / (density_array * np.prod(spacing)) * 1.60217657e-19
    for i in range(9):
        plt.subplot(3,3,i+1)
        k = int(i * (dose.shape[2] - 1) / 9.)
        plt.imshow(np.squeeze(dose[:,:,k]))
    plt.show(block=True)


