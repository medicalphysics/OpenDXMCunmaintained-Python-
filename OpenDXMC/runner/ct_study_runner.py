# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:30:02 2015

@author: erlean
"""
import numpy as np
from scipy.ndimage.interpolation import affine_transform, spline_filter
from scipy.ndimage.filters import gaussian_filter
from opendxmc.engine import score_energy
from opendxmc.tube.tungsten import specter as tungsten_specter
from opendxmc.runner.phase_space import ct_phase_space
from opendxmc.runner.phase_space import ct_seq
from opendxmc.utils import circle_mask
import time
from opendxmc.utils import human_time

import logging
logger = logging.getLogger('OpenDXMC')


def log_elapsed_time(time_start, elapsed_exposures, total_exposures,
                     n_histories=None):
    time_delta = time.clock() - time_start
    p = np.round(float(elapsed_exposures) / float(total_exposures) * 100., 1)
    eta = time_delta * (total_exposures / float(elapsed_exposures) - 1.)
    if elapsed_exposures == total_exposures:
        if n_histories is not None:
            logger.info('{0}: [{1}%] Finished {2} histories in {3}'.format(
                time.ctime(), p, n_histories*total_exposures,
                human_time(time_delta)))
        else:
            logger.info('{0}: [{1}%] Finished {2} exposures in {3}'.format(
                time.ctime(), p, total_exposures, human_time(time_delta)))
    else:
        logger.info('{0}: [{1}%] estimated ETA is {2}, finished exposure {3}'
                    ' of {4}'.format(time.ctime(), p, human_time(eta),
                                     elapsed_exposures, total_exposures))


def recarray_to_dict(arr, key='key', value='value', value_is_string=False):
    assert key in arr.dtype.names
    assert value in arr.dtype.names
    if value_is_string:
        return {arr[key][i]: str(arr[value][i], encoding='utf-8')
                for i in range(arr.shape[0])}
    return {arr[key][i]: arr[value][i] for i in range(arr.shape[0])}


def generate_attinuation_lut(materials, material_map, min_eV=None,
                             max_eV=None, ignore_air=False):

    if isinstance(material_map, np.recarray):
        material_map = recarray_to_dict(material_map, value_is_string=True)
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


def prepare_geometry_from_ct_array(ctarray, scale ,specter, materials):
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
        ctarray = gaussian_filter(ctarray, scale)
        ctarray = affine_transform(spline_filter(ctarray, 
                                                 order=3, 
                                                 output=np.int16), 
                                   scale, 
                                   output_shape=np.floor(np.array(ctarray.shape)/scale), 
                                   cval=-1000, output=np.int16, prefilter=False)        
        
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
        HU_bins = (np.array(material_HU_list)[:-1, 1] +
                   np.array(material_HU_list)[1:, 1]) / 2.

        material_array = np.digitize(
            ctarray.ravel(), HU_bins).reshape(ctarray.shape).astype(np.int)

        density_array = np.asarray(material_array, dtype=np.float)
        np.choose(material_array,
                  [material_dens[i] for i in range(len(material_dens))],
                  out=density_array)
        return material_map, material_array, density_array


def ct_runner_validate_simulation(simulation, materials, second_try=False):
    """
    validating a ct mc simulation
    """

    # testing for required attributes
    for att in ['material_map', 'material', 'density']:
        if getattr(simulation, att) is None:
            if simulation.ctarray is None:
                logger.warning('CT study {} has no CT images. Simulation not '
                               'started'.format(simulation.name))
                raise ValueError('CT study {} must have CT images to run a '
                                 'simulation'.format(simulation.name))

            logger.info('CT study {0} do not have a {1}. Recalculating from CT'
                        ' array.'.format(simulation.name, att))
            specter = tungsten_specter(simulation.kV,
                                       filtration_materials='al',
                                       filtration_mm=simulation.al_filtration)
            vals = prepare_geometry_from_ct_array(simulation.ctarray,
                                                  simulation.scaling,
                                                  specter,
                                                  materials)
            simulation.material_map = vals[0]
            simulation.material = vals[1]
            simulation.density = vals[2]
            if simulation.energy_imparted is not None:
                logger.warning('Erasing dosematrix for simulation {} due to '
                               'recalculating of tissue materials'
                               ''.format(simulation.name))
                simulation.energy_imparted = None
            break
    else:
        # Testing for required materials if the simulation have a material_map
        material_names = [m.name for m in materials]
        for m_name in simulation.material_map['value']:
            if str(m_name, encoding='utf-8') not in material_names:
                raise ValueError('Provided materials are not in ct study')

    # test for correct material geometry and mapping
    materials_ind = list(np.unique(simulation.material))
    materials_key = list(recarray_to_dict(simulation.material_map,
                                          value_is_string=True).keys())
    materials_range = list(range(len(materials_ind)))

    try:
        for ind, key in zip(materials_ind, materials_key):
            assert key == ind
            assert key in materials_range

    except AssertionError:
        if not second_try:
            logger.warning('Something went wrong with material definitions, '
                           'attempting to recreate material mapping for '
                           'current simulation.')
            simulation.material_map = None
            ct_runner_validate_simulation(simulation, materials,
                                          second_try=True)
        else:
            raise ValueError('Error in material definitions for simulation')


def ct_runner(simulation, materials, energy_imparted_to_dose_conversion=True, callback=None):
    """Runs a MC simulation on a simulation object, and updates the
    energy_imparted property.

    INPUT:

        simulation : Simulation instance

        materials : a list of Material instances to be used in the simulation

        ignore_air : [optional] If set, ignores the material 'air' in
            MC calculation

    OUTPUT:
        None, but updates the energy_imparted property of simulation
    """
    logger.info('Preparing simulation for {}'.format(simulation.name))
    materials_organic = [m for m in materials if m.organic]

    # Validating if everything is in place
    ct_runner_validate_simulation(simulation, materials_organic)

    phase_space = ct_phase_space(simulation)
    n_histories = simulation.histories
#    del simulation

#    plt.imshow(material_array[:,:,20])
#    plt.show(block=True)

    N = np.array(simulation.material.shape, dtype=np.double)

    offset = np.zeros(3, dtype=np.double)
    spacing = simulation.spacing * simulation.scaling
   
    lut = generate_attinuation_lut(materials_organic, simulation.material_map,
                                   max_eV=500.e3,
                                   ignore_air=simulation.ignore_air)

    energy_imparted = np.zeros_like(simulation.density, dtype=np.double)
    tot_histories = simulation.histories * simulation.exposures

    if tot_histories > 1e6:
        coffe_msg = ', go get coffe!'
    else:
        coffe_msg = '.'
    logger.info('{0}: Starting simulation with {1} histories per '
                'rotation{2}'.format(time.ctime(), tot_histories, coffe_msg))

    time_start = time.clock()
    for p, e, n in phase_space:
#        ###test
#        import pylab as plt
#        from mpl_toolkits.mplot3d import Axes3D
#        shape = simulation.ctarray.shape
#        box = np.zeros((10, 3))
#        box[0, :] = np.zeros(3)
#        box[1, :] = np.array([shape[0], 0, 0]) * simulation.spacing
#        box[2, :] = np.array([shape[0], shape[1], 0]) * simulation.spacing
#        box[3, :] = np.array([0, shape[1], 0]) * simulation.spacing
#        box[4, :] = np.array([0, 0, 0]) * simulation.spacing
#        box[5, :] = np.array([0, 0, shape[2]]) * simulation.spacing
#        box[6, :] = np.array([shape[0], 0, shape[2]]) * simulation.spacing
#        box[7, :] = np.array([shape[0], shape[1], shape[2]]) * simulation.spacing
#        box[8, :] = np.array([0, shape[1], shape[2]]) * simulation.spacing
#        box[9, :] = np.array([0, 0, shape[2]]) * simulation.spacing
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#                
#        ax.plot(box[:, 0], box[:, 1],box[:, 2], label='box')
#        ax.plot(box[:, 0], box[:, 1],box[:, 2], 'o')
#       
#        l = []        
#    
#        ax.plot(p[0, :].ravel(), p[1, :].ravel(),p[2, :].ravel(), 'o', label='spiral')
#        ax.plot(p[0, :] + p[3,:]*30, p[1, :]+p[4,:]*30,p[2, :]+p[5,:]*30, 'o',label='spiral')
##        for k in range(p.shape[1]):
##            v = np.empty((3, 2))
##            v[:, 0] = p[:3, k] 
##            v[:, 1] = p[3:6, k]*30 + v[:, 0]
##            l.append(ax.plot(v[0, :], v[1, :], v[2,:]))
##            
#        ax.legend()
#  
#        plt.show(block=True)
        ######TEST
        
#        import pdb; pdb.set_trace()
        score_energy(p, N, spacing, offset, simulation.material,
                     simulation.density, lut, energy_imparted)
        log_elapsed_time(time_start, e+1, n, n_histories=n_histories)
        if callback is not None:
            callback(simulation.name, energy_imparted, e + 1)
        simulation.start_at_exposure_no = e + 1

    generate_dose_conversion_factor(simulation, materials)
    simulation.energy_imparted = energy_imparted
    simulation.start_at_exposure_no = 0


def generate_dose_conversion_factor(simulation, materials):
    air, pmma = None, None
    for m in materials:
        if m.name == 'pmma':
            pmma = m
        elif m.name == 'air':
            air = m
    
    if (simulation.ctdi_air100 > 0.) and (air is not None):
        obtain_ctdiair_conversion_factor(simulation, air)
    elif (simulation.ctdi_w100 > 0.) and (pmma is not None) and (air is not None):
        obtain_ctdiw_conversion_factor(simulation, pmma, air)
    else:
        msg = """Need a combination of air material and ctdi air or ctdi_w100
                 pmma material and ctdiw_100 to generate energy to dose
                 conversion factor."""
        logger.warning(msg)


def obtain_ctdiair_conversion_factor(simulation, air_material):

    logger.info('Starting simulating CTDIair100 measurement for '
                '{0}. CTDIair100 is {1}mGy'.format(simulation.name, simulation.ctdi_air100))
    spacing = np.array((1, 1, 10), dtype=np.double)

    N = np.rint(np.array((simulation.sdd / spacing[0],
                          simulation.sdd / spacing[1], 1),
                         dtype=np.double))

    offset = -N * spacing / 2.
    material_array = np.zeros(N, dtype=np.intc)
    material_map = {0: air_material.name}
    density_array = np.zeros(N, dtype=np.double) + air_material.density
    lut = generate_attinuation_lut([air_material], material_map, max_eV=0.5e6)
    dose = np.zeros_like(density_array, dtype=np.double)

    en_specter = tungsten_specter(simulation.kV, angle_deg=10.,
                                  filtration_materials='Al',
                                  filtration_mm=simulation.al_filtration)

    phase_space = ct_seq(simulation.scan_fov, simulation.sdd,
                         simulation.total_collimation, start=0, stop=0, step=0,
                         exposures=simulation.exposures,
                         histories=simulation.histories,
                         energy_specter=en_specter, 
                         batch_size=simulation.batch_size)

    t0 = time.clock()
    for batch, i, n in phase_space:
        score_energy(batch, N, spacing, offset, material_array,
                     density_array, lut, dose)
        log_elapsed_time(t0, i+1, n)

    center = np.floor(N / 2).astype(np.int)
    d = dose[center[0], center[1], center[2]]
    simulation.conversion_factor_ctdiair = simulation.ctdi_air100 / d


def generate_ctdi_phantom(simulation, pmma, air, size=32.):
    spacing = np.array((1, 1, 2.5), dtype=np.double)
    N = np.rint(np.array((simulation.sdd / spacing[0],
                          simulation.sdd / spacing[1], 6),
                         dtype=np.double))

    offset = -N * spacing / 2.
    material_array = np.zeros(N, dtype=np.intc)
    radii_phantom = size * spacing[0]
    radii_meas = 2. * spacing[0]
    center = (N * spacing / 2.)[:2]
    radii_pos = (size - 2.) * spacing[0]
    pos = [(center[0], center[1])]
    for ang in [0, 90, 180, 270]:
        dx = radii_pos * np.sin(np.deg2rad(ang))
        dy = radii_pos * np.cos(np.deg2rad(ang))
        pos.append((center[0] + dx, center[1] + dy))

    for i in range(int(N[2])):
        material_array[:, :, i] += circle_mask((N[0], N[1]),
                                               radii_phantom)
        for p in pos:
            material_array[:, :, i] += circle_mask((N[0], N[1]),
                                                   radii_meas,
                                                   center=p)

    material_map = {0: air.name, 1: pmma.name, 2: air.name}
    density_array = np.zeros_like(material_array, dtype=np.double)
    density_array[material_array == 0] = air.density
    density_array[material_array == 1] = pmma.density
    density_array[material_array == 2] = air.density

#        density_array = np.zeros(N, dtype=np.double) + material.density
    lut = generate_attinuation_lut([air, pmma], material_map, max_eV=0.5e6)
    return N, spacing, offset, material_array, density_array, lut, pos


def obtain_ctdiw_conversion_factor(simulation, pmma, air,
                                   callback=None, phantom_size=32.):

    logger.info('Starting simulating CTDIw100 measurement for '
                '{}'.format(simulation.name))
    args = generate_ctdi_phantom(simulation, pmma, air, size=phantom_size)
    N, spacing, offset, material_array, density_array, lut, meas_pos = args

    dose = np.zeros_like(density_array)

    en_specter = tungsten_specter(simulation.kV, angle_deg=10.,
                                  filtration_materials='Al',
                                  filtration_mm=simulation.al_filtration)

    phase_space = ct_seq(simulation.scan_fov, simulation.sdd,
                         simulation.total_collimation, start=0, stop=0, step=0,
                         exposures=simulation.exposures,
                         histories=simulation.histories,
                         energy_specter=en_specter, 
                         batch_size=simulation.batch_size)

    t0 = time.clock()
    for batch, i, n in phase_space:
        score_energy(batch, N, spacing, offset, material_array,
                     density_array, lut, dose)
        log_elapsed_time(t0, i+1, n)

    d = []
    for p in meas_pos:
        x, y = int(p[0]), int(p[1])
        d.append(dose[x, y, 1:-1].sum())

    ctdiv = d.pop(0) / 3.
    ctdiv += 2. * sum(d) / 3. / 4.
    simulation.conversion_factor_ctdiw = simulation.ctdi_w100 / ctdiv
