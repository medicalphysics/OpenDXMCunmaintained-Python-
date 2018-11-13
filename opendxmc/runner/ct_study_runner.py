# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:30:02 2015

@author: erlean
"""
import numpy as np
from scipy.ndimage.interpolation import affine_transform, spline_filter
from scipy.ndimage.filters import gaussian_filter
import itertools
from opendxmc.engine import Engine

from opendxmc.tube.tungsten import specter as tungsten_specter
from opendxmc.runner.ct_sources import ct_source_space
from opendxmc.runner.ct_sources import ct_seq
from opendxmc.utils import circle_mask
import time
from opendxmc.utils import human_time, rebin

import logging
logger = logging.getLogger('OpenDXMC')


def log_elapsed_time(time_start, elapsed_exposures, total_exposures,
                     start_exposure, n_histories=None):
    time_delta = time.clock() - time_start
    p = np.round(float(elapsed_exposures) / float(total_exposures) * 100., 1)
    eta = time_delta * (float(total_exposures-start_exposure) / float(elapsed_exposures-start_exposure) - 1.)
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
    return '{0}% ETA: '.format(p) + human_time(eta)


def recarray_to_dict(arr, key=None, value=None, value_is_string=False):
    key, value = arr.dtype.names
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
    elif isinstance(material_map, np.ndarray):
        material_map = recarray_to_dict(material_map, value_is_string=True)
    assert isinstance(material_map, dict)
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
                             'The material_map requires {0} material'.format(value))
        atts[key] = materials[ind].attinuation
        if value == 'air':
            air_key = key

    energies = np.unique(np.hstack([a['energy'] for a in list(atts.values())]))
    e_ind = (energies <= max_eV) * (energies >= min_eV)
    if not any(e_ind):
        raise ValueError('Supplied minimum or maximum energies '
                         'are out of range')
    energies = energies[e_ind]
    lut = np.empty((len(atts), 5, len(energies)), dtype='float64')
    for i, a in list(atts.items()):
        lut[i, 0, :] = energies
        if ignore_air and air_key == i:
            lut[i, 1:, :] = 0
        else:
            for j, key in enumerate(['total', 'rayleigh', 'photoelectric',
                                     'compton']):
                lut[i, j+1, :] = np.interp(energies, a['energy'], a[key])
    return lut

def prepare_geometry_from_organ_array(organ, organ_material_map, scale, materials):
        """genereate material and density arrays and material map from
           a list of materials to use
           INPUT:
               specter for this study
               list of materials
           OUTPUT :
               material_map, material_array, density_array
        """
        if organ is None:
            raise ValueError('No organ definitions in phantom')

        organ_material_map = recarray_to_dict(organ_material_map,
                                              value_is_string=True)
        #testing if we have the materials we need
        material_names = {m.name: m for m in materials}
        for value in organ_material_map.values():
            if not value in material_names:
                raise ValueError('Phantom requires missing material {}'.format(value))

#        organ = affine_transform(organ,
#                                 scale,
#                                 output_shape=np.floor(np.array(organ.shape)/scale),
#                                 cval=0, output=np.uint8, prefilter=True,
#                                 order=0).astype(np.uint8)
        scale_int = list([int(s) for s in scale])
        organ = organ[::scale_int[0], ::scale_int[1], ::scale_int[2]]
        material_array = np.asarray(organ, dtype=np.uint8)
        density_array = np.zeros(organ.shape, dtype='float64')

        material_map = {}
        key = 0
        for i in np.unique(organ):
            try:
                material_name = organ_material_map[i]
            except:
                import pdb
                pdb.set_trace()
            if material_name not in material_map:
                material_map[material_name] = key
                key += 1

            ind = np.nonzero(organ == i)
            density_array[ind] = material_names[material_name].density
            material_array[ind] = material_map[material_name]


        # reversing material_map
        material_map = {key: value for value, key in material_map.items()}
        return material_map, material_array.astype('int32'), density_array

def attinuation_to_ct_numbers(material_attinuations, air_key, water_key):
    a = 1000./(material_attinuations[water_key] -material_attinuations[air_key])
    b = -a * material_attinuations[water_key]
    atts = []

    for key, att in material_attinuations.items():
        HU = a * att + b
        if HU > 1000:
            HU=1000
        atts.append((key,HU))
    return atts

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
#        ctarray = affine_transform(spline_filter(ctarray,
#                                                 order=3,
#                                                 output=np.int16),
#                                   scale,
#                                   output_shape=np.floor(np.array(ctarray.shape)/scale),
#                                   cval=-1000, output=np.int16, prefilter=False)
        ctarray = rebin(ctarray, scale)
        specter = (specter[0], specter[1]/specter[1].mean())

        water_key = None
        material_map = {}
        material_att = {}
        material_dens = {}
        materials = [m for m in materials if m.organic]
        materials.sort(key=lambda x: x.density)
        for i, mat in enumerate(materials):
            material_map[i] = mat.name
            material_dens[i] = float(mat.density)
            # interpolationg and integrating attinuation coefficient
#            material_att[i] =
#            material_att[i] = np.trapz(np.interp(specter[0],
#                                                 mat.attinuation['energy'],
#                                                 mat.attinuation['total']),
#                                       specter[0])
            material_att[i] = np.sum(np.exp(-np.interp(specter[0],
                                     mat.attinuation['energy'],
                                     mat.attinuation['total']))*specter[1]
                           )

            material_att[i] *= mat.density

            if mat.name == 'water':
                water_key = i
            elif mat.name == 'air':
                air_key = i

        assert water_key is not None  # we need to include water in materials
        assert air_key is not None  # we need to include water in materials

        # getting a list of attinuation
        material_HU_list = [(key, (att / material_att[water_key] - 1.)*1000.)
                            for key, att in material_att.items()]
        material_HU_list = attinuation_to_ct_numbers(material_att, air_key, water_key)
        material_HU_list.sort(key=lambda x: x[1])
        HU_bins = (np.array(material_HU_list)[:-1, 1] +
                   np.array(material_HU_list)[1:, 1]) / 2.

        if HU_bins[-2] < 300:
            HU_bins[-1] = 300

        material_array = np.digitize(
            ctarray.ravel(), HU_bins).reshape(ctarray.shape).astype('int32')
        # Using densities fromdefined materials
        density_array = np.asarray(material_array, dtype='float64')
        np.choose(material_array,
                  [material_dens[i] for i, _ in material_HU_list],
                  out=density_array)

        # using densities defined by HU units in ct array
        HU_list = np.zeros(len(material_HU_list), dtype=np.int)
        for ind, HU in material_HU_list:
            HU_list[ind] = HU
        ctarray_model = np.choose(material_array, HU_list)
        density_array *= np.clip((ctarray - ctarray.min()+1) / (ctarray_model - ctarray.min() + 1), 0.5, 1.5)
        ##########################

        return material_map, material_array, density_array


def ct_runner_validate_simulation(materials, simulation, ctarray=None, organ=None,
                                  organ_material_map=None):
    """
    validating a ct mc simulation
    """

    # testing for required attributes
    if ctarray is not None:
        logger.info('Recalculating material mapping from CT array for {}'.format(simulation['name']))
        specter = tungsten_specter(simulation['aquired_kV'],
                                   filtration_materials='al',
                                   filtration_mm=simulation['al_filtration'])
        vals = prepare_geometry_from_ct_array(ctarray,
                                              simulation['scaling'],
                                              specter,
                                              materials)
    elif (organ is not None) and (organ_material_map is not None):
        logger.info('Recalculating material mapping from organ array for {}'.format(simulation['name']))

        vals = prepare_geometry_from_organ_array(organ,
                                                 organ_material_map,
                                                 simulation['scaling'],
                                                 materials)
    else:
        logger.warning('CT study {} has no CT images or organ arrays. Simulation not '
                       'started'.format(simulation['name']))
        raise ValueError('CT study {} must have CT images or organ array to run a '
                         'simulation'.format(simulation['name']))


    material_map = vals[0]
    material = vals[1]
    density = vals[2]

    # Testing for required materials if the simulation have a material_map
    material_names = [m.name for m in materials]
    for m_name in material_map.values():
        if m_name not in material_names:
            raise ValueError('Provided materials are not in ct study')

    # test for correct material geometry and mapping
    materials_ind = list(np.unique(material))
    materials_key = list(material_map.keys())
    materials_range = list(range(len(materials_ind)))

    try:
        for ind, key in zip(materials_ind, materials_key):
            assert key == ind
            assert key in materials_range

    except AssertionError:
        logger.warning('Something went wrong with material definitions, '
                       'attempting to recreate material mapping for '
                       'current simulation.')
        material_map = None
        raise ValueError('Error in material definitions for simulation')
    return material.astype('int32'), material_map, density.astype('float64')


def ct_runner(materials, simulation, ctarray=None, organ=None,
              organ_material_map=None, exposure_modulation=None,
              energy_imparted_to_dose_conversion=True, callback=None,
              energy_imparted=None, **kwargs):
    """Runs a MC simulation on a simulation object, and updates the
    energy_imparted property.

    INPUT:

        simulation :

        materials : a list of Material instances to be used in the simulation

        ignore_air : [optional] If set, ignores the material 'air' in
            MC calculation

    OUTPUT:
        None, but updates the energy_imparted property of simulation
    """
    logger.info('Preparing simulation for {}'.format(simulation['name']))
    if simulation['is_phantom']:
        materials_organic = [m for m in materials]
    else:
        materials_organic = [m for m in materials if m.organic]

    # Validating if everything is in place
    material, material_map, density = ct_runner_validate_simulation(materials, simulation,
                                                                    ctarray=ctarray,
                                                                    organ_material_map=organ_material_map,
                                                                    organ=organ)

    phase_space = ct_source_space(simulation, exposure_modulation)

    N = np.array(material.shape, dtype='int32')

    offset = np.zeros(3, dtype='float64')
    spacing = (simulation['spacing'] * simulation['scaling']).astype('float64')

    lut = generate_attinuation_lut(materials_organic, material_map,
                                   max_eV=500.e3,
                                   ignore_air=simulation['ignore_air'])
    del energy_imparted
    energy_imparted = None
    if energy_imparted is None:
        energy_imparted = np.zeros_like(density, dtype='float64')


    tot_histories = simulation['histories'] * simulation['exposures']

    if tot_histories > 1e7:
        coffe_msg = ', go get coffe!'
    else:
        coffe_msg = '.'
    logger.info('{0}: Starting simulation with about {1} histories per '
                'rotation{2}'.format(time.ctime(), tot_histories, coffe_msg))

    lut_shape = np.array(lut.shape, dtype='int32')

    logger.info('Initializing geometry')
    use_siddon = np.array([simulation['use_siddon']], dtype='int32')
    engine = Engine()
    geometry = engine.setup_simulation(N, spacing, offset, material, density, lut_shape, lut, energy_imparted, use_siddon)

    start_exposure = simulation['start_at_exposure_no']
    time_start = time.clock()
    exposure_time = time_start

    for p, e, n in phase_space:
        source = engine.setup_source_bowtie(*p)
        engine.run_bowtie(source, simulation['histories'], geometry)
        engine.cleanup(source=source)

        if (time.clock() - exposure_time) > 5:
            eta = log_elapsed_time(time_start, e+1, n, start_exposure)
            if callback is not None:
                callback(simulation['name'], progressbar_data=[np.squeeze(energy_imparted.max(axis=0)), spacing[1] ,spacing[2] ,eta, True])
#                simulation['start_at_exposure_no'] = e + 1
            exposure_time = time.clock()


    engine.cleanup(simulation=geometry)
#    time_start = time.clock()
#    for p, e, n in phase_space:
#        score_energy(p, N, spacing, offset, material,
#                     density, lut, energy_imparted)
#        eta = log_elapsed_time(time_start, e+1, n, start_exposure)
#        if callback is not None:
#            callback(simulation['name'], {'energy_imparted': energy_imparted}, e + 1, eta)
#        simulation['start_at_exposure_no'] = e + 1

    if callback is not None:
        callback(simulation['name'], progressbar_data=[np.squeeze(energy_imparted.max(axis=0)), spacing[1] ,spacing[2] ,'Preforming dose calibration', True])
#        callback(simulation['name'], {'energy_imparted': None}, e + 1, 'Preforming dose calibration')

    generate_dose_conversion_factor(simulation, materials, callback)

    if callback is not None:
        callback(simulation['name'], {'energy_imparted': None}, e + 1, progressbar_data=[np.squeeze(energy_imparted.max(axis=1)), spacing[0] ,spacing[2] ,'Done', False])
    simulation['start_at_exposure_no'] = 0
    simulation['MC_finished'] = True
    simulation['MC_running'] = False
    simulation['MC_ready'] = False
    return simulation, {'density': density, 'material': material, 'material_map': material_map, 'energy_imparted': energy_imparted}


def generate_dose_conversion_factor(simulation, materials, callback=None):
    air, pmma = None, None
    for m in materials:
        if m.name == 'pmma':
            pmma = m
        elif m.name == 'air':
            air = m

    if (simulation['ctdi_air100'] > 0.) and (air is not None):
        obtain_ctdiair_conversion_factor(simulation, air, callback=callback)
    if (simulation['ctdi_w100'] > 0.) and (pmma is not None) and (air is not None):
        size = simulation['ctdi_phantom_diameter']
        obtain_ctdiw_conversion_factor(simulation, pmma, air, size=size, callback=callback)
#    else:
#        msg = """Need a combination of air material and ctdi air or ctdi_w100
#                 pmma material and ctdiw_100 to generate energy to dose
#                 conversion factor."""
#        logger.warning(msg)



def obtain_ctdiair_conversion_factor(simulation, air_material, callback=None):

    logger.info('Starting simulating CTDIair100 measurement for '
                '{0}. CTDIair100 is {1}mGy'.format(simulation['name'], simulation['ctdi_air100']))
    spacing = np.array((2, 2, 10), dtype='float64')

    N = np.rint(np.array((simulation['sdd'] / spacing[0],
                          simulation['sdd'] / spacing[1], 3),
                         dtype='int32')).astype('int')

    offset = (-N * spacing / 2.).astype('float64')
    material_array = np.zeros(N, dtype='int32')
    material_map = {0: air_material.name}
    density_array = np.zeros(N, dtype='float64') + air_material.density

    lut = generate_attinuation_lut([air_material], material_map, max_eV=0.15e6).astype('float64')
    lut_shape = np.array(lut.shape, dtype='int32')
    dose = np.zeros_like(density_array, dtype='float64')

    
    total_collimation = simulation['detector_rows'] * simulation['detector_width']


    engine = Engine()
    use_siddon = np.array([simulation['use_siddon']], dtype=np.int)
    geometry = engine.setup_simulation(N, spacing, offset, material_array, density_array, lut_shape, lut, dose, use_siddon)
    teller = 0
    center = np.floor(N / 2).astype(np.int)
    center_dose = 0
    t1 = time.clock()
    while center_dose < 150000:
        if teller > 0:
            logger.debug('Not sufficient data, running again. Dose in center is now {0}, max dose: {1}.'.format(center_dose, dose.max()))
        teller += 1
        
        if simulation['use_tube_B']:
            wA = simulation.get('tube_weight_A')
            wB = simulation.get('tube_weight_A')
            wS = wA + wB
            en_specter_A = tungsten_specter(simulation['kV_A'], 
                              angle_deg=simulation['anode_angle'],
                              filtration_materials='Al',
                              filtration_mm=simulation['al_filtration'])
        
            phase_space_A = ct_seq(simulation['scan_fov'], simulation['sdd'],
                                 total_collimation, start=0, stop=0, step=1,
                                 exposures=simulation['exposures'],
                                 histories=simulation['histories'],
                                 energy_specter=en_specter_A,
                                 bowtie_radius=simulation['bowtie_radius'],
                                 bowtie_distance=simulation['bowtie_distance'],
                                 weight=wA/wS                
                                 )
            en_specter_B = tungsten_specter(simulation['kV_B'], 
                              angle_deg=simulation['anode_angle'],
                              filtration_materials='Al',
                              filtration_mm=simulation['al_filtration'])
        
            phase_space_B = ct_seq(simulation['scan_fov'], simulation['sdd'],
                                 total_collimation, start=0, stop=0, step=1,
                                 exposures=simulation['exposures'],
                                 histories=simulation['histories'],
                                 energy_specter=en_specter_B,
                                 bowtie_radius=simulation['bowtie_radius'],
                                 bowtie_distance=simulation['bowtie_distance'],
                                 weight=wB/wS                
                                 )
            phase_space = itertools.chain(phase_space_A, phase_space_B)
            
        else:
    
            en_specter = tungsten_specter(simulation['kV_A'], 
                                          angle_deg=simulation['anode_angle'],
                                          filtration_materials='Al',
                                          filtration_mm=simulation['al_filtration'])
            
            phase_space = ct_seq(simulation['scan_fov'], simulation['sdd'],
                                 total_collimation, start=0, stop=0, step=1,
                                 exposures=simulation['exposures'],
                                 histories=simulation['histories']/2,
                                 energy_specter=en_specter,
                                 bowtie_radius=simulation['bowtie_radius'],
                                 bowtie_distance=simulation['bowtie_distance']
                                 )


        for batch, e, n in phase_space:
            source = engine.setup_source_bowtie(*batch)
            engine.run_bowtie(source, simulation['histories'], geometry)
            engine.cleanup(source=source)
    #        break
            if (time.clock() - t1) > 1:
#                eta = log_elapsed_time(t0, e+1, n, 0)
                t1 = time.clock()
                if callback:
                    callback(simulation['name'], progressbar_data=[np.squeeze(dose.max(axis=2)), spacing[0] ,spacing[1] ,'Run number {0}'.format(teller+1), True])
#                    callback(simulation['name'], {'energy_imparted':dose}, 0, '', save=False)
        center_dose += np.sum(dose[center[0], center[1], center[2]])
    callback(simulation['name'], progressbar_data=[np.squeeze(dose.max(axis=2)), spacing[0] ,spacing[1] ,'Done', True])
    engine.cleanup(simulation=geometry)

#    engine.cleanup(simulation=geometry, energy_imparted=dose)
#    dose = gaussian_filter(dose, (1., 1., 0.))

    d = center_dose / teller / (air_material.density * np.prod(spacing)) / simulation['pitch']
    simulation['conversion_factor_ctdiair'] = np.nan_to_num(1. / d * total_collimation)
    return dose

def generate_ctdi_phantom(simulation, pmma, air, size=32., callback=None):
    spacing = np.array((.1, .1, 2.5), dtype='float64')
#    N = np.rint(np.array((simulation['sdd'] / spacing[0],
#                          simulation['sdd'] / spacing[0], 6),
#                         )).astype('int32')
    N = np.rint(np.array((size*1.15 / spacing[0],
                          size*1.15 / spacing[0], 6),
                         )).astype('int32')

    offset = (-N * spacing / 2.).astype('float64')
    material_array = np.zeros(N, dtype='int32')
    radii_phantom = size / spacing[0] / 2
    radii_meas = (1.3 / 2 / spacing[0])
    center = (N / 2.)[:2]
    radii_pos = radii_phantom - (1 + 1.3 / 2) / spacing[0]
    pos = [(center[0], center[1])]

    for ang in [0, 90, 180, 270]:
        dx = radii_pos * np.sin(np.deg2rad(ang))
        dy = radii_pos * np.cos(np.deg2rad(ang))
        pos.append((center[0] + dx, center[1] + dy))

    measure_indices = []

    for i in range(int(N[2])):
        material_array[:, :, i] += circle_mask((N[0], N[1]),
                                               radii_phantom)
#        for x, y in pos:
#            material_array
        for p in pos:
            mask = circle_mask((N[0], N[1]), radii_meas, center=p)
            material_array[:, :, i] += mask
            if i == 0:
                measure_indices.append(np.argwhere(mask))

    material_map = {0: air.name, 1: pmma.name, 2: air.name}
    density_array = np.zeros_like(material_array, dtype='float64')
    density_array[material_array == 0] = air.density
    density_array[material_array == 1] = pmma.density
    density_array[material_array == 2] = air.density

    lut = generate_attinuation_lut([air, pmma], material_map)
    return N, spacing, offset, material_array, density_array, lut, measure_indices


def obtain_ctdiw_conversion_factor(simulation, pmma, air,
                                   size=32., callback=None):

    logger.info('Starting simulating CTDIw100 measurement for '
                '{}'.format(simulation['name']))
    args = generate_ctdi_phantom(simulation, pmma, air, size=size)
    N, spacing, offset, material_array, density_array, lut, meas_pos = args

    lut_shape = np.array(lut.shape, dtype='int32')

    dose = np.zeros_like(density_array, dtype='float64')

    total_collimation = simulation['detector_rows'] * simulation['detector_width']


    
    
    if simulation['use_tube_B']:
        wA = simulation.get('tube_weight_A')
        wB = simulation.get('tube_weight_A')
        wS = wA + wB
        en_specter_A = tungsten_specter(simulation['kV_A'], 
                          angle_deg=simulation['anode_angle'],
                          filtration_materials='Al',
                          filtration_mm=simulation['al_filtration'])
    
        phase_space_A = ct_seq(simulation['scan_fov'], simulation['sdd'],
                             total_collimation, start=0, stop=0, step=1,
                             exposures=simulation['exposures'],
                             histories=simulation['histories'],
                             energy_specter=en_specter_A,
                             bowtie_radius=simulation['bowtie_radius'],
                             bowtie_distance=simulation['bowtie_distance'],
                             weight=wA/wS                
                             )
        en_specter_B = tungsten_specter(simulation['kV_B'], 
                          angle_deg=simulation['anode_angle'],
                          filtration_materials='Al',
                          filtration_mm=simulation['al_filtration'])
    
        phase_space_B = ct_seq(simulation['scan_fov'], simulation['sdd'],
                             total_collimation, start=0, stop=0, step=1,
                             exposures=simulation['exposures'],
                             histories=simulation['histories'],
                             energy_specter=en_specter_B,
                             bowtie_radius=simulation['bowtie_radius'],
                             bowtie_distance=simulation['bowtie_distance'],
                             weight=wB/wS                
                             )
        phase_space = itertools.chain(phase_space_A, phase_space_B)
        
    else:

        en_specter = tungsten_specter(simulation['kV_A'], 
                                      angle_deg=simulation['anode_angle'],
                                      filtration_materials='Al',
                                      filtration_mm=simulation['al_filtration'])
        
        phase_space = ct_seq(simulation['scan_fov'], simulation['sdd'],
                             total_collimation, start=0, stop=0, step=1,
                             exposures=simulation['exposures'],
                             histories=simulation['histories']/2,
                             energy_specter=en_specter,
                             bowtie_radius=simulation['bowtie_radius'],
                             bowtie_distance=simulation['bowtie_distance']
                             )

    
                         
                         
    use_siddon = np.array([simulation['use_siddon']], dtype=np.int)
    engine = Engine()
    geometry = engine.setup_simulation(N, spacing, offset, material_array,
                            density_array, lut_shape, lut, dose, use_siddon)


    history_factor = int(1e8 / simulation['histories'] / simulation['exposures'])
    if history_factor < 1:
        history_factor = 1
    histories = simulation['histories'] * history_factor
    t0 = time.clock()
    t1 = t0


    for batch, e, n in phase_space:
        source = engine.setup_source_bowtie(*batch)
        engine.run_bowtie(source, histories, geometry)
        engine.cleanup(source=source)

        if (time.clock() - t1) > 5:
            eta = log_elapsed_time(t0, e+1, n, 0)
            if callback:
                callback(simulation['name'], progressbar_data=[np.squeeze(dose.max(axis=2)), spacing[0] ,spacing[1] ,eta, True])
#                callback(simulation['name'], {'energy_imparted':dose}, 0, '', save=False)
            t1 = time.clock()

    if callback:
        callback(simulation['name'], progressbar_data=[np.squeeze(dose.max(axis=2)), spacing[0] ,spacing[1] ,'Done', True])
#        callback(simulation['name'], {'energy_imparted':dose}, 0, 'Done', save=False)

    dose /= history_factor

    engine.cleanup(simulation=geometry)
#
#    plt.imshow(dose[:,:,1])
#    plt.show()
#    dose = gaussian_filter(dose, [.5, .5, .1])
#    plt.imshow(dose[:,:,1])
#    plt.show()
    d = []
    for p in meas_pos:
        x, y = p[:, 0], p[:, 1]
        d.append(dose[x, y, 1:-2].mean() / (air.density * np.prod(spacing)))
        dose[x, y, 1:-2] = dose.max()
    logger.debug('Estimated dose for CTDI phantom; center: {0}, periphery: {1}, {2}, {3}, {4}.'.format(*d))
#    pdb.set_trace()
    ctdiv = d.pop(0) / 3.
    ctdiv += (2. * sum(d) / 3. / 4.)
    simulation['conversion_factor_ctdiw'] = np.nan_to_num(simulation['ctdi_w100'] / ctdiv * total_collimation)
#    pdb.set_trace()
#    import pdb
#    pdb.set_trace()
    return dose
