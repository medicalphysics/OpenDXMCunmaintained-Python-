# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:05:21 2016

@author: ander
"""

import numpy as np

from opendxmc.engine import Engine
from opendxmc.database import get_stored_materials
from opendxmc.runner import generate_attinuation_lut
from opendxmc.tube.tungsten import specter, attinuation
from opendxmc.utils import sphere_mask
import time
from matplotlib import pylab as plt
from scipy.ndimage.filters import gaussian_filter

def get_performance():
    presicion = 'float64'
    pathing = 'Siddon'
    n_voxels = [64, 128, 256, 512]
    n_spacing = [.1, .5, 1]
    n_histories = [10000, 50000 ,100000]
    
    for v in n_voxels:
        for s in n_spacing:
            for h in n_histories:
                preformance_example(presicion, pathing, v, s, h)

def preformance_example(presicion, pathing, n_voxels, n_spacing, n_histories):
        # get_stored_materials returns an iterator of materials already stored in 
    # the application. 

#    presicion = 'float64'
#    pathing = 'Woodcock'
#    n_voxels = 50
#    n_spacing = 1.0
#    n_histories = 1000

    materials_dict = {mat.name: mat for mat in get_stored_materials()}
    materials = list(materials_dict.values())
    print('Listing imported materials:')
    for ind, m in enumerate(materials):
        print('{0}: Name: {1}, Density {2}[g/cm3]'.format(ind, m.name, m.density))

    
    # lets create a water box surronded by air, we need a numpy array of 
    # material indices and a material table
    
    # First we specify dimensions af the voxelized box    
    N = np.array([n_voxels, n_voxels, n_voxels], dtype='int')    
    # Then spacing of each voxel in cm
    spacing = np.array([n_spacing, n_spacing, n_spacing], dtype=presicion)    
    
    # lets specify a lookup table as a dictionary for the materials we are 
    # using. The key in the dictionary corresponds to the values in the 
    # material_indices array
    material_lut = {0: 'air', 1: 'water', 2: 'lead'}
    material_indices = np.zeros(N, dtype='int')
    #Lets fill a region of the box with water
#    material_indices[20:-20, 20:-20, 20:-20] = 1  
    material_indices += sphere_mask(material_indices.shape, n_voxels // 2)
#    material_indices[60:-60, 60:-60, 60:-60] = 2
    
    # Now we create a density array as same shape as the material_indices array
    # We could spesify different densities for each voxel, but we are just 
    # going to give each material its room temperature default density
    
    air_material = materials_dict['air']
    water_material = materials_dict['water']
    lead_material = materials_dict['lead']
    
    densities = np.empty(N, dtype=presicion)
    densities[material_indices==0] = air_material.density
    densities[material_indices==1] = water_material.density
#    densities[60:-60, 60:-60, 60:-60] = lead_material.density
    # Next we need to get the attinuation lookup table for the specified 
    # materials. This is a numpy float64 array with shape: 
    # [number_of_materials, 5, number_of_energies_we_have_interaction_constants_at]
    # The generate_attinuation_lut function is a conveniance function that will 
    # generate this LUT
    lut = generate_attinuation_lut([air_material, water_material, lead_material], material_lut)
    lut = lut.astype(presicion)
    #Now from the lut we can plot the attenuation coefficients for water:
    plt.subplot(2, 3, 1)
    plt.loglog(lut[1, 0, :], lut[1, 1, :], label='Total', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 2, :], label='Rayleigh scattering', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 3, :], label='Photoelectric effect', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 4, :], label='Compton scattering', linewidth=5)
    plt.legend()
    plt.title('Attenuation coefficients for water')
    plt.ylabel('Attenuation coefficient [$cm^2 / g$]')
    plt.xlabel('Energy [$eV$]')
    
    
    # Now we are ready to set up a simulation:
    # initializing the monte carlo engine
    engine = Engine(precision = presicion)    
    # In the simulation geometry the first voxel will have coordinates (0, 0, 0)
    # we can specify an offset to set the center of our box to origo
    offset = -N * spacing / 2. 
    offset = offset.astype(presicion)
    # we also need the lut shape as an array
    lut_shape = np.array(lut.shape, dtype='int')
    # and an array to store imparted energy
    energy_imparted = np.zeros_like(densities, dtype=presicion)
    simulation = engine.setup_simulation(N, 
                                         spacing, 
                                         offset, 
                                         material_indices, 
                                         densities, 
                                         lut_shape, 
                                         lut,
                                         energy_imparted)
    
    
    # Next we setup a beam source
    sdd = np.array([119], dtype=presicion)
    collimation = np.array([3.8], dtype=presicion)
    fov = np.array([50], dtype=presicion)
    
    source_position = np.array([-sdd[0]/2, 0, 0], dtype=presicion)    
    source_direction = np.array([1, 0, 0], dtype=presicion) # this needs to be a unit vector
    scan_axis = np.array([0, 0, 1], dtype=presicion) # this needs to be a unit vector and orthonormal to source direction
    # The fan angle of the beam in scan_axid direction is gives as angle = arctan(collimation / sdd)
    
    # the fan angle of the beam in source_direction cross scan_axis is given as angle = arctan(2 * fov / sdd)
    rot_fan_angle = np.array([np.arctan(fov[0]/sdd[0]) * 2],dtype=presicion)
    scan_fan_angle = np.array([np.arctan(collimation[0] *.5 / sdd[0]) * 2], dtype=presicion)
    
    
    
    #To define wich photon energies we will draw from, we need to specify a specter cummulative propability distribution 
    # and the corresponding energies. To only draw three equaly probable energies, we may specify this as following
#    specter_probabilities = np.array([0, .25, .25, .50], dtype='float64')
#    specter_probabilities /= specter_probabilities.sum() # we normalize to be certain we get a valid cum. prob. dist.
#    specter_energies = np.array([0, 30e3, 60e3, 90e3], dtype='float64') # energy in eV, here 30, 60 and 90 keV
    specter_energies, specter_probabilities = specter(120., angle_deg=12, filtration_materials='al', filtration_mm=7.)
    specter_probabilities/=specter_probabilities.sum()
    specter_cpd = np.cumsum(specter_probabilities, dtype=presicion)
    specter_cpd /= specter_cpd[-1]
    specter_energies=specter_energies.astype(presicion)
    
    #generating bowtile filter
    bowtie_angle = np.linspace(-rot_fan_angle[0]/2, rot_fan_angle[0]/2, 200, dtype=presicion)
    bowtie_lenghts= (1/np.cos(bowtie_angle) - 1)
    bowtie_weighs = np.empty_like(bowtie_angle, dtype=presicion)
    bowtie_att = attinuation(specter_energies/1000, name='aluminum', density=True).astype(presicion)
    
    for i in range(bowtie_lenghts.shape[0]):
        bowtie_weighs[i] = np.sum(specter_probabilities*np.exp(-bowtie_att*bowtie_lenghts[i]))
     
    # last we may specify a weight factor for the source. This should be 1 
    # unless you create multiple sources and want to apply differet weights 
    # for each source/beam
    weight = np.array([1], dtype=presicion)
    # We now have all we need to specify a beam source, lets create one:
    beam = engine.setup_source_bowtie(source_position,
                                      source_direction, 
                                      scan_axis, 
                                      scan_fan_angle, 
                                      rot_fan_angle, 
                                      weight, 
                                      specter_cpd, 
                                      specter_energies,
                                      bowtie_weighs,
                                      bowtie_angle)
                               
    # Running the simulation
    
    t0 = time.clock()
    engine.run_bowtie(beam, n_histories, simulation)
    t0 = time.clock() - t0
    print('Simulated {0} photons in {1} seconds'.format(n_histories, t0))                           

    # let's add one more beam to the simulation
    source_position_2 = np.array([0, sdd[0]/2, 0], dtype=presicion)    
    source_direction_2 = np.array([0, -1, 0], dtype=presicion)  # th
    beam2 = engine.setup_source_bowtie(source_position_2,
                                       source_direction_2, 
                                       scan_axis, 
                                       scan_fan_angle, 
                                       rot_fan_angle, 
                                       weight, 
                                       specter_cpd, 
                                       specter_energies,
                                       bowtie_weighs,
                                       bowtie_angle)
    
    t1 = time.clock()
    engine.run_bowtie(beam2, n_histories, simulation)
    t1 = time.clock()-t1
    print('Simulated another {0} photons in {1} seconds'.format(n_histories, t1))                           

#    with open("performance.txt", 'a') as f:
#        f.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(pathing, n_voxels, n_spacing, n_histories, presicion, t0, t1))

    #cleanup of simulation and sources, the monte carlo engine will leak
    # memory if these functions are not called
    engine.cleanup(simulation=simulation, energy_imparted=energy_imparted)
#    engine.cleanup(source=beam)
#    engine.cleanup(source=beam2)



    plt.subplot(2, 3, 2)
    plt.imshow(material_indices[:, :, N[2] // 2], cmap='gray')
    plt.title('Material index')
    
    plt.subplot(2, 3, 3)
    plt.plot(sdd[0]/2 * np.sin(bowtie_angle) + 25, bowtie_weighs)
    plt.xlim((0, 50))
    
#    plt.ylim((0, 1))
#    dose = gaussian_filter(energy_imparted, 1) / (np.prod(spacing) * densities)
    dose = energy_imparted / (np.prod(spacing) * densities)
    plt.subplot(2, 3, 4)
    plt.imshow(np.log(dose[:, :, N[2] // 2]))
    plt.title('XY Logarithm of dose [eV / grams]')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.log(dose[N[0]//2, :, :]))
    plt.title('YZ Logarithm of dose [eV / grams]')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.log(dose[:, N[1]//2, :]))
    plt.title('XZ Logarithm of dose [eV / grams]')
    
    
    plt.show()    



def bowtie_example():
        # get_stored_materials returns an iterator of materials already stored in 
    # the application. 
    materials_dict = {mat.name: mat for mat in get_stored_materials()}
    materials = list(materials_dict.values())
    print('Listing imported materials:')
    for ind, m in enumerate(materials):
        print('{0}: Name: {1}, Density {2}[g/cm3]'.format(ind, m.name, m.density))

    
    # lets create a water box surronded by air, we need a numpy array of 
    # material indices and a material table
    
    # First we specify dimensions af the voxelized box    
    N = np.array([50, 50, 50], dtype='int')    
    # Then spacing of each voxel in cm
    spacing = np.array([1, 1, 1], dtype='float64')    
    
    # lets specify a lookup table as a dictionary for the materials we are 
    # using. The key in the dictionary corresponds to the values in the 
    # material_indices array
    material_lut = {0: 'air', 1: 'water', 2: 'lead'}
    material_indices = np.zeros(N, dtype='int')
    #Lets fill a region of the box with water
#    material_indices[20:-20, 20:-20, 20:-20] = 1  
    material_indices += sphere_mask(material_indices.shape, 25)
#    material_indices[60:-60, 60:-60, 60:-60] = 2
    
    # Now we create a density array as same shape as the material_indices array
    # We could spesify different densities for each voxel, but we are just 
    # going to give each material its room temperature default density
    
    air_material = materials_dict['air']
    water_material = materials_dict['water']
    lead_material = materials_dict['lead']
    
    densities = np.empty(N, dtype='float64')
    densities[material_indices==0] = air_material.density
    densities[material_indices==1] = water_material.density
#    densities[60:-60, 60:-60, 60:-60] = lead_material.density
    # Next we need to get the attinuation lookup table for the specified 
    # materials. This is a numpy float64 array with shape: 
    # [number_of_materials, 5, number_of_energies_we_have_interaction_constants_at]
    # The generate_attinuation_lut function is a conveniance function that will 
    # generate this LUT
    lut = generate_attinuation_lut([air_material, water_material, lead_material], material_lut)
    
    #Now from the lut we can plot the attenuation coefficients for water:
    plt.subplot(2, 3, 1)
    plt.loglog(lut[1, 0, :], lut[1, 1, :], label='Total', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 2, :], label='Rayleigh scattering', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 3, :], label='Photoelectric effect', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 4, :], label='Compton scattering', linewidth=5)
    plt.legend()
    plt.title('Attenuation coefficients for water')
    plt.ylabel('Attenuation coefficient [$cm^2 / g$]')
    plt.xlabel('Energy [$eV$]')
    
    
    # Now we are ready to set up a simulation:
    # initializing the monte carlo engine
    engine = Engine()    
    # In the simulation geometry the first voxel will have coordinates (0, 0, 0)
    # we can specify an offset to set the center of our box to origo
    offset = -N * spacing / 2. 
    # we also need the lut shape as an array
    lut_shape = np.array(lut.shape, dtype='int')
    # and an array to store imparted energy
    energy_imparted = np.zeros_like(densities)
    simulation = engine.setup_simulation(N, 
                                         spacing, 
                                         offset, 
                                         material_indices, 
                                         densities, 
                                         lut_shape, 
                                         lut,
                                         energy_imparted)
    
    
    # Next we setup a beam source
    sdd = np.array([119], dtype='float64')
    collimation = np.array([3.8], dtype='float64')
    fov = np.array([50], dtype='float64')
    
    source_position = np.array([-sdd[0]/2, 0, 0], dtype='float64')    
    source_direction = np.array([1, 0, 0], dtype='float64') # this needs to be a unit vector
    scan_axis = np.array([0, 0, 1], dtype='float64') # this needs to be a unit vector and orthonormal to source direction
    # The fan angle of the beam in scan_axid direction is gives as angle = arctan(collimation / sdd)
    
    # the fan angle of the beam in source_direction cross scan_axis is given as angle = arctan(2 * fov / sdd)
    rot_fan_angle = np.array([np.arctan(fov[0]/sdd[0]) * 2],dtype='float64')
    scan_fan_angle = np.array([np.arctan(collimation[0] *.5 / sdd[0]) * 2], dtype='float64')
    
    
    
    #To define wich photon energies we will draw from, we need to specify a specter cummulative propability distribution 
    # and the corresponding energies. To only draw three equaly probable energies, we may specify this as following
#    specter_probabilities = np.array([0, .25, .25, .50], dtype='float64')
#    specter_probabilities /= specter_probabilities.sum() # we normalize to be certain we get a valid cum. prob. dist.
#    specter_energies = np.array([0, 30e3, 60e3, 90e3], dtype='float64') # energy in eV, here 30, 60 and 90 keV
    specter_energies, specter_probabilities = specter(120., angle_deg=12, filtration_materials='al', filtration_mm=7.)
    specter_probabilities/=specter_probabilities.sum()
    specter_cpd = np.cumsum(specter_probabilities, dtype='float64')
    specter_cpd /= specter_cpd[-1]
    specter_energies=specter_energies.astype('float64')
    
    #generating bowtile filter
    bowtie_angle = np.linspace(-rot_fan_angle[0]/2, rot_fan_angle[0]/2, 200, dtype='float64')
    bowtie_lenghts= (1/np.cos(bowtie_angle) - 1)
    bowtie_weighs = np.empty_like(bowtie_angle)
    bowtie_att = attinuation(specter_energies/1000, name='aluminum', density=True)
    
    for i in range(bowtie_lenghts.shape[0]):
        bowtie_weighs[i] = np.sum(specter_probabilities*np.exp(-bowtie_att*bowtie_lenghts[i]))
     
    # last we may specify a weight factor for the source. This should be 1 
    # unless you create multiple sources and want to apply differet weights 
    # for each source/beam
    weight = np.array([1], dtype='float64')
    # We now have all we need to specify a beam source, lets create one:
    beam = engine.setup_source_bowtie(source_position,
                                      source_direction, 
                                      scan_axis, 
                                      scan_fan_angle, 
                                      rot_fan_angle, 
                                      weight, 
                                      specter_cpd, 
                                      specter_energies,
                                      bowtie_weighs,
                                      bowtie_angle)
                               
    # Running the simulation
    n_histories = 100000
    t0 = time.clock()
    engine.run_bowtie(beam, n_histories, simulation)
    print('Simulated {0} photons in {1} seconds'.format(n_histories, time.clock()-t0))                           

    # let's add one more beam to the simulation
    source_position_2 = np.array([sdd[0]/2, 0, 0], dtype='float64')    
    source_direction_2 = np.array([-1, 0, 0], dtype='float64')  # th
    beam2 = engine.setup_source_bowtie(source_position_2,
                                       source_direction_2, 
                                       scan_axis, 
                                       scan_fan_angle, 
                                       rot_fan_angle, 
                                       weight, 
                                       specter_cpd, 
                                       specter_energies,
                                       bowtie_weighs,
                                       bowtie_angle)
    
    t0 = time.clock()
    engine.run_bowtie(beam2, n_histories, simulation)
    print('Simulated another {0} photons in {1} seconds'.format(n_histories, time.clock()-t0))                           

    #cleanup of simulation and sources, the monte carlo engine will leak
    # memory if these functions are not called
    engine.cleanup(simulation=simulation, energy_imparted=energy_imparted)
#    engine.cleanup(source=beam)
#    engine.cleanup(source=beam2)



    plt.subplot(2, 3, 2)
    plt.imshow(material_indices[:, :, N[2] // 2], cmap='gray')
    plt.title('Material index')
    
    plt.subplot(2, 3, 3)
    plt.plot(sdd[0]/2 * np.sin(bowtie_angle) + 25, bowtie_weighs)
    plt.xlim((0, 50))
    
#    plt.ylim((0, 1))
#    dose = gaussian_filter(energy_imparted, 1) / (np.prod(spacing) * densities)
    dose = energy_imparted / (np.prod(spacing) * densities)
    plt.subplot(2, 3, 4)
    plt.imshow(np.log(dose[:, :, N[2] // 2]))
    plt.title('XY Logarithm of dose [eV / grams]')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.log(dose[N[0]//2, :, :]))
    plt.title('YZ Logarithm of dose [eV / grams]')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.log(dose[:, N[1]//2, :]))
    plt.title('XZ Logarithm of dose [eV / grams]')
    
    
    plt.show()
    
    import pdb; pdb.set_trace()

def cone_beam_example():
    # get_stored_materials returns an iterator of materials already stored in 
    # the application. 
    materials_dict = {mat.name: mat for mat in get_stored_materials()}
    materials = list(materials_dict.values())
    print('Listing imported materials:')
    for ind, m in enumerate(materials):
        print('{0}: Name: {1}, Density {2}[g/cm3]'.format(ind, m.name, m.density))

    
    # lets create a water box surronded by air, we need a numpy array of 
    # material indices and a material table
    
    # First we specify dimensions af the voxelized box    
    N = np.array([128, 128, 128], dtype='int')    
    # Then spacing of each voxel in cm
    spacing = np.array([0.1, 0.1, 0.1], dtype='float64')    
    
    # lets specify a lookup table as a dictionary for the materials we are 
    # using. The key in the dictionary corresponds to the values in the 
    # material_indices array
    material_lut = {0: 'air', 1: 'water', 2: 'lead'}
    material_indices = np.zeros(N, dtype='int')
    #Lets fill a region of the box with water
    material_indices[20:-20, 20:-20, 20:-20] = 1  
    material_indices[60:-60, 60:-60, 60:-60] = 2
    
    # Now we create a density array as same shape as the material_indices array
    # We could spesify different densities for each voxel, but we are just 
    # going to give each material its room temperature default density
    
    air_material = materials_dict['air']
    water_material = materials_dict['water']
    lead_material = materials_dict['lead']
    
    densities = np.empty(N, dtype='float64')
    densities[:, :, :] = air_material.density
    densities[20:-20, 20:-20, 20:-20] = water_material.density
    densities[60:-60, 60:-60, 60:-60] = lead_material.density
    # Next we need to get the attinuation lookup table for the specified 
    # materials. This is a numpy float64 array with shape: 
    # [number_of_materials, 5, number_of_energies_we_have_interaction_constants_at]
    # The generate_attinuation_lut function is a conveniance function that will 
    # generate this LUT
    lut = generate_attinuation_lut([air_material, water_material, lead_material], material_lut)
    
    #Now from the lut we can plot the attenuation coefficients for water:
    plt.subplot(2, 2, 1)
    plt.loglog(lut[1, 0, :], lut[1, 1, :], label='Total', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 2, :], label='Rayleigh scattering', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 3, :], label='Photoelectric effect', linewidth=5)
    plt.loglog(lut[1, 0, :], lut[1, 4, :], label='Compton scattering', linewidth=5)
    plt.legend()
    plt.title('Attenuation coefficients for water')
    plt.ylabel('Attenuation coefficient [$cm^2 / g$]')
    plt.xlabel('Energy [$eV$]')
    
    
    # Now we are ready to set up a simulation:
    # initializing the monte carlo engine
    engine = Engine()    
    # In the simulation geometry the first voxel will have coordinates (0, 0, 0)
    # we can specify an offset to set the center of our box to origo
    offset = -N * spacing / 2. 
    # we also need the lut shape as an array
    lut_shape = np.array(lut.shape, dtype='int')
    # and an array to store imparted energy
    energy_imparted = np.zeros_like(densities)
    simulation = engine.setup_simulation(N, 
                                         spacing, 
                                         offset, 
                                         material_indices, 
                                         densities, 
                                         lut_shape, 
                                         lut,
                                         energy_imparted)
    
    
    # Next we setup a beam source

    source_position = np.array([-20, 0, 0], dtype='float64')    
    source_direction = np.array([1, 0, 0], dtype='float64') # this needs to be a unit vector
    scan_axis = np.array([0, 0, 1], dtype='float64') # this needs to be a unit vector and orthonormal to source direction
    # The fan angle of the beam in scan_axid direction is gives as angle = arctan(collimation / sdd)
    sdd = np.array([1], dtype='float64')
    collimation = np.array([0.25], dtype='float64') / 4
    # the fan angle of the beam in source_direction cross scan_axis is given as angle = arctan(2 * fov / sdd)
    fov = np.array([0.125], dtype='float64') / 4
    
    #To define wich photon energies we will draw from, we need to specify a specter cummulative propability distribution 
    # and the corresponding energies. To only draw three equaly probable energies, we may specify this as following
    specter_probabilities = np.array([.25, .25, .50], dtype='float64')
    specter_probabilities /= specter_probabilities.sum() # we normalize to be certain we get a valid cum. prob. dist.
    specter_energies = np.array([30e3, 60e3, 90e3], dtype='float64') # energy in eV, here 30, 60 and 90 keV
    specter_cpd = np.cumsum(specter_probabilities)

    # last we may specify a weight factor for the source. This should be 1 
    # unless you create multiple sources and want to apply differet weights 
    # for each source/beam
    weight = np.array([1], dtype='float64')
    # We now have all we need to specify a beam source, lets create one:
    beam = engine.setup_source(source_position,
                               source_direction, 
                               scan_axis, 
                               sdd, 
                               fov, 
                               collimation, 
                               weight, 
                               specter_cpd, 
                               specter_energies)
                               
    # Running the simulation
    n_histories = 1000000
    t0 = time.clock()
    engine.run(beam, n_histories, simulation)
    print('Simulated {0} photons in {1} seconds'.format(n_histories, time.clock()-t0))                           

    # let's add one more beam to the simulation
    source_position_2 = np.array([0, -20, 0], dtype='float64')    
    source_direction_2 = np.array([0, 1, 0], dtype='float64')  # th
    beam2 = engine.setup_source(source_position_2,
                               source_direction_2, 
                               scan_axis, 
                               sdd, 
                               fov, 
                               collimation, 
                               weight, 
                               specter_cpd, 
                               specter_energies)
    t0 = time.clock()
    engine.run(beam2, n_histories, simulation)
    print('Simulated another {0} photons in {1} seconds'.format(n_histories, time.clock()-t0))                           

    #cleanup of simulation and sources, the monte carlo engine will leak
    # memory if these functions are not called
    engine.cleanup(simulation=simulation, energy_imparted=energy_imparted)
    engine.cleanup(source=beam)
    engine.cleanup(source=beam2)



    plt.subplot(2, 2, 2)
    plt.imshow(energy_imparted[:, :, N[2] // 2])
    plt.title('Energy Imparted [eV]')
    plt.subplot(2, 2, 3)
    plt.imshow(material_indices[:, :, N[2] // 2], cmap='gray')
    plt.title('Material index')
    plt.subplot(2, 2, 4)
    dose = energy_imparted / (np.prod(spacing) * densities)
    plt.imshow(np.log(dose[:, :, N[2] // 2]))
    plt.title('Logarithm of dose [eV / grams]')
    plt.show()    
    
    
    
    
if __name__ == '__main__':
#    bowtie_example()
#    preformance_example()
    get_performance()
#    cone_beam_example()
    