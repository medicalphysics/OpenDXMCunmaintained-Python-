# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:05:21 2016

@author: ander
"""

import numpy as np

from opendxmc.engine import Engine
from opendxmc.database import get_stored_materials
from opendxmc.runner import generate_attinuation_lut
import time
from matplotlib import pylab as plt

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
    N = np.array([128, 128, 128], dtype='int32')    
    # Then spacing of each voxel in cm
    spacing = np.array([0.1, 0.1, 0.1], dtype='float64')    
    
    # lets specify a lookup table as a dictionary for the materials we are 
    # using. The key in the dictionary corresponds to the values in the 
    # material_indices array
    material_lut = {0: 'air', 1: 'water'}
    material_indices = np.zeros(N, dtype='int32')
    #Lets fill a region of the box with water
    material_indices[20:-20, 20:-20, 20:-20] = 1  
    
    # Now we create a density array as same shape as the material_indices array
    # We could spesify different densities for each voxel, but we are just 
    # going to give each material its room temperature default density
    
    air_material = materials_dict['air']
    water_material = materials_dict['water']
    
    densities = np.empty(N, dtype='float64')
    densities[:, :, :] = air_material.density
    densities[20:-20, 20:-20, 20:-20] = water_material.density
    
    # Next we need to get the attinuation lookup table for the specified 
    # materials. This is a numpy float64 array with shape: 
    # [number_of_materials, 5, number_of_energies_we_have_interaction_constants_at]
    # The generate_attinuation_lut function is a conveniance function that will 
    # generate this LUT
    lut = generate_attinuation_lut([air_material, water_material], material_lut)
    
    #Now from the lut we can plot the attenuation coefficients for water:
    plt.subplot(2, 2, 1)
    plt.loglog(lut[1, 0, :], lut[1, 1, :], label='Total')
    plt.loglog(lut[1, 0, :], lut[1, 2, :], label='Rayleigh scattering')
    plt.loglog(lut[1, 0, :], lut[1, 3, :], label='Photoelectric effect')
    plt.loglog(lut[1, 0, :], lut[1, 4, :], label='Compton scattering')
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
    lut_shape = np.array(lut.shape, dtype='int32')
    # and an array to store imparted energy
    energy_imparted = np.zeros_like(densities)
    simulation = engine.setup_simulation(N, 
                                         spacing, 
                                         offset, 
                                         material_indices, 
                                         densities, 
                                         lut_shape, 
                                         lut,
                                         energy_imparted,
                                         use_siddon=np.zeros(1, dtype='int'))
    
    
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
    n_specter=np.array(specter_cpd.shape, dtype='int32')
    beam = engine.setup_source(source_position,
                               source_direction, 
                               scan_axis, 
                               sdd, 
                               fov, 
                               collimation, 
                               weight, 
                               specter_cpd, 
                               specter_energies,
                               n_specter)
                               
    # Running the simulation
    n_histories = 1000000
    t0 = time.clock()
    import pdb;pdb.set_trace()
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
                               specter_energies,
                               n_specter)
    
    
    engine.run(beam2, n_histories, simulation)
    print('Simulated another {0} photons in {1} seconds'.format(n_histories, time.clock()-t0))                           

    #cleanup of simulation and sources, the monte carlo engine will leak
    # memory if these functions are not called
    engine.cleanup(simulation=simulation)
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
    cone_beam_example()
    