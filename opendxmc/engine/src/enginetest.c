// enginetest.cpp : Defines the entry point for the console application.
//
/*
#include <stdlib.h>
#include "enginelib.h"
#include <math.h>



void setup_test_environment(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, double *energy_imparted)
{
	int i, j, k;
	size_t ind;

	//geometry
	for (i = 0; i < 3; i++)
	{
		offset[i] = -shape[i] * spacing[i] / 2.;
	}

	//particles
	
	for (i = 0; i < shape[0];i++)
		for (j = 0; j < shape[1]; j++)
			for (k = 0; k < shape[2]; k++)
			{
				ind = i * shape[2] * shape[1] + j * shape[2] + k;
				material_map[ind] = 0;
				density_map[ind] = 1;
				energy_imparted[ind] = 0;
			}

	//lut
	for (int i = 0; i < 2; i++)
	{
		//energy 
		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 0] = 1000;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 0] = 0.34;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 0] = 0.05;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 0] = 6.8;

		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 1] = 10000;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 1] = 0.0246;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 1] = 0.358;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 1] = 0.00277;

		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 2] = 50000;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 2] = 0.00101;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 2] = 0.3344;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 2] = 0.000011;

		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 3] = 69000;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 3] = 0.00005;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 3] = 0.317;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 3] = 0.000003;

		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 4] = 100000;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 4] = 0.000276;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 4] = 0.29;
		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 4] = 0.000000987;

		for (int k = 0; k < 5; k++)
		{
			ind = i*att_shape[1] * att_shape[2] + 1 * att_shape[2] + k;
			attenuation_lut[ind] = 0;
		}

		for (int j = 2; j < 5; j++)
		{
			for (int k = 0; k < 5; k++)
			{
				ind = i*att_shape[1] * att_shape[2] + j * att_shape[2] + k;
				attenuation_lut[i*att_shape[1] * att_shape[2] + 1 * att_shape[2] + k] += attenuation_lut[ind];
			}
		}
	}
}


int main()
{

	int number_of_devices; 

	size_t n_particles = 512*5;
	int shape[3] = {64, 64, 64 };
	int lut_shape[3] = { 2, 5, 5 };
	double spacing[3] = {1, 1, 1};
	double offset[3];
	
	// init geometry variables
	//double *particles = (double *)malloc(n_particles * 8 * sizeof(double));
	int *material_map = (int *)malloc(shape[0] * shape[1] * shape[2] * sizeof(int));
	double *density_map = (double *)malloc(shape[0] * shape[1] * shape[2] * sizeof(double));
	double *attenuation_lut = (double *)malloc(lut_shape[0] * lut_shape[1] * lut_shape[2] * sizeof(double));
	double *energy_imparted = (double *)malloc(shape[0] * shape[1] * shape[2] * sizeof(double));
	
	// initialazing geometry
	setup_test_environment(shape, spacing, offset, material_map, density_map, lut_shape, attenuation_lut, energy_imparted);
	

	void* sim;
	sim = setup_simulation(shape, spacing, offset, material_map, density_map, lut_shape, attenuation_lut, energy_imparted);

	//init source variables
	double source_position[3] = {-100, 0, 0};
	double source_direction[3] = { 1, 0, 0 };
	double scan_axis[3] = {0, 0, 1 };
	double sdd = 119;
	double fov = 50;
	double collimation = 4;
	double weight = 1;

	double specter_cpd[3] = { 0.33, 0.66, 1};
	double specter_energy[3] = { 60000, 70000, 80000};
	int specter_elements = 3;

	void* geo;
	geo = setup_source(source_position, source_direction, scan_axis, &sdd, &fov, &collimation, &weight, specter_cpd, specter_energy, &specter_elements);


	run_simulation(geo, n_particles, sim);
	
	

	cleanup_simulation(sim, shape, energy_imparted);
	cleanup_source(geo);

	
	
	size_t index;
	double energy1234 = 0;
	for (int i = 0; i < shape[0]; i++)
	{
		for (int j = 0; j < shape[1]; j++)
		{
			for (int k = 0; k < shape[2]; k++)
			{
				
				index = shape[1] * shape[2] * i + shape[2] * j + k;
				if (energy_imparted[index] > 0.000000000001)
				{
					energy1234 = energy_imparted[index];
					printf("\n energy in %d %d %d = %f", i, j, k, energy_imparted[index]);
				}
			}
		}
	}
	
}
*/

