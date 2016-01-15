#ifndef ENGINELIB
#define ENGINELIB

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern "C" typedef struct
{
	int *shape;
	double *spacing;
	double *offset;
	int *material_map;
	double *density_map;
	int *lut_shape;
	double *attenuation_lut;
	double *energy_imparted;
	double *max_density;
	uint64_t *seed;
}Simulation;

extern "C" typedef struct
{
	double *source_position;
	double *source_direction;
	double *scan_axis;
	double *sdd;
	double *fov;
	double *collimation;
	double *weight;
	int *specter_elements;
	double *specter_cpd;
	double *specter_energy;
}Source;

extern "C" int number_of_cuda_devices();

extern "C" void cuda_device_name(int device_number, char* name);

extern "C" void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *lut, double *energy_imparted);

extern "C" void* setup_source(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, double *specter_cpd, double *specter_energy, int *specter_elements);

extern "C" void run_simulation(void *source, size_t n_particles, void *simulation);

extern "C" void cleanup_simulation(void *simulation, int *shape, double *energy_imparted);

extern "C" void cleanup_source(void *source);
#endif 