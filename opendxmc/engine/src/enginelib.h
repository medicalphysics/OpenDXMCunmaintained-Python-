#ifndef ENGINELIB
#define ENGINELIB

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

	typedef struct
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

	typedef struct
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

	__declspec(dllexport) int number_of_cuda_devices();

	__declspec(dllexport) void cuda_device_name(int device_number, char* name);

	__declspec(dllexport) void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *lut, double *energy_imparted);

	__declspec(dllexport) void* setup_source(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, double *specter_cpd, double *specter_energy, int *specter_elements);

	__declspec(dllexport) void run_simulation(void *source, size_t n_particles, void *simulation);

	__declspec(dllexport) void cleanup_simulation(void *simulation, int *shape, double *energy_imparted);

	__declspec(dllexport) void cleanup_source(void *source);
#ifdef __cplusplus
}
#endif
#endif 