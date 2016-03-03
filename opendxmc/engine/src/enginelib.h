#ifndef ENGINELIB
#define ENGINELIB

//#define USINGCUDA
#ifndef USINGCUDA
#include <omp.h>
#include <stdbool.h>
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif


#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
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

	typedef struct
	{
		double *source_position;
		double *source_direction;
		double *scan_axis;
		double *scan_angle;
		double *rot_angle;
		double *weight;
		int *specter_elements;
		double *specter_cpd;
		double *specter_energy;
		int *bowtie_elements;
		double *bowtie_weight;
		double *bowtie_angle;
	}SourceBowtie;

	__declspec(dllexport) int number_of_cuda_devices();

	__declspec(dllexport) void cuda_device_name(int device_number, char* name);

	__declspec(dllexport) void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *lut, double *energy_imparted);

	__declspec(dllexport) void* setup_source(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, double *specter_cpd, double *specter_energy, int *specter_elements);

	__declspec(dllexport) void* setup_source_bowtie(double *source_position, double *source_direction, double *scan_axis, double *scan_angle, double *rot_angle, double *weight, double *specter_cpd, double *specter_energy, int *specter_elements, double* bowtie_weight, double* bowtie_angle, int* bowtie_elements);

	__declspec(dllexport) void run_simulation(void *source, size_t n_particles, void *simulation);

	__declspec(dllexport) void run_simulation_bowtie(void *dev_source, size_t n_particles, void *dev_simulation);

	__declspec(dllexport) void cleanup_simulation(void *simulation, int *shape, double *energy_imparted);

	__declspec(dllexport) void cleanup_source(void *source);
#ifdef __cplusplus
}
#endif
#endif 