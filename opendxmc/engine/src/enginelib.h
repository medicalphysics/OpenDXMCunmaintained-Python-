#ifndef ENGINELIB
#define ENGINELIB

#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
//#define USING_FLOAT

#ifndef USING_FLOAT
#define FLOAT double
#define FMAX fmax
#define FMIN fmin
#define SQRT sqrt
#define FABS fabs
#define EXP exp
#define LOG log
#define SIN sin
#define COS cos
#define TAN tan
#define ACOS acos
#define ASIN asin
#define POW pow
#define CEIL ceil
#else
#define FLOAT float
#define FMAX fmaxf
#define FMIN fminf
#define SQRT sqrtf
#define FABS fabsf
#define EXP expf
#define LOG logf
#define SIN sinf
#define COS cosf
#define TAN tanf
#define ACOS acosf
#define ASIN asinf
#define POW powf
#define CEIL ceilf
#endif


#ifdef _MSC_VER
#define INLINE __forceinline
#else
#define INLINE inline
#endif

#ifdef __linux__
#define EXTERN extern
#else
#define EXTERN __declspec(dllexport)
#endif

// Some constants
const FLOAT ERRF = 1e-6; // Precision error
const FLOAT ELECTRON_MASS = 510998.9;  //  eV/(c*c)
const FLOAT PI = 3.14159265359;
const FLOAT ENERGY_CUTOFF = 1000; // eV
const FLOAT WEIGHT_CUTOFF = 0.01;
const FLOAT RUSSIAN_RULETTE_CHANCE = .2; //CHANCE probability of photon survival



#ifdef __cplusplus
extern "C" {
#endif

	typedef struct
	{
		int *shape;
		FLOAT *spacing;
		FLOAT *offset;
		int *material_map;
		FLOAT *density_map;
		int *lut_shape;
		FLOAT *attenuation_lut;
		FLOAT *energy_imparted;
		FLOAT *max_density;
		uint64_t *seed;
		int *use_siddon_pathing;
	}Simulation;

	typedef struct
	{
		FLOAT *source_position;
		FLOAT *source_direction;
		FLOAT *scan_axis;
		FLOAT *sdd;
		FLOAT *fov;
		FLOAT *collimation;
		FLOAT *weight;
		int *specter_elements;
		FLOAT *specter_cpd;
		FLOAT *specter_energy;
	}Source;

	typedef struct
	{
		FLOAT *source_position;
		FLOAT *source_direction;
		FLOAT *scan_axis;
		FLOAT *scan_axis_fan_angle;
		FLOAT *rot_axis_fan_angle;
		FLOAT *weight;
		int *specter_elements;
		FLOAT *specter_cpd;
		FLOAT *specter_energy;
		int *bowtie_elements;
		FLOAT *bowtie_weight;
		FLOAT *bowtie_angle;
	}SourceBowtie;

	typedef bool(*trackingFuncPtr)(size_t *, FLOAT *, int *, FLOAT *, FLOAT *, int *, FLOAT *, int *, FLOAT *, FLOAT *, uint64_t *);

	EXTERN void* setup_simulation(int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *lut_shape, FLOAT *lut, FLOAT *energy_imparted, int *use_siddon);

	EXTERN void* setup_source(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *sdd, FLOAT *fov, FLOAT *collimation, FLOAT *weight, FLOAT *specter_cpd, FLOAT *specter_energy, int *specter_elements);

	EXTERN void* setup_source_bowtie(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *scan_axis_fan_angle, FLOAT *rot_axis_fan_angle, FLOAT *weight, FLOAT *specter_cpd, FLOAT *specter_energy, int *specter_elements, FLOAT* bowtie_weight, FLOAT* bowtie_angle, int *bowtie_elements);

	EXTERN void run_simulation(void *source, int64_t n_particles, void *simulation);

	EXTERN void run_simulation_bowtie(void *dev_source, int64_t n_particles, void *dev_simulation);

	EXTERN void cleanup_simulation(void *simulation);

	EXTERN void cleanup_source(void *source);
#ifdef __cplusplus
}
#endif

#endif 