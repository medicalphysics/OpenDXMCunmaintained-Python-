
//#define USINGCUDA
#ifndef USINGCUDA
//#include <omp.h>
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif
#include "math.h"
//#include "curand.h"
//#include "curand_kernel.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>


// CUDA Constants
#ifdef USINGCUDA
__device__ __constant__ double ERRF = 0e-9; // Precision error
__device__ __constant__ double ERRG = 0e-3; // Geometric error
__device__ __constant__ double ELECTRON_MASS = 510998.9;  //  eV/(c*c)
__device__ __constant__ double PI = 3.14159265359;
__device__ __constant__ double ENERGY_CUTOFF = 1000; // eV
__device__ __constant__ double ENERGY_MAXVAL = 300000; // eV
__device__ __constant__ double WEIGHT_CUTOFF = 0.01;
__device__ __constant__ double RUSSIAN_RULETTE_CHANCE = 2; // 1 / CHANCE probability of photon survival
#else
const double ERRF = 0e-9; // Precision error
const double ERRG = 0e-3; // Geometric error
const double ELECTRON_MASS = 510998.9;  //  eV/(c*c)
const double PI = 3.14159265359;
const double ENERGY_CUTOFF = 1000; // eV
const double ENERGY_MAXVAL = 300000; // eV
const double WEIGHT_CUTOFF = 0.01;
const double RUSSIAN_RULETTE_CHANCE = 2; // 1 / CHANCE probability of photon survival
#endif


#ifdef USINGCUDA
__device__
#endif
uint64_t xorshift128plus(uint64_t *s) {
	// init seed must not be zero
	uint64_t x = s[0];
	uint64_t const y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1] + y;
}

#ifdef USINGCUDA
__device__
#endif
double randomduniform(uint64_t *seed)
{ 
	//uint64_t k = xorshift128plus(seed);
	//double a = (double)xorshift128plus(seed) / (double)UINT64_MAX;
	return (double)xorshift128plus(seed) / (double)UINT64_MAX;
}


#ifdef USINGCUDA
__device__ double atomicAdd(double *address, double val)
/* Atomic add of double to array. Returns old value. */
{
	unsigned long long int *address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

	} while (assumed != old);
	return __longlong_as_double(old);
}
#else
double atomicAdd(double *address, double val)
{
	#pragma omp atomic
		address[0] += val;
	return val;
}
#endif

#ifdef USINGCUDA
__device__ 
#endif
bool particle_on_plane(double *particle, int *shape, double *spacing, double *offset, size_t plane_dimension)
/* Bondary test if the particle is resting on plane p laying on one of the edges of the scoring volume. Returns true if the point on the plane is on the edge of the volume*/
{
	double llim, ulim;
	for (size_t i = 0; i < 3; i++)
	{
		if (i != plane_dimension)
		{
			ulim = offset[i] + shape[i] * spacing[i] + ERRF;
			llim = offset[i] - ERRF;
			if ((particle[i] < llim) || (particle[i] > ulim)) { return false; }
		}
	}
	return true;
}

#ifdef USINGCUDA
__device__
#endif
bool particle_inside_volume(double *particle, int *shape, double *spacing, double *offset)
/* Test for particle inside volume. If inside returns true*/
{
	double llim, ulim;
	for (size_t i = 0; i < 3; i++)
	{
		ulim = offset[i] + shape[i] * spacing[i] + ERRF;
		llim = offset[i] - ERRF;
		if ((particle[i] < llim) || (particle[i] > ulim)) { return false; }
	}
	return true;
}

#ifdef USINGCUDA
__device__
#endif
bool particle_is_intersecting_volume(double *particle, int *shape, double *spacing, double *offset)
/*Tests if particle intersects with dose scoring volume. If intersecting and outside scoring volume
the particle is transported along its direction to the volume edge. Returns true if the particle intersects scoring volume,
returns false if the particle misses scoring volume.*/
{
	if (particle_inside_volume(particle, shape, spacing, offset))
	{
		return true;
	}
	size_t i, j;
	double t[2], t_cand;
	double pos[3];
	int plane_intersection = -1;
	for (i = 0; i < 3; i++)
	{
		if (fabs(particle[i + 3]) > ERRF)
		{
			// lowest planes
			t_cand = (offset[i] - particle[i]) / particle[i + 3];
			if (t_cand >= 0)
			{
				for (j = 0; j < 3; j++)
				{
					pos[j] = particle[j] + t_cand * particle[j + 3];
				}

				if (particle_on_plane(pos, shape, spacing, offset, i))
				{
					plane_intersection++;
					t[plane_intersection] = t_cand;
				}
			}
			// highest planes
			t_cand = (offset[i] + spacing[i] * shape[i] - particle[i]) / particle[i + 3];
			if (t_cand >= 0)
			{
				for (j = 0; j < 3; j++)
				{
					pos[j] = particle[j] + t_cand * particle[j + 3];
				}
				if (particle_on_plane(pos, shape, spacing, offset, i))
				{
					plane_intersection++;
					t[plane_intersection] = t_cand;
				}
			}
		}
		// we break if we find two intersections
		if (plane_intersection >= 1)
		{
			break;
		}
	}
	// return if we dont find any intersections
	if (plane_intersection == -1)
	{
		return false;
	}

	//finding closest plane
	t_cand = fmin(t[0], t[1]);

	// moving particle to closest plane
	for (i = 0; i < 3; i++)
	{
		particle[i] = particle[i] + particle[i + 3] * t_cand;
	}
	return true;
}

#ifdef USINGCUDA
__device__
#endif
double interp(double x, double x1, double x2, double y1, double y2)
{
	return y1 + (y2 - y1) * ((x - x1) / (x2 - x1));
}

#ifdef USINGCUDA
__device__
#endif
double lut_interpolator(int material, int interaction, double energy, int *lut_shape, double *lut, size_t *lower_index)
{
	// binary search for indices in the lut closest to the requested value , the lower index pointer vil 
	int first = 0;
	int last = lut_shape[2] - 1;
	int middle = (first + last) / 2;
	size_t ind = material * lut_shape[1] * lut_shape[2];

	while (first <= last) {
		if (lut[ ind + middle + 1] < energy)
			first = middle + 1;
		else if ((lut[ind + middle] <= energy) && (lut[ind + middle + 1] > energy)) {
			
			break;
		}
		else
			last = middle - 1;
		middle = (first + last) / 2;
	}
	if (first > last)
	{
		// if we dont find the element we return the extreme values at the appropriate side of the table 
		
		if (energy <= lut[ind])
		{
			lower_index[0] = 0;
			return lut[ind + interaction * lut_shape[2]];
		}
		else
		{
			lower_index[0] = lut_shape[2] - 1;
			return lut[ind + interaction * lut_shape[2] + lut_shape[2] - 1];
		}
	}
	// end binary search
	lower_index[0] = ind + middle;
	return interp(energy, lut[lower_index[0]], lut[lower_index[0] + 1], lut[lower_index[0] + lut_shape[2] * interaction], lut[lower_index[0] + lut_shape[2] * interaction + 1]);
}

#ifdef USINGCUDA
__device__
#endif
size_t particle_array_index(double *particle, int *shape, double *spacing, double *offset)
{ /*Returns the voxel index for particle in the volume arrays */
	size_t i = (size_t)((particle[0] - offset[0]) / spacing[0]);
	size_t j = (size_t)((particle[1] - offset[1]) / spacing[1]);
	size_t k = (size_t)((particle[2] - offset[2]) / spacing[2]);
	if (i >= shape[0])
		i = shape[0] - 1;
	if (j >= shape[1])
		j = shape[1] - 1;
	if (k >= shape[2])
		k = shape[2] - 1;

	/*printf("\nArray index: %d, %d, %d,  %d,    Coordiantes: %f, %f, %f", i, j, k, i * shape[2] * shape[1] + j * shape[2] + k, (particle[0] - offset[0]) / spacing[0], (particle[1] - offset[1]) / spacing[1], (particle[2] - offset[2]) / spacing[2]);
	printf("\nParticle %f, %f, %f, %f, %f, %f", particle[0], particle[1], particle[2], particle[3], particle[4], particle[5]);
	printf("\nshape %d, %d, %d", shape[0], shape[1], shape[2]);
	printf("\nspacing %f, %f, %f", spacing[0], spacing[1], spacing[2]);
	printf("\noffset %f, %f, %f", offset[0], offset[1], offset[2]);*/
	return i * shape[2] * shape[1] + j * shape[2] + k;

}

#ifdef USINGCUDA
__device__
#endif
bool woodcock_step(size_t *volume_index, double *particle, int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, uint64_t *state)
{ /*Make the particle take a woodcock step until an interaction occurs or the particle steps out of volume, returns true if an interaction occurs, then volume index contains the voxel_index for the interaction*/

	bool interaction = false;
	bool valid = particle_is_intersecting_volume(particle, shape, spacing, offset);
	//printf("\nValid: %d", valid);
	
	double smin, scur, step, smin_dens;
	size_t lut_index, i, index;
	smin = 0;
	for (i = 0; i < att_shape[0]; i++)
	{
		smin = fmax(smin, lut_interpolator(i, 1, particle[6], att_shape, attenuation_lut, &lut_index));
	}

	while (valid && !interaction)
	{
		index = particle_array_index(particle, shape, spacing, offset);
		smin_dens = smin * density_map[index]; // correcting for density
		//smin *= density_map[particle_array_index(particle, shape, spacing, offset)]; // correcting for density

		// samplinf distance
		step = -log(randomduniform(&state[0])) / smin_dens;///////// step from smin 
		//printf("step size %f\n", step);
		//moving particle a random 1/smin step
		for (i = 0; i < 3; i++)
		{
			particle[i] += particle[i + 3] * step;
		}
		// updating valid
		valid = particle_inside_volume(particle, shape, spacing, offset); // skips intersection test

		if (valid)
		{
			volume_index[0] = particle_array_index(particle, shape, spacing, offset);
			scur = lut_interpolator(material_map[volume_index[0]], 1, particle[6], att_shape, attenuation_lut, &lut_index) * density_map[volume_index[0]]; // basicly total attenuation(E) * density
			if ((scur / smin_dens) > randomduniform(&state[0]))
				/*debug[0] = curand_uniform(&state[0]);
				debug[1] = (exp(-step*scur));
				interaction = debug[0] >= debug[1];*/
				interaction = randomduniform(&state[0]) >= (exp(-step*scur));
		}
	}
	return interaction;
}


#ifdef USINGCUDA
__device__
#endif
void normalize_3Dvector(double *vector)
{
	double vector_sum = sqrt((vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]));
	vector[0] /= vector_sum;
	vector[1] /= vector_sum;
	vector[2] /= vector_sum;
}

#ifdef USINGCUDA
__device__
#endif
void rotate_3Dvector(double *vector, double *axis, double degrees)
// angle must be inside (0, PI]
{
	double out[3];
	double sang = sin(degrees);
	double cang = cos(degrees);
	double midterm = (1. - cang) * (vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2]);

	out[0] = cang * vector[0] + midterm * axis[0] + sang * (axis[1] * vector[2] - axis[2] * vector[1]);
	out[1] = cang * vector[1] + midterm * axis[1] + sang * (-axis[0] * vector[2] + axis[2] * vector[0]);
	out[2] = cang * vector[2] + midterm * axis[2] + sang * (axis[0] * vector[1] - axis[1] * vector[0]);
	vector[0] = out[0];
	vector[1] = out[1];
	vector[2] = out[2];
}

#ifdef USINGCUDA
__device__
#endif
void rotate_particle(double *particle, double theta, double phi)
// rotates a particle theta degrees from its current direction theta degrees about a random axis orthogonal to the direction vector (this axis is rotated phi degrees about the particles direction vector)
{
	// First we find a vector orthogonal to the particle direction
	// k_xy = v x k   where k = ({1, 0, 0} , {0, 1, 0}, {0, 0, 1}) depending on the smallest magnitude of the particles direction vector (we do this to make the calculation more robust)
	double k_xy[3];
	if ((fabs(particle[3]) < fabs(particle[4])) && (fabs(particle[3]) < fabs(particle[5])))
	{
		k_xy[0] = 0;
		k_xy[1] = particle[5];
		k_xy[2] = -particle[4];
	}
	else if ((fabs(particle[4]) < fabs(particle[3])) && (fabs(particle[4]) < fabs(particle[5])))
	{
		k_xy[0] = -particle[5];
		k_xy[1] = 0;
		k_xy[2] = particle[3];
	}
	else
	{
		k_xy[0] = particle[4];
		k_xy[1] = -particle[3];
		k_xy[2] = 0;
	}
	normalize_3Dvector(k_xy); // assure the vector is a unit vector
	if (phi < 0) // assure we can rotate negative degrees
	{
		phi = -phi;
		k_xy[0] = -k_xy[0];
		k_xy[1] = -k_xy[1];
		k_xy[2] = -k_xy[2];
	}
	rotate_3Dvector(k_xy, &particle[3], phi);
	rotate_3Dvector(&particle[3], k_xy, theta);
}

#ifdef USINGCUDA
__device__
#endif
void rayleigh_event_draw_theta(double *angle, uint64_t *state)
{
	double r, c, A;
	r = randomduniform(&state[0]);
	c = 4. - 8. * r;
	if (c > 0)
		A = pow((fabs(c) + sqrt(c * c + 4.)) / 2., (1. / 3.));
	else
		A = -pow((fabs(c) + sqrt(c * c + 4.)) / 2., (1. / 3.));
	angle[0] = acos(A - 1. / A);
}

#ifdef USINGCUDA
__device__
#endif
double compton_event_draw_energy_theta(double energy, double* theta, uint64_t *state)
{
	//Draws scattered energy and angle, based on Geant4 implementation, returns scattered energy and sets theta to scatter angle
	double epsilon_0, alpha1, alpha2, r1, r2, r3, epsilon, qsin_theta, t, k;
	k = energy / ELECTRON_MASS;
	epsilon_0 = 1. / (1. + 2. * k);
	alpha1 = log(1. / epsilon_0);
	alpha2 = (1. - epsilon_0 * epsilon_0) / 2.;
	while (true)
	{
		r1 = randomduniform(&state[0]);
		r2 = randomduniform(&state[0]);
		r3 = randomduniform(&state[0]);

		if (r1 < alpha1 / (alpha1 + alpha2))
		{
			epsilon = exp(-r2 * alpha1);
		}
		else
		{
			epsilon = sqrt((epsilon_0 * epsilon_0 + (1. - epsilon_0 * epsilon_0) * r2));
		}

		t = ELECTRON_MASS * (1. - epsilon) / (energy * epsilon);
		qsin_theta = t * (2. - t);

		if ((1. - epsilon / (1. + epsilon * epsilon) * qsin_theta) >= r3)
			break;
	}
	//theta[0] = acos(1. + 1. / k - 1. / epsilon / k);
	theta[0] = acos(1. + (1. - 1. / epsilon) / k);
	return epsilon * energy;
}

#ifdef USINGCUDA
__global__
#endif
void transport_particles(double *particles, size_t *n_particles, int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, double *energy_imparted, uint64_t *states)
{
	#ifdef USINGCUDA
		size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	#else
		size_t id = n_particles[0];
	#endif
	
	double particle[8];
	double rayleight, photoelectric,  r_interaction, scatter_angle, scatter_energy;
	size_t volume_index, lut_index;

	#ifdef USINGCUDA
	if (id >= n_particles[0])
	{
		return;
	}
	#endif

	for (size_t j = 0; j < 8; j++)
	{
		particle[j] = particles[id * 8 + j];
	}
	
	// preforming woodcock until interaction
	while (woodcock_step(&volume_index, particle, shape, spacing, offset, material_map, density_map, att_shape, attenuation_lut, &states[id * 2]))
	{
		rayleight = lut_interpolator(material_map[volume_index], 2, particle[6], att_shape, attenuation_lut, &lut_index);
		photoelectric = interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2] * 3], attenuation_lut[lut_index + att_shape[2] * 3 + 1]);

		//r_interaction = curand_uniform(&states[id]) * interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2]], attenuation_lut[lut_index + att_shape[2] + 1]);
		r_interaction = randomduniform(&states[id * 2]) * interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2]], attenuation_lut[lut_index + att_shape[2] + 1]);

		if (rayleight > r_interaction) //rayleigh scatter event
		{
			rayleigh_event_draw_theta(&scatter_angle, &states[id * 2]);
			//rotate_particle(particle, scatter_angle, (curand_uniform(&states[id]) * 2. - 1. ) * PI);
			rotate_particle(particle, scatter_angle, (randomduniform(&states[id * 2]) * 2. - 1.) * PI);
		}
		else if ((rayleight + photoelectric) > r_interaction) //photoelectric event
		{ 
			atomicAdd(&energy_imparted[volume_index], particle[6] * particle[7]);
			break;
		}
		else // compton event
		{
			scatter_energy = compton_event_draw_energy_theta(particle[6], &scatter_angle, &states[id * 2]);
			rotate_particle(particle, scatter_angle, (randomduniform(&states[id * 2]) * 2. - 1.) * PI);
			atomicAdd(&energy_imparted[volume_index], (particle[6] - scatter_energy) * particle[7]);
			particle[6] = scatter_energy;
		}

		//test for energy cutoff and weight cutoff

		if (particle[6] < ENERGY_CUTOFF)
		{
			atomicAdd(&energy_imparted[volume_index], particle[6] * particle[7]);
			break;
		}
		if (particle[7] < WEIGHT_CUTOFF)
		{
			break;
		}
	
	}
}


#ifdef USINGCUDA
__global__ void init_random_seed(uint64_t seed, size_t *n_particles, uint64_t *states)
{
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < n_particles[0])
	{
		//states[id * 2] = seed;
		//states[id * 2 + 1] = id + 1;
		uint64_t m = UINT64_MAX / (n_particles[0] + 1);
		uint64_t s[2];
		s[0] = seed;
		s[1] = (id + 1) * m;
		states[id * 2] = xorshift128plus(s);
		states[id * 2 + 1] = xorshift128plus(s);
	}
}
#else
void init_random_seed(uint64_t seed, size_t *n_particles, uint64_t *states)
{
	uint64_t m = UINT64_MAX / (n_particles[0] + 1);
	#pragma omp parallel for
	for (long long int id = 0; id < n_particles[0]; id++)
	{
		uint64_t s[2];
		s[0] = seed;
		s[1] = (id + 1) * m;
		states[id * 2] = xorshift128plus(s);
		states[id * 2 + 1] = xorshift128plus(s);
	}
}
#endif


extern "C"
{
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
	}Simulation;

	__declspec(dllexport)
	#ifdef USINGCUDA
	__host__ int number_of_cuda_devices()
	{
		int err;
		int device_number;
		err = cudaGetDeviceCount(&device_number);
		if (err != 0){ return 0;}
		return device_number;
	}
	#else
		int number_of_cuda_devices(){return -1;}
	#endif
	
	__declspec(dllexport)
	#ifdef USINGCUDA
	__host__ void cuda_device_name(int device_number, char* name )
	{
		int err;
		err = cudaGetDeviceCount(&device_number);
		if (err != 0){return;}
		struct cudaDeviceProp props;
		err = cudaGetDeviceProperties(&props, device_number);
		if (err != 0){ return; }
		name = props.name;
	}
	#else
		void cuda_device_name(int device_number, char* name ){return;}
	#endif

	#ifdef USINGCUDA
	__declspec(dllexport) __host__ void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *attenuation_lut, double *energy_imparted)
	{

		// device declarations
		int *shape_dev;
		double *spacing_dev;
		double *offset_dev;
		int *material_map_dev;
		double *density_map_dev;
		int *lut_shape_dev;
		double *lut_dev;
		double *energy_dev;

		size_t array_size = shape[0] * shape[1] * shape[2];
		size_t lut_size = lut_shape[0] * lut_shape[1] * lut_shape[2];

		// device allocations
		cudaMalloc(&shape_dev, 3 * sizeof(int));
		cudaMalloc(&spacing_dev, 3 * sizeof(double));
		cudaMalloc(&offset_dev, 3 * sizeof(double));
		cudaMalloc(&material_map_dev, array_size * sizeof(int));
		cudaMalloc(&density_map_dev, array_size * sizeof(double));
		cudaMalloc(&lut_shape_dev, 3 * sizeof(int));
		cudaMalloc(&lut_dev, lut_size * sizeof(double));
		cudaMalloc(&energy_dev, array_size * sizeof(double));

		// check for error
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}



		// memory transfer to device
		cudaMemcpy(shape_dev, shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(spacing_dev, spacing, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(offset_dev, offset, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(material_map_dev, material_map, array_size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(density_map_dev, density_map, array_size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(lut_shape_dev, lut_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(lut_dev, attenuation_lut, lut_size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(energy_dev, energy_imparted, array_size * sizeof(double), cudaMemcpyHostToDevice);

		// creating simulation structure
		Simulation *sim_dev = (Simulation*)malloc(sizeof(Simulation));
		sim_dev->shape = shape_dev;
		sim_dev->spacing = spacing_dev;
		sim_dev->offset = offset_dev;
		sim_dev->material_map = material_map_dev;
		sim_dev->density_map = density_map_dev;
		sim_dev->lut_shape = lut_shape_dev;
		sim_dev->attenuation_lut = lut_dev;
		sim_dev->energy_imparted = energy_dev;

		return (void*)sim_dev;
	}
	#else
	__declspec(dllexport) void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *attenuation_lut, double *energy_imparted)
	{
		Simulation *sim_dev = (Simulation*)malloc(sizeof(Simulation));
		sim_dev->shape = shape;
		sim_dev->spacing = spacing;
		sim_dev->offset = offset;
		sim_dev->material_map = material_map;
		sim_dev->density_map = density_map;
		sim_dev->lut_shape = lut_shape;
		sim_dev->attenuation_lut = attenuation_lut;
		sim_dev->energy_imparted = energy_imparted;
		return (void*)sim_dev;
	}
	#endif
	
	#ifdef USINGCUDA
	__declspec(dllexport) __host__ void run_simulation(double *particles, size_t n_particles, void *dev_simulation)
	{
		dim3 blocks((int)(n_particles / 512 + 1), 1, 1);
		dim3 threads(512, 1, 1);
		cudaError_t error;

		// initialize device pointers
		double *dev_particles;
		size_t *dev_n_particles;
		uint64_t *dev_states;

		//allocate memory on device
		cudaMalloc(&dev_particles, n_particles * 8 * sizeof(double));
		cudaMalloc(&dev_n_particles, sizeof(size_t));
		cudaMalloc(&dev_states, n_particles * 2 * sizeof(uint64_t));
		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		// transfer data to device
		cudaMemcpy(dev_particles, particles, n_particles * 8 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_n_particles, &n_particles, sizeof(size_t), cudaMemcpyHostToDevice);

		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		// setting up random number generators 
		init_random_seed <<< blocks, threads >>>(time(NULL), dev_n_particles, dev_states);	
		//init_random_seed <<< blocks, threads >>>(1337, dev_n_particles, dev_states);	
		cudaDeviceSynchronize();
		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error init random seed: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
		// simulating particles

		transport_particles <<< blocks, threads >>>
			(
			dev_particles,
			dev_n_particles,
			((Simulation*)dev_simulation)->shape,
			((Simulation*)dev_simulation)->spacing,
			((Simulation*)dev_simulation)->offset,
			((Simulation*)dev_simulation)->material_map,
			((Simulation*)dev_simulation)->density_map,
			((Simulation*)dev_simulation)->lut_shape,
			((Simulation*)dev_simulation)->attenuation_lut,
		    ((Simulation*)dev_simulation)->energy_imparted,
			dev_states);
		
		cudaDeviceSynchronize();
		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
	    // free device memory
		cudaFree(dev_particles);
		cudaFree(dev_n_particles);
		cudaFree(dev_states);
		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
		return;
	}
	#else
	__declspec(dllexport) void run_simulation(double *particles, size_t n_particles, void *dev_simulation)
	{		
		// initialize random states
		uint64_t *states = (uint64_t*)malloc(n_particles * 2 * sizeof(uint64_t));

		// setting up random number generators 
		init_random_seed(time(NULL), &n_particles, states);
		//init_random_seed(1337, &n_particles, states);

		// simulating particles
        #pragma omp parallel
		{
			size_t p;
			#pragma omp for
			for (long long int i = 0; i < n_particles; i++)
			{
				p = (size_t)i;
				transport_particles
					(
					particles,
					&p,
					((Simulation*)dev_simulation)->shape,
					((Simulation*)dev_simulation)->spacing,
					((Simulation*)dev_simulation)->offset,
					((Simulation*)dev_simulation)->material_map,
					((Simulation*)dev_simulation)->density_map,
					((Simulation*)dev_simulation)->lut_shape,
					((Simulation*)dev_simulation)->attenuation_lut,
					((Simulation*)dev_simulation)->energy_imparted,
					states
					);
			}
		}
		// free  memory
		free(states);
		return;
	}
	#endif

	#ifdef USINGCUDA
	__declspec(dllexport) __host__ void cleanup_simulation(void *dev_simulation, int *shape, double *energy_imparted)
	{
		//struct Simulation *dev_sim = ((struct Simulation*)dev_simulation);
		//copy energy_imparted from device memory to host
		cudaError_t err;
		err = cudaMemcpy(energy_imparted, ((Simulation*)dev_simulation)->energy_imparted, shape[0] * shape[1] * shape[2] * sizeof(double), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("CUDA error: %s\n", cudaGetErrorString(err));
			exit(-1);
		}
		// free device memory
		cudaFree(((Simulation*)dev_simulation)->shape);
		cudaFree(((Simulation*)dev_simulation)->spacing);
		cudaFree(((Simulation*)dev_simulation)->offset);
		cudaFree(((Simulation*)dev_simulation)->material_map);
		cudaFree(((Simulation*)dev_simulation)->density_map);
		cudaFree(((Simulation*)dev_simulation)->lut_shape);
		cudaFree(((Simulation*)dev_simulation)->attenuation_lut);
		cudaFree(((Simulation*)dev_simulation)->energy_imparted);
		free(dev_simulation);
		
		return;
	}
#else
	__declspec(dllexport) void cleanup_simulation(void *dev_simulation, int *shape, double *energy_imparted)
	{
		/*
		// free device memory
		free(((Simulation*)dev_simulation)->shape);
		free(((Simulation*)dev_simulation)->spacing);
		free(((Simulation*)dev_simulation)->offset);
		free(((Simulation*)dev_simulation)->material_map);
		free(((Simulation*)dev_simulation)->density_map);
		free(((Simulation*)dev_simulation)->lut_shape);
		free(((Simulation*)dev_simulation)->attenuation_lut);
		free(((Simulation*)dev_simulation)->energy_imparted);
		*/
		free(dev_simulation);

		return;
	}
#endif
}

//
//__global__ void particle_is_intersecting_volume_test_kernel(double *particles, size_t *n_particles, int *shape, double *spacing, double *offset, bool *result)
//{
//	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	double particle[8];
//	if (id < n_particles[0])
//	{
//		for (size_t j = 0; j < 8; j++)
//			particle[j] = particles[id * 8 + j];
//		result[id] = particle_is_intersecting_volume(&particle[0], shape, spacing, offset);
//		for (size_t j = 0; j < 8; j++)
//			particles[j + id * 8] = particle[j];
//	}
//}
//__global__ void lut_interpolator_test_kernel(double *particles, size_t *n_particles, int *lut_shape, double *lut)
//{
//	int id = threadIdx.x + blockIdx.x * blockDim.x;
//	double particle[8];
//	size_t lower_index;
//	if (id < n_particles[0])
//	{
//		for (int j = 0; j < 8; j++)
//			particle[j] = particles[id * 8 + j];
//		particles[7 + id * 8] = lut_interpolator(0, 4, particle[6], lut_shape, lut, &lower_index);
//	}
//}
//
//__global__ void woodcock_test_kernel(double *particles, size_t *n_particles, int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, double *energy_imparted, uint64_t *states, bool *result)
//{
//	int id = threadIdx.x + blockIdx.x * blockDim.x;
//	double particle[8];
//
//	size_t volume_index;
//	if (id < n_particles[0])
//	{
//		for (int j = 0; j < 8; j++)
//			particle[j] = particles[id * 8 + j];
//		result[id] = woodcock_step(&volume_index, particle, shape, spacing, offset, material_map, density_map, att_shape, attenuation_lut, states);
//		for (int j = 0; j < 8; j++)
//			particles[id * 8 + j] = particle[j];
//	}
//}
//
//
//
//__host__ void setup_test_environment(double *particles, size_t n_particles, int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, double *energy_imparted)
//{
//	int i, j, k;
//	size_t ind;
//
//	//geometry
//	for (i = 0; i < 3; i++)
//	{
//		spacing[i] = 0.1;
//		offset[i] = -shape[i] * spacing[i] / 2.;
//		//offset[i] = 0;
//	}
//
//
//	//particles
//	for (i = 0; i < (int)n_particles; i++)
//	{
//		particles[i * 8] = -1000.;
//		particles[i * 8 + 1] = 0;
//		particles[i * 8 + 2] = 0;
//		particles[i * 8 + 3] = 1;
//		particles[i * 8 + 4] = 0;
//		particles[i * 8 + 5] = 0;
//		particles[i * 8 + 6] = 70000;
//		particles[i * 8 + 7] = 1;
//	}
//	for (i = 0; i < shape[0]; i++)
//		for (j = 0; j < shape[1]; j++)
//			for (k = 0; k < shape[2]; k++)
//			{
//				ind = i * shape[2] * shape[1] + j * shape[2] + k;
//				material_map[ind] = 0;
//				density_map[ind] = 1;
//				energy_imparted[ind] = 0;
//			}
//
//	//lut
//	for (int i = 0; i < 2; i++)
//	{
//		//energy 
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 0] = 1000;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 0] = 0.34;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 0] = 0.05;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 0] = 6.8;
//
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 1] = 10000;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 1] = 0.0246;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 1] = 0.358;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 1] = 0.00277;
//
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 2] = 50000;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 2] = 0.00101;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 2] = 0.3344;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 2] = 0.000011;
//
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 3] = 69000;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 3] = 0.00005;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 3] = 0.317;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 3] = 0.000003;
//
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 0 * att_shape[2] + 4] = 100000;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 2 * att_shape[2] + 4] = 0.000276;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 4 * att_shape[2] + 4] = 0.29;
//		attenuation_lut[i*att_shape[1] * att_shape[2] + 3 * att_shape[2] + 4] = 0.000000987;
//
//		for (int k = 0; k < 5; k++)
//		{
//			ind = i*att_shape[1] * att_shape[2] + 1 * att_shape[2] + k;
//			attenuation_lut[ind] = 0;
//		}
//
//		for (int j = 2; j < 5; j++)
//		{
//			for (int k = 0; k < 5; k++)
//			{
//				ind = i*att_shape[1] * att_shape[2] + j * att_shape[2] + k;
//				attenuation_lut[i*att_shape[1] * att_shape[2] + 1 * att_shape[2] + k] += attenuation_lut[ind];
//			}
//		}
//	}
//}
//
//
//int main()
//{
//	// init geometry
//	size_t n_particles = 100000;
//	int shape[3] = { 64, 64, 64 };
//	int lut_shape[3] = { 2, 5, 5 };
//	double spacing[3];
//	double offset[3];
//
//	// init geometry variables
//	double *particles = (double *)malloc(n_particles * 8 * sizeof(double));
//	int *material_map = (int *)malloc(shape[0] * shape[1] * shape[2] * sizeof(int));
//	double *density_map = (double *)malloc(shape[0] * shape[1] * shape[2] * sizeof(double));
//	double *attenuation_lut = (double *)malloc(lut_shape[0] * lut_shape[1] * lut_shape[2] * sizeof(double));
//	double *energy_imparted = (double *)malloc(shape[0] * shape[1] * shape[2] * sizeof(double));
//
//	// initialazing geometry
//	setup_test_environment(particles, n_particles, shape, spacing, offset, material_map, density_map, lut_shape, attenuation_lut, energy_imparted);
//
//	void * sim;
//	sim = setup_simulation(shape, spacing, offset, material_map, density_map, lut_shape, attenuation_lut, energy_imparted);
//	run_simulation(particles, n_particles, sim);
//	cleanup_simulation(sim, shape, energy_imparted);
//	/*
//	size_t index;
//	for (int i = 0; i < shape[0]; i++)
//	{
//		for (int j = 0; j < shape[1]; j++)
//		{
//			for (int k = 0; k < shape[2]; k++)
//			{
//				index = shape[1] * shape[2] * i + shape[2] * j + k;
//				if (energy_imparted[index] > 0)
//				{
//					printf("\n energy in %d %d %d = %f", i, j, k, energy_imparted[index]);
//				}
//			}
//		}
//	}
//	*/
//	return 0;
//}
