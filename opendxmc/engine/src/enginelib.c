
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

#include "enginelib.h"


// CUDA Constants
#ifdef USINGCUDA
__device__ __constant__ double ERRF = 1e-9; // Precision error
__device__ __constant__ double ERRG = 1e-3; // Geometric error
__device__ __constant__ double ELECTRON_MASS = 510998.9;  //  eV/(c*c)
__device__ __constant__ double PI = 3.14159265359;
__device__ __constant__ double ENERGY_CUTOFF = 1000; // eV
__device__ __constant__ double ENERGY_MAXVAL = 300000; // eV
__device__ __constant__ double WEIGHT_CUTOFF = 0.01;
__device__ __constant__ double RUSSIAN_RULETTE_CHANCE = .2;  // CHANCE probability of photon survival
#else
const double ERRF = 1e-9; // Precision error
const double ELECTRON_MASS = 510998.9;  //  eV/(c*c)
const double PI = 3.14159265359;
const double ENERGY_CUTOFF = 1000; // eV
const double WEIGHT_CUTOFF = 0.01;
const double RUSSIAN_RULETTE_CHANCE = .2; //CHANCE probability of photon survival
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
	double t[2];
	double t_cand;
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
	if (plane_intersection == 0)
	{
		t_cand = t[0];
	}
	else
	{
		t_cand = fmin(t[0], t[1]);
	}

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
bool woodcock_step(size_t *volume_index, double *particle, int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, double *max_density, uint64_t *state)
{ /*Make the particle take a woodcock step until an interaction occurs or the particle steps out of volume, returns true if an interaction occurs, then volume index contains the voxel_index for the interaction*/

	bool interaction = false;
	bool valid = particle_is_intersecting_volume(particle, shape, spacing, offset);
	//printf("\nValid: %d: shape: %d, %d, %d: spacing: %f, %f, %f: Offset: %f, %f, %f", valid, shape[0], shape[1], shape[2], spacing[0], spacing[1], spacing[2], offset[0], offset[1], offset[2]);

	//printf("\nParticle: %f, %f, %f, %f, %f, %f, %f, %f", particle[0], particle[1], particle[2], particle[3], particle[4], particle[5], particle[6], particle[7]);

	double smin, scur, w_step;
	size_t lut_index;
	int i;
	smin = 0;
	for (i = 0; i < att_shape[0]; i++)
	{
		smin = fmax(smin, lut_interpolator(i, 1, particle[6], att_shape, attenuation_lut, &lut_index));
	}
	smin *= max_density[0];

	while (valid && !interaction)
	{
		
		// samplinf distance
		w_step = -log(randomduniform(&state[0])) / smin;

		//moving particle a w_step
		for (i = 0; i < 3; i++)
		{
			particle[i] += particle[i + 3] * w_step;
		}
		// test to see if particle still is inside volume
		valid = particle_inside_volume(particle, shape, spacing, offset); // skips intersection test

		if (valid)
		{
			volume_index[0] = particle_array_index(particle, shape, spacing, offset);
			scur = lut_interpolator(material_map[volume_index[0]], 1, particle[6], att_shape, attenuation_lut, &lut_index) * density_map[volume_index[0]]; // basicly total attenuation(E) * density
			//printf("\nstep size %f, smin %f, 'scur %f, int_prob %f", w_step, smin, scur, exp(-w_step*scur));
			
			/* I beleve this section is wrong 
			if ((scur / smin) > randomduniform(&state[0]))
				interaction = randomduniform(&state[0]) > exp(-w_step*scur);
			Replasing it with the line below*/
			interaction = randomduniform(&state[0]) <= (scur / smin);
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
void rotate_3Dvector(double *vector, double *axis, double angle)
// angle must be inside (0, PI]
{
	double out[3];
	double sang = sin(angle);
	double cang = cos(angle);
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
__device__
#endif
void generate_particle(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, int *specter_elements, double *specter_cpd, double *specter_energy, double *particle, uint64_t *state)
{
	double v_rot[3];
	double v_z_lenght, v_rot_lenght;
	// cross product scan_axis x source_direction
	v_rot[0] = scan_axis[1] * source_direction[2] - scan_axis[2] * source_direction[1];
	v_rot[1] = scan_axis[2] * source_direction[0] - scan_axis[0] * source_direction[2];
	v_rot[2] = scan_axis[0] * source_direction[1] - scan_axis[1] * source_direction[0];

	
	v_z_lenght = collimation[0] / (2 * sdd[0]) * (randomduniform(state) - 0.5) * 2;
	v_rot_lenght = fov[0] * 2 / sdd[0] * (randomduniform(state) - 0.5) * 2;

	
	double inv_vec_lenght = 1. / sqrt(1. + v_rot_lenght * v_rot_lenght + v_z_lenght * v_z_lenght);

	for (size_t i = 0; i < 3; i++)
	{
		particle[i] = source_position[i];
		particle[i + 3] = (source_direction[i] + v_rot[i] * v_rot_lenght + scan_axis[i] * v_z_lenght) * inv_vec_lenght;
	}
	double r1 = randomduniform(state);
	size_t j;
	for (j = 0; j < specter_elements[0]; j++)
	{
		if (specter_cpd[j] > r1)
		{
			particle[6] = specter_energy[j];
			break;
		}
	}

	if (j == specter_elements[0])
	{
		particle[6] = specter_energy[specter_elements[0] - 1];
	}

	particle[7] = weight[0];
}



#ifdef USINGCUDA
__global__
#endif
void transport_particles(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, int *specter_elements, double *specter_cpd, double *specter_energy, size_t *n_particles, int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *att_shape, double *attenuation_lut, double *energy_imparted, double *max_density, uint64_t *states)
{
	#ifdef USINGCUDA
		size_t id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= n_particles[0])
		{
			return;
		}
	#else
		size_t id = n_particles[0];
	#endif

	double particle[8];
	double rayleight, photoelectric,  r_interaction, scatter_angle, scatter_energy;
	size_t volume_index, lut_index;
	generate_particle(source_position, source_direction, scan_axis, sdd, fov, collimation, weight, specter_elements, specter_cpd, specter_energy, particle, &states[id * 2]);
	
	// preforming woodcock until interaction
	while (woodcock_step(&volume_index, particle, shape, spacing, offset, material_map, density_map, att_shape, attenuation_lut, max_density, &states[id * 2]))
	{
		rayleight = lut_interpolator(material_map[volume_index], 2, particle[6], att_shape, attenuation_lut, &lut_index);
		photoelectric = interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2] * 3], attenuation_lut[lut_index + att_shape[2] * 3 + 1]);

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

		//test for low weight threshold and do a russian rulette photon termination
		if (particle[7] < WEIGHT_CUTOFF)
		{
			if (RUSSIAN_RULETTE_CHANCE < randomduniform(&states[id * 2]))
			{
				particle[7] /= RUSSIAN_RULETTE_CHANCE;
			}
			else
			{
				break;
			}
		}
	}
}


#ifdef USINGCUDA
__global__ void init_random_seed(uint64_t *seed, size_t *n_threads, uint64_t *states)
{
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id == 0)
	{
		states[0] = xorshift128plus(seed);
		states[1] = xorshift128plus(seed);
		for (long long int id = 1; id < n_threads[0]; id++)
		{
			states[id * 2] = xorshift128plus(&states[id * 2 - 2]);
			states[id * 2 + 1] = xorshift128plus(&states[id * 2 - 1]);
		}
	}
}
#else
void init_random_seed(uint64_t *seed, size_t *n_threads, uint64_t *states)
{
	states[0] = xorshift128plus(seed);
	states[1] = xorshift128plus(seed);
	for (long long int id = 1; id < n_threads[0]; id++)
	{	
		states[id * 2] = xorshift128plus(&states[id * 2 - 2]);
		states[id * 2 + 1] = xorshift128plus(&states[id * 2 - 1]);
	}
}
#endif

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
	__host__ void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *attenuation_lut, double *energy_imparted)
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
		double *max_dens_dev;

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
		cudaMalloc(&max_dens_dev, sizeof(double));

		// check for error
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		double max_dens[1];
		max_dens[0] = 0;
		for (size_t i = 0; i < shape[0] * shape[1] * shape[2]; i++)
		{
			max_dens[0] = fmax(max_dens[0], density_map[i]);
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
		cudaMemcpy(max_dens_dev, max_dens, sizeof(double), cudaMemcpyHostToDevice);

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
		sim_dev->max_density = max_dens;

		return (void*)sim_dev;
	}
	#else
	void* setup_simulation(int *shape, double *spacing, double *offset, int *material_map, double *density_map, int *lut_shape, double *attenuation_lut, double *energy_imparted)
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
		
		double *max_dens = (double*)malloc(sizeof(double));
		max_dens[0] = 0;
		//#pragma omp parallel for reduction(max : red_max_val)
		for (size_t i = 0; i < shape[0] * shape[1] * shape[2]; i++)
		{
			max_dens[0] = fmax(max_dens[0], density_map[i]);
		}
		sim_dev->max_density = max_dens;

		sim_dev->seed = (uint64_t*)malloc(2 * sizeof(uint64_t));
		(sim_dev->seed)[0] = time(NULL);
		(sim_dev->seed)[1] = shape[0];
		return (void*)sim_dev;
	}
	#endif


#ifdef USINGCUDA
	 __host__ void* setup_source(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, double *specter_cpd, double *specter_energy, int *specter_elements)
	{
		// device declarations
		double *source_position_dev;
		double *source_direction_dev;
		double *scan_axis_dev;
		double *sdd_dev;
		double *fov_dev;
		double *collimation_dev;
		double *weight_dev;
		int *specter_elements_dev;
		double *specter_cpd_dev;
		double *specter_energy_dev;
		
		// device allocations
		cudaMalloc(&source_position_dev, 3 * sizeof(double));
		cudaMalloc(&source_direction_dev, 3 * sizeof(double));
		cudaMalloc(&scan_axis_dev, 3 * sizeof(double));
		cudaMalloc(&sdd_dev, sizeof(double));
		cudaMalloc(&fov_dev, sizeof(double));
		cudaMalloc(&collimation_dev, sizeof(double));
		cudaMalloc(&weight_dev, sizeof(double));
		cudaMalloc(&specter_elements_dev, sizeof(int));
		cudaMalloc(&specter_cpd_dev, specter_elements[0] * sizeof(double));
		cudaMalloc(&specter_energy_dev, specter_elements[0] * sizeof(double));
		
		// check for error
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		// memory transfer to device
		cudaMemcpy(source_position_dev, source_position, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(source_direction_dev, source_direction, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(scan_axis_dev, scan_axis, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(sdd_dev, sdd, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(fov_dev, fov, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(collimation_dev, collimation, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(weight_dev, weight, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(specter_elements_dev, specter_elements, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(specter_cpd_dev, specter_cpd, specter_elements[0] * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(specter_energy_dev, specter_energy, specter_elements[0] * sizeof(double), cudaMemcpyHostToDevice);


		// creating source structure
		Source *source_dev = (Source*)malloc(sizeof(Source));
		source_dev->source_position = source_position_dev;
		source_dev->source_direction = source_direction_dev;
		source_dev->scan_axis = scan_axis_dev;
		source_dev->sdd = sdd_dev;
		source_dev->fov = fov_dev;
		source_dev->collimation = collimation_dev;
		source_dev->weight = weight_dev;
		source_dev->specter_elements = specter_elements_dev;
		source_dev->specter_cpd = specter_cpd_dev;
		source_dev->specter_energy = specter_energy_dev;
		return (void*)source_dev;
	}

#else
	void* setup_source(double *source_position, double *source_direction, double *scan_axis, double *sdd, double *fov, double *collimation, double *weight, double *specter_cpd, double *specter_energy, int *specter_elements)
	{
		Source *source_dev = (Source*)malloc(sizeof(Source));
		source_dev->source_position = source_position;
		source_dev->source_direction = source_direction;
		source_dev->scan_axis = scan_axis;
		source_dev->sdd = sdd;
		source_dev->fov = fov;
		source_dev->collimation = collimation;
		source_dev->weight = weight;
		source_dev->specter_elements=specter_elements;
		source_dev->specter_cpd = specter_cpd;
		source_dev->specter_energy = specter_energy;
		return (void*)source_dev;
	}
#endif



	
#ifdef USINGCUDA
	__host__ void run_simulation(void *dev_source, size_t n_particles, void *dev_simulation)
	{
		dim3 blocks((int)(n_particles / 512 + 1), 1, 1);
		dim3 threads(512, 1, 1);
		cudaError_t error;

		uint64_t seed = time(NULL);


		// initialize device pointers
		size_t *dev_n_particles;
		uint64_t *dev_states;
		uint64_t *dev_seed;

		//allocate memory on device
		cudaMalloc(&dev_n_particles, sizeof(size_t));
		cudaMalloc(&dev_states, n_particles * 2 * sizeof(uint64_t));
		cudaMalloc(&dev_seed, sizeof(uint64_t));
		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		// transfer data to device
		cudaMemcpy(dev_n_particles, &n_particles, sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_seed, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice);

		// check for error
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error in memory copy: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		// setting up random number generators 
		init_random_seed <<< blocks, threads >>>(dev_seed, dev_n_particles, dev_states);	
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
			((Source*)dev_source)->source_position,
			((Source*)dev_source)->source_direction,
			((Source*)dev_source)->scan_axis,
			((Source*)dev_source)->sdd,
			((Source*)dev_source)->fov,
			((Source*)dev_source)->collimation,
			((Source*)dev_source)->weight,
			((Source*)dev_source)->specter_elements,
			((Source*)dev_source)->specter_cpd,
			((Source*)dev_source)->specter_energy,
			dev_n_particles,
			((Simulation*)dev_simulation)->shape,
			((Simulation*)dev_simulation)->spacing,
			((Simulation*)dev_simulation)->offset,
			((Simulation*)dev_simulation)->material_map,
			((Simulation*)dev_simulation)->density_map,
			((Simulation*)dev_simulation)->lut_shape,
			((Simulation*)dev_simulation)->attenuation_lut,
		    ((Simulation*)dev_simulation)->energy_imparted,
			((Simulation*)dev_simulation)->max_density,
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
		cudaFree(dev_n_particles);
		cudaFree(dev_states);
		cudaFree(dev_seed);
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
	void run_simulation(void *dev_source, size_t n_particles, void *dev_simulation)
	{		
		// simulating particles

		size_t thread_number;
		size_t n_threads = omp_get_max_threads();
		
		uint64_t *states = (uint64_t*)malloc(2 * n_threads * sizeof(uint64_t));
		
		init_random_seed(((Simulation*)dev_simulation)->seed, &n_threads, states);
		#pragma omp parallel num_threads(n_threads) private(thread_number)
		{
			thread_number = omp_get_thread_num();
			long long int i;
			#pragma omp for
			for ( i = 0; i < n_particles; i++)
			{
				transport_particles
					(
					((Source*)dev_source)->source_position,
					((Source*)dev_source)->source_direction,
					((Source*)dev_source)->scan_axis,
					((Source*)dev_source)->sdd,
					((Source*)dev_source)->fov,
					((Source*)dev_source)->collimation,
					((Source*)dev_source)->weight,
					((Source*)dev_source)->specter_elements,
					((Source*)dev_source)->specter_cpd,
					((Source*)dev_source)->specter_energy,
					&thread_number,
					((Simulation*)dev_simulation)->shape,
					((Simulation*)dev_simulation)->spacing,
					((Simulation*)dev_simulation)->offset,
					((Simulation*)dev_simulation)->material_map,
					((Simulation*)dev_simulation)->density_map,
					((Simulation*)dev_simulation)->lut_shape,
					((Simulation*)dev_simulation)->attenuation_lut,
					((Simulation*)dev_simulation)->energy_imparted,
					((Simulation*)dev_simulation)->max_density,
					states);
			}		
		}
		// free  memory
		if (states)
		{
			free(states);
		}
		return;
	}
#endif

#ifdef USINGCUDA
	__host__ void cleanup_simulation(void *dev_simulation, int *shape, double *energy_imparted)
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
		cudaFree(((Simulation*)dev_simulation)->max_density);
		free(dev_simulation);
		
		return;
	}
#else
	void cleanup_simulation(void *dev_simulation, int *shape, double *energy_imparted)
	{
		
		// free memory
		/*
		free(((Simulation*)dev_simulation)->shape);
		free(((Simulation*)dev_simulation)->spacing);
		free(((Simulation*)dev_simulation)->offset);
		free(((Simulation*)dev_simulation)->material_map);
		free(((Simulation*)dev_simulation)->density_map);
		free(((Simulation*)dev_simulation)->lut_shape);
		free(((Simulation*)dev_simulation)->attenuation_lut);
		free(((Simulation*)dev_simulation)->energy_imparted);
		*/
		free(((Simulation*)dev_simulation)->max_density);
		free(((Simulation*)dev_simulation)->seed);
		free(dev_simulation);
		return;
	}
#endif

#ifdef USINGCUDA
	__host__ void cleanup_source(void *dev_source)
	{
		// free device memory
		cudaFree(((Source*)dev_source)->source_position);
		cudaFree(((Source*)dev_source)->source_direction);
		cudaFree(((Source*)dev_source)->scan_axis);
		cudaFree(((Source*)dev_source)->sdd);
		cudaFree(((Source*)dev_source)->fov);
		cudaFree(((Source*)dev_source)->collimation);
		cudaFree(((Source*)dev_source)->weight);
		cudaFree(((Source*)dev_source)->specter_elements);
		cudaFree(((Source*)dev_source)->specter_cpd);
		cudaFree(((Source*)dev_source)->specter_energy);
		
		free(dev_source);
		return;
	}
#else
	void cleanup_source(void *dev_source)
	{
		/*
		free(((Source*)dev_source)->source_position);
		free(((Source*)dev_source)->source_direction);
		free(((Source*)dev_source)->scan_axis);
		free(((Source*)dev_source)->sdd);
		free(((Source*)dev_source)->fov);
		free(((Source*)dev_source)->collimation);
		free(((Source*)dev_source)->weight);
		free(((Source*)dev_source)->specter_elements);
		free(((Source*)dev_source)->specter_cpd);
		free(((Source*)dev_source)->specter_energy);
		*/
		free(dev_source);
		return;
	}
#endif
//}
