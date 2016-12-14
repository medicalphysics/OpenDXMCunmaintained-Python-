
#include "enginelib.h"

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
FLOAT randomduniform(uint64_t *seed)
{
	return (FLOAT)xorshift128plus(seed) / (FLOAT)UINT64_MAX;
}



#ifdef USINGCUDA
#ifdef USING_DOUBLE
__device__ double atomicAdd(double *address, double val)
/* Atomic add of FLOAT to array. Returns old value. */
{
	unsigned long long int *address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

	} while (assumed != old);
	return __longlong_as_double(old);
}
// atomic add for floats are already implemented by nvidia 
#endif

#else
FLOAT atomicAdd(FLOAT *address, FLOAT val)
{
#pragma omp atomic
	address[0] += val;
	return val;
}
#endif


#ifdef USINGCUDA
__device__
#endif
FLOAT interp(FLOAT x, FLOAT x1, FLOAT x2, FLOAT y1, FLOAT y2)
{
	return y1 + ((y2 - y1) * ((x - x1)) / (x2 - x1));
}

#ifdef USINGCUDA
__device__
#endif
void binary_search(FLOAT *arr, FLOAT value, size_t *start, size_t *stop)
{
	size_t mid = (stop[0] + start[0]) / 2;
	while (mid != start[0])
	{
		if (value < arr[mid])
		{
			stop[0] = mid;
		}
		else
		{
			start[0] = mid;
		}
		mid = (stop[0] + start[0]) / 2;
	}
}

#ifdef USINGCUDA
__device__
#endif
FLOAT interp_array(FLOAT *xarr, FLOAT *yarr, size_t array_size, FLOAT xval)
{
	size_t low = 0;
	size_t high = array_size - 1;

	if (xval <= xarr[low])
	{
		return yarr[low];
	}
	if (xval >= xarr[high])
	{
		return yarr[high];
	}

	binary_search(xarr, xval, &low, &high);
	return interp(xval, xarr[low], xarr[high], yarr[low], yarr[high]);
}


#ifdef USINGCUDA
__device__
#endif
FLOAT lut_interpolator(int material, int interaction, FLOAT energy, int *lut_shape, FLOAT *lut, size_t *lower_index)
{
	lower_index[0] = material * lut_shape[1] * lut_shape[2];
	size_t higher_index = lower_index[0] + lut_shape[2] - 1;
	binary_search(lut, energy, lower_index, &higher_index);
	return interp(energy, lut[lower_index[0]], lut[lower_index[0] + 1], lut[lower_index[0] + lut_shape[2] * interaction], lut[lower_index[0] + lut_shape[2] * interaction + 1]);
}


#ifdef USINGCUDA
__device__
#endif
size_t particle_array_index(FLOAT *particle, int *shape, FLOAT *spacing, FLOAT *offset)
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
	return i * shape[2] * shape[1] + j * shape[2] + k;

}
///////////////////////////////////siddon/////////////////////////////////


#ifdef USINGCUDA
__device__
#endif
int min_index3(FLOAT *arr)
{
	if ((arr[0] <= arr[1]) && (arr[0] <= arr[2]))
	{
		return 0;
	}
	else if ((arr[1] <= arr[0]) && (arr[1] <= arr[2]))
	{
		return 1;
	}
	return 2;
}


#ifdef USINGCUDA
__device__
#endif
void normalize_ray(FLOAT *ray)
{
	FLOAT inv_len = 1.f / SQRT(ray[3] * ray[3] + ray[4] * ray[4] + ray[5] * ray[5]);
	ray[3] *= inv_len;
	ray[4] *= inv_len;
	ray[5] *= inv_len;
}

#ifdef USINGCUDA
__device__
#endif
void calculate_alphas_extreme(FLOAT *ray, int *N, FLOAT *spacing, FLOAT *offset, FLOAT *aupdate, FLOAT *aglobalmin, FLOAT *aglobalmax)
//calculates alpha for planes , planes : int[3]
{
	FLOAT a0, aN;
	aglobalmax[0] = INFINITY;
	aglobalmin[0] = 0;
	for (size_t i = 0; i < 3; i++)
	{
		//test if direction is zero
		if (FABS(ray[i + 3]) > ERRF)
		{
			a0 = FMAX((offset[i] - ray[i]) / ray[i + 3], 0.f);
			aN = FMAX((offset[i] + N[i] * spacing[i] - ray[i]) / ray[i + 3], 0.f);
			if (a0 <= aN)
			{
				aglobalmin[0] = FMAX(aglobalmin[0], a0);
				aglobalmax[0] = FMIN(aglobalmax[0], aN);
			}
			else
			{
				aglobalmin[0] = FMAX(aglobalmin[0], aN);
				aglobalmax[0] = FMIN(aglobalmax[0], a0);
			}
			aupdate[i] = spacing[i] / FABS(ray[i + 3]);
		}
		else
		{
			aupdate[i] = INFINITY;
		}
	}

}

#ifdef USINGCUDA
__device__
#endif
void calculate_first_indices(FLOAT *ray, int *N, FLOAT *spacing, FLOAT *offset, FLOAT *aglobalmin, FLOAT *aupdate, size_t *indicesmin, int *indexupdate)
{
	FLOAT a_cand = aglobalmin[0] + ERRF;
	FLOAT index;
	for (size_t i = 0; i < 3; i++)
	{
		if (ray[i + 3] > 0.f)
		{
			index = ((ray[i] + a_cand * ray[i + 3] - offset[i]) / spacing[i]);
			indexupdate[i] = 1;
		}
		else
		{
			index = ((ray[i] + a_cand * ray[i + 3] - offset[i]) / spacing[i]);
			indexupdate[i] = -1;
		}
		
		indicesmin[i] = (size_t)index;
		if (index < 0)
		{
			indicesmin[i] = 0;
		}
		else if (index > (N[i] - 1))
		{
			indicesmin[i] = N[i] - 1;
		}
		else
		{
			indicesmin[i] = (size_t)index;
		}
	}
}

#ifdef USINGCUDA
__device__
#endif
bool siddon_path(size_t *volume_index, FLOAT *ray, int *N, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *att_shape, FLOAT *attenuation_lut, FLOAT *max_density, uint64_t *state)
{
	/*
	The ray is a FLOAT [6] array: (start_x, start_y, start_z, direction_x, direction_y, direction_z). The vector
	describing dirction of the ray must be a unit vector. N is a int [3] array giving the shape of the uniform voxel volume.
	spacing FLOAT [3] array is voxel size, offset FLOAT [3] array is the posiyional offset of the first voxel in the volume.

	The ray is parameterized by p(alpha) = start + alpha * direction
	*/
	FLOAT amin[3], aupdate[3], aglobalmin, aglobalmax;
	calculate_alphas_extreme(ray, N, spacing, offset, aupdate, &aglobalmin, &aglobalmax);

	if (FABS(aglobalmin - aglobalmax) < ERRF)
	{ //ray is not intersecting
		return false;
	}

	size_t indices[3];
	int indexupdate[3];
	calculate_first_indices(ray, N, spacing, offset, &aglobalmin, aupdate, indices, indexupdate);

	size_t dim_index;
	size_t attenuation_index;
	FLOAT pixel_path_lenght;
	FLOAT pixel_interaction_lenght;

	FLOAT cum_interaction_prob = 1;
	FLOAT interaction_prob;
	FLOAT attenuation_coef;
	FLOAT r1 = randomduniform(state);

	//FLOAT cum_pixel_path_lenght = 0;

	//doing one iteration to get indicec right
	amin[0] = aglobalmin + aupdate[0];
	amin[1] = aglobalmin + aupdate[1];
	amin[2] = aglobalmin + aupdate[2];

	while ((aglobalmin - aglobalmax) < -ERRF)
	{
		dim_index = min_index3(amin);
		pixel_path_lenght = amin[dim_index] - aglobalmin;
		//cum_pixel_path_lenght += pixel_path_lenght;
		volume_index[0] = (size_t)(indices[0] * (size_t)N[1] * (size_t)N[2] + indices[1] * (size_t)N[2] + indices[2]);
		attenuation_coef = density_map[volume_index[0]] * lut_interpolator(material_map[volume_index[0]], 1, ray[6], att_shape, attenuation_lut, &attenuation_index);
		interaction_prob = EXP(-attenuation_coef  * pixel_path_lenght);
		cum_interaction_prob *= interaction_prob;
		if (cum_interaction_prob <= r1)
		{
			//finding interaction path lenght in voxel
			pixel_interaction_lenght = aglobalmin + LOG(r1 *interaction_prob / cum_interaction_prob) / (-attenuation_coef);
			ray[0] += ray[3] * pixel_interaction_lenght;
			ray[1] += ray[4] * pixel_interaction_lenght;
			ray[2] += ray[5] * pixel_interaction_lenght;
			return true;
		}

		//radiological path calc goes here

		aglobalmin = amin[dim_index];
		amin[dim_index] += aupdate[dim_index];
		indices[dim_index] += indexupdate[dim_index];
	}
	return false;
}


////////////////////////////////////////////////////////////////////////////


////////////////////////////////////WOODCOCK////////////////////////////
#ifdef USINGCUDA
__device__ 
#endif
bool particle_on_plane(FLOAT *particle, int *shape, FLOAT *spacing, FLOAT *offset, size_t plane_dimension)
/* Bondary test if the particle is resting on plane p laying on one of the edges of the scoring volume. Returns true if the point on the plane is on the edge of the volume*/
{
	FLOAT llim, ulim;
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
bool particle_inside_volume(FLOAT *particle, int *shape, FLOAT *spacing, FLOAT *offset)
/* Test for particle inside volume. If inside returns true*/
{
	FLOAT llim, ulim;
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
bool particle_is_intersecting_volume(FLOAT *particle, int *shape, FLOAT *spacing, FLOAT *offset)
/*Tests if particle intersects with dose scoring volume. If intersecting and outside scoring volume
the particle is transported along its direction to the volume edge. Returns true if the particle intersects scoring volume,
returns false if the particle misses scoring volume.*/
{
	if (particle_inside_volume(particle, shape, spacing, offset))
	{
		return true;
	}
	size_t i, j;
	FLOAT t[2];
	FLOAT t_cand;
	FLOAT pos[3];
	int plane_intersection = -1;
	for (i = 0; i < 3; i++)
	{
		if (FABS(particle[i + 3]) > ERRF)
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
		t_cand = FMIN(t[0], t[1]);
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
bool woodcock_step(size_t *volume_index, FLOAT *particle, int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *att_shape, FLOAT *attenuation_lut, FLOAT *max_density, uint64_t *state)
{ /*Make the particle take a woodcock step until an interaction occurs or the particle steps out of volume, returns true if an interaction occurs, then volume index contains the voxel_index for the interaction*/

	bool interaction = false;
	bool valid = particle_is_intersecting_volume(particle, shape, spacing, offset);

	FLOAT smin, scur, w_step;
	size_t lut_index;
	size_t i;
	smin = 0;
	for (i = 0; i < att_shape[0]; i++)
	{
		smin = FMAX(smin, lut_interpolator((int)i, 1, particle[6], att_shape, attenuation_lut, &lut_index));
	}
	smin *= max_density[0];

	while (valid && !interaction)
	{
		// sampling distance
		w_step = -LOG(randomduniform(&state[0])) / smin;

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

			interaction = randomduniform(&state[0]) <= (scur / smin);
		}
	}
	return interaction;
}

////////////////////////////////////////////////////////////////////////





#ifdef USINGCUDA
__device__
#endif
void normalize_3Dvector(FLOAT *vector)
{
	FLOAT vector_inv_sum = 1.f / SQRT((vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]));
	vector[0] *= vector_inv_sum;
	vector[1] *= vector_inv_sum;
	vector[2] *= vector_inv_sum;
}

#ifdef USINGCUDA
__device__
#endif
void rotate_3Dvector(FLOAT *vector, FLOAT *axis, FLOAT angle)
// angle must be inside (0, PI]
{
	FLOAT out[3];
	FLOAT sang = SIN(angle);
	FLOAT cang = COS(angle);
	FLOAT midterm = (1.f - cang) * (vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2]);

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
void rotate_particle(FLOAT *particle, FLOAT theta, FLOAT phi)
// rotates a particle theta degrees from its current direction phi degrees about a random axis orthogonal to the direction vector (this axis is rotated phi degrees about the particles direction vector)
{
	// First we find a vector orthogonal to the particle direction
	// k_xy = v x k   where k = ({1, 0, 0} , {0, 1, 0}, {0, 0, 1}) depending on the smallest magnitude of the particles direction vector (we do this to make the calculation more robust)
	FLOAT k_xy[3];
	if ((FABS(particle[3]) < FABS(particle[4])) && (FABS(particle[3]) < FABS(particle[5])))
	{
		k_xy[0] = 0;
		k_xy[1] = particle[5];
		k_xy[2] = -particle[4];
	}
	else if ((FABS(particle[4]) < FABS(particle[3])) && (FABS(particle[4]) < FABS(particle[5])))
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
	
	//rotate_3Dvector(&particle[3], k_xy, theta);
	//This is cheaper
	FLOAT tsin = SIN(theta);
	FLOAT tcos = COS(theta);
	for (size_t i = 0; i < 3; i++)
	{
		particle[i+3] = particle[i+3] * tcos + k_xy[i] * tsin;
	}
}
#ifdef USINGCUDA
__device__
#endif
void rayleigh_event_draw_theta(FLOAT *angle, uint64_t *state)
{
	FLOAT r, c, A;
	r = randomduniform(&state[0]);
	c = 4.f - 8.f * r;
	if (c > 0)
		A = POW((FABS(c) + SQRT(c * c + 4.f)) / 2.f, (1.f / 3.f));
	else
		A = -POW((FABS(c) + SQRT(c * c + 4.f)) / 2.f, (1.f / 3.f));
	angle[0] = ACOS(A - 1.f / A);
}

#ifdef USINGCUDA
__device__
#endif
FLOAT compton_event_draw_energy_theta(FLOAT energy, FLOAT *theta, uint64_t *state)
{
	//Draws scattered energy and angle, based on Geant4 implementation, returns scattered energy and sets theta to scatter angle
	FLOAT epsilon_0, alpha1, alpha2, r1, r2, r3, epsilon, qsin_theta, t, k;
	k = energy / ELECTRON_MASS;
	epsilon_0 = 1.f / (1.f + 2.f * k);
	alpha1 = LOG(1.f / epsilon_0);
	alpha2 = (1.f - epsilon_0 * epsilon_0) / 2.f;
	while (true)
	{
		r1 = randomduniform(&state[0]);
		r2 = randomduniform(&state[0]);
		r3 = randomduniform(&state[0]);

		if (r1 < alpha1 / (alpha1 + alpha2))
		{
			epsilon = EXP(-r2 * alpha1);
		}
		else
		{
			epsilon = SQRT((epsilon_0 * epsilon_0 + (1.f - epsilon_0 * epsilon_0) * r2));
		}

		t = ELECTRON_MASS * (1.f - epsilon) / (energy * epsilon);
		qsin_theta = t * (2.f - t);

		if ((1.f - epsilon / (1.f + epsilon * epsilon) * qsin_theta) >= r3)
			break;
	}
	
	theta[0] = ACOS(1.f + (1.f - 1.f / epsilon) / k);
	return epsilon * energy;
}


#ifdef USINGCUDA
__device__
#endif
void generate_particle(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *sdd, FLOAT *fov, FLOAT *collimation, FLOAT *weight, int *specter_elements, FLOAT *specter_cpd, FLOAT *specter_energy, FLOAT *particle, uint64_t *state)
{
	FLOAT v_rot[3];
	FLOAT v_z_lenght, v_rot_lenght;
	// cross product scan_axis x source_direction
	v_rot[0] = scan_axis[1] * source_direction[2] - scan_axis[2] * source_direction[1];
	v_rot[1] = scan_axis[2] * source_direction[0] - scan_axis[0] * source_direction[2];
	v_rot[2] = scan_axis[0] * source_direction[1] - scan_axis[1] * source_direction[0];

	v_z_lenght = collimation[0] / (2.f * sdd[0]) * (randomduniform(state) - 0.5f) * 2.f;
	v_rot_lenght = fov[0] * 2.f / sdd[0] * (randomduniform(state) - 0.5f) * 2.f;

	FLOAT inv_vec_lenght = 1.f / SQRT(1.f + v_rot_lenght * v_rot_lenght + v_z_lenght * v_z_lenght);

	for (size_t i = 0; i < 3; i++)
	{
		particle[i] = source_position[i];
		particle[i + 3] = (source_direction[i] + v_rot[i] * v_rot_lenght + scan_axis[i] * v_z_lenght) * inv_vec_lenght;
	}

	FLOAT r1 = randomduniform(state);

	particle[6] = interp_array(specter_cpd, specter_energy, specter_elements[0], r1);
	particle[7] = weight[0];
}

#ifdef USINGCUDA
__device__
#endif
void generate_particle_bowtie(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *scan_axis_fan_angle, FLOAT *rot_axis_fan_angle, FLOAT *weight, int *specter_elements, FLOAT *specter_cpd, FLOAT *specter_energy, int *bowtie_elements, FLOAT *bowtie_weight, FLOAT *bowtie_angle,
FLOAT *particle, uint64_t *state)
{
	FLOAT v_rot[3];
	FLOAT v_z_lenght, v_rot_lenght, r_ang;;
	// cross product scan_axis x source_direction
	v_rot[0] = scan_axis[1] * source_direction[2] - scan_axis[2] * source_direction[1];
	v_rot[1] = scan_axis[2] * source_direction[0] - scan_axis[0] * source_direction[2];
	v_rot[2] = scan_axis[0] * source_direction[1] - scan_axis[1] * source_direction[0];

	r_ang = rot_axis_fan_angle[0] * (randomduniform(state) - 0.5f);
	v_z_lenght = ASIN(scan_axis_fan_angle[0] * (randomduniform(state) - 0.5f));
	v_rot_lenght = ASIN(r_ang);

	FLOAT inv_vec_lenght = 1.f / SQRT(1.f + v_rot_lenght * v_rot_lenght + v_z_lenght * v_z_lenght);

	//setting position and direction of particle
	for (size_t i = 0; i < 3; i++)
	{
		particle[i] = source_position[i];
		particle[i + 3] = (source_direction[i] + v_rot[i] * v_rot_lenght + scan_axis[i] * v_z_lenght) * inv_vec_lenght;
	}

	/////////////selecting energy///////////////////
	FLOAT r1 = randomduniform(state);
	particle[6] = interp_array(specter_cpd, specter_energy, specter_elements[0], r1);
	//selecting weight of particle
	particle[7] = weight[0];
	if (bowtie_elements[0] == 1)
	{
		particle[7] *= bowtie_weight[0];
	}
	else
	{
		particle[7] *= interp_array(bowtie_angle, bowtie_weight, bowtie_elements[0], r_ang);
	}
}


#ifdef USINGCUDA
__global__
#endif
void transport_particles(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *sdd, FLOAT *fov, FLOAT *collimation, FLOAT *weight, int *specter_elements, FLOAT *specter_cpd, FLOAT *specter_energy, size_t *n_particles, int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *att_shape, FLOAT *attenuation_lut, FLOAT *energy_imparted, FLOAT *max_density, trackingFuncPtr tracking_func, uint64_t *states)
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

	FLOAT particle[8];
	FLOAT rayleight, photoelectric, r_interaction, scatter_angle, scatter_energy;
	size_t volume_index, lut_index;
	generate_particle(source_position, source_direction, scan_axis, sdd, fov, collimation, weight, specter_elements, specter_cpd, specter_energy, particle, &states[id * 2]);

	while ((*tracking_func)(&volume_index, particle, shape, spacing, offset, material_map, density_map, att_shape, attenuation_lut, max_density, &states[id * 2]))
	{
		rayleight = lut_interpolator(material_map[volume_index], 2, particle[6], att_shape, attenuation_lut, &lut_index);
		photoelectric = interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2] * 3], attenuation_lut[lut_index + att_shape[2] * 3 + 1]);

		r_interaction = randomduniform(&states[id * 2]) * interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2]], attenuation_lut[lut_index + att_shape[2] + 1]);

		if (rayleight > r_interaction) //rayleigh scatter event
		{
			rayleigh_event_draw_theta(&scatter_angle, &states[id * 2]);
			rotate_particle(particle, scatter_angle, (randomduniform(&states[id * 2]) * 2.f - 1.f) * PI);
		}
		else if ((rayleight + photoelectric) > r_interaction) //photoelectric event
		{
			atomicAdd(&energy_imparted[volume_index], particle[6] * particle[7]);
			break;
		}
		else // compton event
		{
			scatter_energy = compton_event_draw_energy_theta(particle[6], &scatter_angle, &states[id * 2]);
			rotate_particle(particle, scatter_angle, (randomduniform(&states[id * 2]) * 2.f - 1.f) * PI);
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
__global__
#endif
void transport_particles_bowtie(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *scan_axis_fan_angle, FLOAT *rot_axis_fan_angle, FLOAT *weight, int *specter_elements, FLOAT *specter_cpd, FLOAT *specter_energy, int *bowtie_elements, FLOAT *bowtie_weight, FLOAT *bowtie_angle, size_t *n_particles, int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *att_shape, FLOAT *attenuation_lut, FLOAT *energy_imparted, FLOAT *max_density, trackingFuncPtr tracking_func, uint64_t *states)
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

	FLOAT particle[8];
	FLOAT rayleight, photoelectric, r_interaction, scatter_angle, scatter_energy;
	size_t volume_index, lut_index;
	generate_particle_bowtie(source_position, source_direction, scan_axis, scan_axis_fan_angle, rot_axis_fan_angle, weight, specter_elements, specter_cpd, specter_energy, bowtie_elements, bowtie_weight, bowtie_angle, particle, &states[id * 2]);

	while ((*tracking_func)(&volume_index, particle, shape, spacing, offset, material_map, density_map, att_shape, attenuation_lut, max_density, &states[id * 2]))
	{
		rayleight = lut_interpolator(material_map[volume_index], 2, particle[6], att_shape, attenuation_lut, &lut_index);
		//Here we take a shortcut, instead of interpolating the array again we just jump to the already calculated index in the lut table and do a between two points interpolation 
		photoelectric = interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2] * 3], attenuation_lut[lut_index + att_shape[2] * 3 + 1]);

		r_interaction = randomduniform(&states[id * 2]) * interp(particle[6], attenuation_lut[lut_index], attenuation_lut[lut_index + 1], attenuation_lut[lut_index + att_shape[2]], attenuation_lut[lut_index + att_shape[2] + 1]);

		if (rayleight > r_interaction) //rayleigh scatter event
		{
			rayleigh_event_draw_theta(&scatter_angle, &states[id * 2]);
			rotate_particle(particle, scatter_angle, (randomduniform(&states[id * 2]) * 2.f - 1.f) * PI);
		}
		else if ((rayleight + photoelectric) > r_interaction) //photoelectric event
		{
			atomicAdd(&energy_imparted[volume_index], particle[6] * particle[7]);
			break;
		}
		else // compton event
		{
			scatter_energy = compton_event_draw_energy_theta(particle[6], &scatter_angle, &states[id * 2]);
			rotate_particle(particle, scatter_angle, (randomduniform(&states[id * 2]) * 2.f - 1.f) * PI);
			atomicAdd(&energy_imparted[volume_index], (particle[6] - scatter_energy) * particle[7]);
			particle[6] = scatter_energy;
		}

		//test for energy cutoff
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
				//if the photon survives russian rulette, we give it extra weigh to conserve energy
				particle[7] /= RUSSIAN_RULETTE_CHANCE;
			}
			else
			{
				// else we deposit the energy in current voxel
				atomicAdd(&energy_imparted[volume_index], particle[6] * particle[7]);
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
	for (uint64_t id = 1; id < n_threads[0]; id++)
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
int number_of_cuda_devices(){ return -1; }
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
void cuda_device_name(int device_number, char* name){ return; }
#endif

#ifdef USINGCUDA
__host__ void* setup_simulation(int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *lut_shape, FLOAT *attenuation_lut, FLOAT *energy_imparted)
{

	// device declarations
	int *shape_dev;
	FLOAT *spacing_dev;
	FLOAT *offset_dev;
	int *material_map_dev;
	FLOAT *density_map_dev;
	int *lut_shape_dev;
	FLOAT *lut_dev;
	FLOAT *energy_dev;
	FLOAT *max_dens_dev;

	size_t array_size = shape[0] * shape[1] * shape[2];
	size_t lut_size = lut_shape[0] * lut_shape[1] * lut_shape[2];

	// device allocations
	cudaMalloc(&shape_dev, 3 * sizeof(int));
	cudaMalloc(&spacing_dev, 3 * sizeof(FLOAT));
	cudaMalloc(&offset_dev, 3 * sizeof(FLOAT));
	cudaMalloc(&material_map_dev, array_size * sizeof(int));
	cudaMalloc(&density_map_dev, array_size * sizeof(FLOAT));
	cudaMalloc(&lut_shape_dev, 3 * sizeof(int));
	cudaMalloc(&lut_dev, lut_size * sizeof(FLOAT));
	cudaMalloc(&energy_dev, array_size * sizeof(FLOAT));
	cudaMalloc(&max_dens_dev, sizeof(FLOAT));

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	FLOAT max_dens[1];
	max_dens[0] = 0;
	for (size_t i = 0; i < shape[0] * shape[1] * shape[2]; i++)
	{
		max_dens[0] = FMAX(max_dens[0], density_map[i]);
	}


	// memory transfer to device
	cudaMemcpy(shape_dev, shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(spacing_dev, spacing, 3 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(offset_dev, offset, 3 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(material_map_dev, material_map, array_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(density_map_dev, density_map, array_size * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(lut_shape_dev, lut_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(lut_dev, attenuation_lut, lut_size * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(energy_dev, energy_imparted, array_size * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(max_dens_dev, max_dens, sizeof(FLOAT), cudaMemcpyHostToDevice);

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
void* setup_simulation(int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *lut_shape, FLOAT *attenuation_lut, FLOAT *energy_imparted, int *use_siddon)
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
	sim_dev->use_siddon_pathing = use_siddon;

	FLOAT *max_dens = (FLOAT*)malloc(sizeof(FLOAT));
	max_dens[0] = 0;

	for (size_t i = 0; i < sim_dev->shape[0] * sim_dev->shape[1] * sim_dev->shape[2]; i++)
	{
		max_dens[0] = FMAX(max_dens[0], density_map[i]);
	}

	sim_dev->max_density = max_dens;

	sim_dev->seed = (uint64_t*)malloc(2 * sizeof(uint64_t));
	(sim_dev->seed)[0] = time(NULL);
	(sim_dev->seed)[1] = shape[0];
	return (void*)sim_dev;
}
#endif


#ifdef USINGCUDA
__host__ void* setup_source(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *sdd, FLOAT *fov, FLOAT *collimation, FLOAT *weight, FLOAT *specter_cpd, FLOAT *specter_energy, int *specter_elements)
{
	// device declarations
	FLOAT *source_position_dev;
	FLOAT *source_direction_dev;
	FLOAT *scan_axis_dev;
	FLOAT *sdd_dev;
	FLOAT *fov_dev;
	FLOAT *collimation_dev;
	FLOAT *weight_dev;
	int *specter_elements_dev;
	FLOAT *specter_cpd_dev;
	FLOAT *specter_energy_dev;

	// device allocations
	cudaMalloc(&source_position_dev, 3 * sizeof(FLOAT));
	cudaMalloc(&source_direction_dev, 3 * sizeof(FLOAT));
	cudaMalloc(&scan_axis_dev, 3 * sizeof(FLOAT));
	cudaMalloc(&sdd_dev, sizeof(FLOAT));
	cudaMalloc(&fov_dev, sizeof(FLOAT));
	cudaMalloc(&collimation_dev, sizeof(FLOAT));
	cudaMalloc(&weight_dev, sizeof(FLOAT));
	cudaMalloc(&specter_elements_dev, sizeof(int));
	cudaMalloc(&specter_cpd_dev, specter_elements[0] * sizeof(FLOAT));
	cudaMalloc(&specter_energy_dev, specter_elements[0] * sizeof(FLOAT));

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	// memory transfer to device
	cudaMemcpy(source_position_dev, source_position, 3 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(source_direction_dev, source_direction, 3 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(scan_axis_dev, scan_axis, 3 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(sdd_dev, sdd, sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(fov_dev, fov, sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(collimation_dev, collimation, sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(weight_dev, weight, sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(specter_elements_dev, specter_elements, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(specter_cpd_dev, specter_cpd, specter_elements[0] * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy(specter_energy_dev, specter_energy, specter_elements[0] * sizeof(FLOAT), cudaMemcpyHostToDevice);


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
void* setup_source(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *sdd, FLOAT *fov, FLOAT *collimation, FLOAT *weight, FLOAT *specter_cpd, FLOAT *specter_energy, int *specter_elements)
{
	Source *source_dev = (Source*)malloc(sizeof(Source));
	source_dev->source_position = source_position;
	source_dev->source_direction = source_direction;
	source_dev->scan_axis = scan_axis;
	source_dev->sdd = sdd;
	source_dev->fov = fov;
	source_dev->collimation = collimation;
	source_dev->weight = weight;
	source_dev->specter_elements = specter_elements;
	source_dev->specter_cpd = specter_cpd;
	source_dev->specter_energy = specter_energy;
	return (void*)source_dev;
}
void* setup_source_bowtie(FLOAT *source_position, FLOAT *source_direction, FLOAT *scan_axis, FLOAT *scan_axis_fan_angle, FLOAT *rot_axis_fan_angle, FLOAT *weight, FLOAT *specter_cpd, FLOAT *specter_energy, int *specter_elements, FLOAT *bowtie_weight, FLOAT *bowtie_angle, int* bowtie_elements)
{
	SourceBowtie *source_dev = (SourceBowtie*)malloc(sizeof(SourceBowtie));
	source_dev->source_position = source_position;
	source_dev->source_direction = source_direction;
	source_dev->scan_axis = scan_axis;
	source_dev->scan_axis_fan_angle = scan_axis_fan_angle;
	source_dev->rot_axis_fan_angle = rot_axis_fan_angle;
	source_dev->weight = weight;
	source_dev->specter_elements = specter_elements;
	source_dev->specter_cpd = specter_cpd;
	source_dev->specter_energy = specter_energy;
	source_dev->bowtie_elements = bowtie_elements;
	source_dev->bowtie_weight = bowtie_weight;
	source_dev->bowtie_angle = bowtie_angle;
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
void run_simulation(void *dev_source, int64_t n_particles, void *dev_simulation)
{
	// simulating particles

	size_t thread_number;
	size_t n_threads = omp_get_max_threads();

	trackingFuncPtr tracking_func;

	if (((Simulation*)dev_simulation)->use_siddon_pathing[0] == 1)
	{
		tracking_func = &siddon_path;
	}
	else
	{
		tracking_func = &woodcock_step;
	}

	uint64_t *states = (uint64_t*)malloc(2 * n_threads * sizeof(uint64_t));
	init_random_seed(((Simulation*)dev_simulation)->seed, &n_threads, states);

#pragma omp parallel num_threads(n_threads) private(thread_number)
	{
		thread_number = omp_get_thread_num();
		int64_t i;
#pragma omp for
		for (i = 0; i < n_particles; i++)
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
				tracking_func,
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

void run_simulation_bowtie(void *dev_source, int64_t n_particles, void *dev_simulation)
{
	// simulating particles

	size_t thread_number;
	size_t n_threads = omp_get_max_threads();

	uint64_t *states = (uint64_t*)malloc(2 * n_threads * sizeof(uint64_t));

	init_random_seed(((Simulation*)dev_simulation)->seed, &n_threads, states);

	trackingFuncPtr tracking_func;

	if (((Simulation*)dev_simulation)->use_siddon_pathing[0] == 1)
	{
		tracking_func = &siddon_path;
	}
	else
	{
		tracking_func = &woodcock_step;
	}


#pragma omp parallel num_threads(n_threads) private(thread_number)
	{
		thread_number = omp_get_thread_num();
		int64_t i;
#pragma omp for
		for (i = 0; i < n_particles; i++)
		{
			transport_particles_bowtie(
				((SourceBowtie*)dev_source)->source_position,
				((SourceBowtie*)dev_source)->source_direction,
				((SourceBowtie*)dev_source)->scan_axis,
				((SourceBowtie*)dev_source)->scan_axis_fan_angle,
				((SourceBowtie*)dev_source)->rot_axis_fan_angle,
				((SourceBowtie*)dev_source)->weight,
				((SourceBowtie*)dev_source)->specter_elements,
				((SourceBowtie*)dev_source)->specter_cpd,
				((SourceBowtie*)dev_source)->specter_energy,
				((SourceBowtie*)dev_source)->bowtie_elements,
				((SourceBowtie*)dev_source)->bowtie_weight,
				((SourceBowtie*)dev_source)->bowtie_angle,
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
				tracking_func,
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
__host__ void cleanup_simulation(void *dev_simulation, int *shape, FLOAT *energy_imparted)
{
	//struct Simulation *dev_sim = ((struct Simulation*)dev_simulation);
	//copy energy_imparted from device memory to host
	cudaError_t err;
	err = cudaMemcpy(energy_imparted, ((Simulation*)dev_simulation)->energy_imparted, shape[0] * shape[1] * shape[2] * sizeof(FLOAT), cudaMemcpyDeviceToHost);
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
void cleanup_simulation(void *dev_simulation)
{
	// free memory
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




void setup_test_environment(int *shape, FLOAT *spacing, FLOAT *offset, int *material_map, FLOAT *density_map, int *att_shape, FLOAT *attenuation_lut, FLOAT *energy_imparted)
{
	int i, j, k;
	size_t ind;

	//geometry
	for (i = 0; i < 3; i++)
	{
		offset[i] = -(FLOAT)shape[i] * spacing[i] / 2.f;
	}

	//particles

	for (i = 0; i < shape[0]; i++)
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
	int use_siddon = 0;
	size_t n_particles = 500000;
	int shape[3] = { 100, 100, 100 };
	int lut_shape[3] = { 2, 5, 5 };
	FLOAT spacing[3] = { 1, 1, 1 };
	FLOAT offset[3];

	// init geometry variables
	//FLOAT *particles = (FLOAT *)malloc(n_particles * 8 * sizeof(FLOAT));
	int *material_map = (int *)malloc(shape[0] * shape[1] * shape[2] * sizeof(int));
	FLOAT *density_map = (FLOAT *)malloc(shape[0] * shape[1] * shape[2] * sizeof(FLOAT));
	FLOAT *attenuation_lut = (FLOAT *)malloc(lut_shape[0] * lut_shape[1] * lut_shape[2] * sizeof(FLOAT));
	FLOAT *energy_imparted = (FLOAT *)malloc(shape[0] * shape[1] * shape[2] * sizeof(FLOAT));

	// initialazing geometry
	setup_test_environment(shape, spacing, offset, material_map, density_map, lut_shape, attenuation_lut, energy_imparted);


	void* sim;
	sim = setup_simulation(shape, spacing, offset, material_map, density_map, lut_shape, attenuation_lut, energy_imparted, &use_siddon);

	//init source variables
	FLOAT source_position[3] = { -7, 0, 0 };
	FLOAT source_direction[3] = { 1, 1, 0 };
	normalize_3Dvector(source_direction);

	FLOAT scan_axis[3] = { 0, 0, 1 };
	FLOAT sdd = 119;
	FLOAT fov = 0;// 50;
	FLOAT collimation = 0;// 4;
	FLOAT weight = 1;

	FLOAT rot_angle = TAN(fov / sdd) * 2.f;
	FLOAT scan_angle = TAN(collimation / sdd / 2.f) * 2.f;

	FLOAT specter_cpd[3] = { 0.33, 0.66, 1 };
	FLOAT specter_energy[3] = { 60000, 70000, 80000 };
	int specter_elements = 3;


	void* geo;
	geo = setup_source(source_position, source_direction, scan_axis, &sdd, &fov, &collimation, &weight, specter_cpd, specter_energy, &specter_elements);

	FLOAT bowtie_angle[5] = { -rot_angle, -rot_angle / 2.f, 0, rot_angle / 2.f, rot_angle };
	FLOAT bowtie_weights[5] = { .4, .8, 1, .8, .4 };
	int bowtie_size = 5;

	void *geo2 = setup_source_bowtie(source_position, source_direction, scan_axis, &scan_angle, &rot_angle, &weight, specter_cpd, specter_energy, &specter_elements, bowtie_weights, bowtie_angle, &bowtie_size);

	run_simulation(geo, n_particles, sim);
	run_simulation_bowtie(geo2, n_particles, sim);


	cleanup_simulation(sim);
	cleanup_source(geo);

	return 1;

	size_t index;
	FLOAT energy1234 = 0;
	for (int i = 0; i < shape[0]; i++)
	{
		for (int j = 0; j < shape[1]; j++)
		{
			for (int k = 0; k < shape[2]; k++)
			{

				index = shape[1] * shape[2] * i + shape[2] * j + k;
				if (energy_imparted[index] > 0.000000000001)
				{
					energy1234 += energy_imparted[index];
					printf("\n energy in %d %d %d = %f", i, j, k, energy_imparted[index]);
				}
			}
		}
	}
	printf("total energy %f", energy1234);
	return 1;
}
