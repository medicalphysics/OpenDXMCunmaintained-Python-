# -*- coding: utf-8 -*-
#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: nonecheck=False
import numpy as np


cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

from cython.parallel import prange
#from cython cimport openmp
cimport _siddon_func
#import random
from cyrandom cimport Random
cdef Random random
random = Random()


cdef extern from "math.h":
    double sin(double) nogil
    double cos(double) nogil
    double fabs(double) nogil
    double floor(double) nogil
    double ceil(double) nogil
    double sqrt(double) nogil
    double exp(double) nogil
    double log(double) nogil
    double asin(double) nogil
    double acos(double) nogil
    int isnan(double x) nogil

cdef double ERRF = 1.e-9

cdef double ELECTRON_MASS = 510998.9  # eV/c^2
cdef double PI = 3.14159265359
cdef double ENERGY_CUTOFF = 10.e3 # eV
cdef double ENERGY_MAXVAL = 300.e3 # eV
cdef double WEIGHT_CUTOFF = 0.01


cdef inline double sign(double a) nogil: return -1 if a > 0 else 1
cdef inline double interpolate(double x, double x1, double x2, double y1, double y2) nogil: return y1 + (y2-y1) * (x - x1) / (x2 - x1)


cdef int argmin(double* u) nogil:
    cdef double v0, v1, v2
    v0 = fabs(u[0])
    v1 = fabs(u[1])
    v2 = fabs(u[2])
    if (v0 <= v1) and (v0 <= v2):
        return <int>0
    if (v1 <= v0) and (v1 <= v2):
        return <int>1
    if (v2 <= v0) and (v2 <= v1):
        return <int>2

cdef void normalize_vector(double* v) nogil:
    cdef int i
    cdef double s = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    for i in range(3):
        v[i] /= s
#    return <int>1

cdef void normal_vector(double* v, double* u) nogil:
    cdef int i  = argmin(v)

    if i == 0:
        u[0] = 0
        u[1] = v[2]
        u[2] = -v[1]
    elif i ==1:
        u[0] = v[2]
        u[i] = 0
        u[2] = -v[0]
    elif i ==2:
        u[0] = v[1]
        u[1] = -v[0]
        u[2] = 0
    normalize_vector(u)


cdef void full_basis(double* v, double* u, double* n) nogil:
    n[0] = v[1]*u[2] - v[2]*u[1]
    n[1] = -v[0]*u[2] + v[2]*u[0]
    n[2] = v[0]*u[1] - v[1]*u[0]


cdef void rot_particle(double* particle, double theta) nogil:
    cdef double* u = <double *>malloc(3 * sizeof(double))
    cdef double* v = <double *>malloc(3 * sizeof(double))
    cdef double* w = <double *>malloc(3 * sizeof(double))

    cdef double r1 = random.random() * 2. * PI
    cdef int i
    for i in range(3):
        v[i] = particle[i+3]
    normal_vector(v, u)
    full_basis(v, u, w)

    cdef double c_theta, s_theta, c_r1, s_r1
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_r1 = cos(r1)
    s_r1 = sin(r1)

    for i in range(3):
        particle[i+3] = c_theta * v[i] + s_theta * (s_r1 * u[i] + c_r1 * w[i])
    free(u)
    free(v)
    free(w)
    return

cdef double att_linear(double[:,:,:] atts, int tissue_ind, int interaction, double energy) nogil:
    cdef double x1, y1, x2, y2
    cdef int i, N
    N = <int>atts.shape[2]

    for i in range(1, N):
        if energy < atts[tissue_ind, 0, i]:
            y1 = atts[tissue_ind, interaction, i-1]
            y2 = atts[tissue_ind, interaction, i]
            x1 = atts[tissue_ind, 0, i-1]
            x2 = atts[tissue_ind, 0, i]
            return y1 + (y2-y1) * (energy - x1) / (x2 - x1)
    return atts[tissue_ind, interaction, N-1]


cdef double compton_event_draw_energy_theta(double energy, double* theta) nogil:
    """Draws scattered energy and angle, based on Geant4 implementation
    returns tuple (scatter angle, energy)
    """
    cdef double epsilon_0, alpha1, alpha2, r1, r2, r3, epsilon, qsin_theta, t, k
    k = energy / ELECTRON_MASS
    epsilon_0 = 1. / (1. + 2. * k)


    alpha1 = log(1. / epsilon_0)
    alpha2 = (1. - epsilon_0**2) / 2.
    while True:
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()

        if r1 < alpha1 / (alpha1 + alpha2):
            epsilon = exp(-r2 * alpha1)
        else:
            epsilon = sqrt((epsilon_0**2 + (1. - epsilon_0**2) * r2))

        t = ELECTRON_MASS * (1. - epsilon) / (energy * epsilon)
        qsin_theta = t * (2. - t)

        if (1. - epsilon / (1. + epsilon**2) * qsin_theta) >= r3:
            break
    theta[0] = acos(1. + 1./k - 1./epsilon/k)
    return epsilon * energy



cdef double rayleigh_event_draw_theta() nogil:
    cdef double r, c, A
    r = random.random()
    c = 4. - 8. * r
    A = -sign(c) * ((fabs(c) + (c**2 + 4.)**.5) / 2.)**(1./3.)
    return acos(A - 1. / A)


cdef void cumulative_interaction_prob(int* ind, double* lenghts, int N, int[:,:,:] material_map, double[:,:,:] density_map, double[:,:,:] attinuation_lut, double energy, double* cum_prob) nogil:
    cdef double att, cum_sum
    cdef int e_index, N_energy, material, i
    N_energy = <int>attinuation_lut.shape[2]
    #finding upper energy
    for e_index in range(N_energy):
        if energy < attinuation_lut[0, 0, e_index]:
            break
    else:
        for i in range(N):
            cum_prob[i] = 0
        return
    cum_sum = 0
    for i in range(N):
        material = material_map[ind[3 * i], ind[3 * i+1], ind[3 * i+2]]
        att = interpolate(energy, attinuation_lut[material, 0, e_index -1], attinuation_lut[material, 0, e_index], attinuation_lut[material, 1, e_index -1], attinuation_lut[material, 1, e_index])
        cum_sum += att * density_map[ind[3 * i], ind[3*i+1], ind[3 * i+2]] * lenghts[i]
        cum_prob[i] = 1 - exp(-cum_sum)


cdef void interaction_point(double* particle, double[:] spacing, double[:] offset, int* ind, double* lenghts, int N, double* weight, int* index, int[:,:,:] material_map, double[:,:,:] density_map, double[:,:,:] attinuation_lut, double* stop) nogil:

    cdef int i, j
    cdef double att, cum_sum, r1, delta_r, dist

    cdef double* u = <double*> malloc(N*sizeof(double))
    cdef double* cum_prob = <double*> malloc(N*sizeof(double))

#    cumulative_interaction_prob(ind, lenghts, N, material_map, density_map, attinuation_lut, particle[6], cum_prob)
    cum_sum = 0
    for i in range(N):
        att = att_linear(attinuation_lut, material_map[ind[3*i], ind[3*i+1], ind[3*i+2]], 1, particle[6])
        u[i] = att * density_map[ind[3*i], ind[3*i+1], ind[3*i+2]]
        cum_sum += u[i] * lenghts[i]
        cum_prob[i] = 1 - exp(-cum_sum)

    #test for zero prob
    if cum_prob[N-1] < ERRF:
        for j in range(3):
            stop[j] = 0
        index[0] = <int>-1
        weight[0] = <int>0
        free(u)
        free(cum_prob)
        return

    weight[0] = <int>1

    r1 = random.random()

    for i in range(N):
        if r1 < cum_prob[i]:
            if i > 0:
                delta_r = (r1 - cum_prob[i - 1])
                dist = delta_r / (cum_prob[i] - cum_prob[i - 1]) * lenghts[i]
            else:
                delta_r = r1
                dist = delta_r / cum_prob[i] * lenghts[i]

            index[0] = <int>i
            for j in range(3):
                stop[j] = _siddon_func.plane(spacing, offset, j, ind[i*3+j]) + dist * particle[j+3]
#            print [_siddon_func.plane(spacing, offset, j, i) for j in range(3)], dist, cum_prob
            free(u)
            free(cum_prob)
            return

    for j in range(3):
        stop[j] = 0
    index[0] = <int>-1
    weight[0] = <int>0
    free(u)
    free(cum_prob)
    return



cdef void interaction_point_forced(double* particle, double[:] spacing, double[:] offset, int* ind, double* lenghts, int N, double* weight, int* index, int[:,:,:] material_map, double[:,:,:] density_map, double[:,:,:] attinuation_lut, double* stop) nogil:

    cdef int i, j
    cdef double att, cum_sum, r1, delta_r, dist
#    cdef np.ndarray[np.double_t, ndim=1] u, cum_prob
    cdef double* u = <double*> malloc(N * sizeof(double))
    cdef double* cum_prob = <double*> malloc(N * sizeof(double))

    cum_sum = 0
    for i in range(N):
        att = att_linear(attinuation_lut,material_map[ind[3 * i], ind[3 * i+1], ind[3 * i+2]], 1, particle[6])
        u[i] = att * density_map[ind[3 * i], ind[3*i+1], ind[3 * i+2]]
        cum_sum += u[i] * lenghts[i]
        cum_prob[i] = 1 - exp(-cum_sum)
#    cumulative_interaction_prob(ind, lenghts, N, material_map, density_map, attinuation_lut, particle[6], cum_prob)



    #test for zero prob
    if cum_prob[N-1] < ERRF:
        for j in range(3):
            stop[j] = 0
        index[0] = -1
        weight[0] = 0
        free(u)
        free(cum_prob)
        return

    r1 = random.random()
    for i in range(N):
        if r1 < cum_prob[i] / cum_prob[N-1]:
            if i > 0:
                delta_r = (r1 - cum_prob[i - 1])
                dist = delta_r / (cum_prob[i] - cum_prob[i - 1]) * lenghts[i]
            else:
                delta_r = r1
                dist = delta_r / cum_prob[i] * lenghts[i]
            break

    index[0] = 3*i

    for j in range(3):
        stop[j] = _siddon_func.plane(spacing, offset, j, ind[i*3 + j]) + dist * particle[j+3]
    weight[0] = particle[7] * cum_prob[N-1]
    free(u)
    free(cum_prob)
#    print [_siddon_func.plane(spacing, offset, j, i) for j in range(3)], dist,  cum_prob
    return


cdef void transport_particle(double[:,:] particles, long particle_index, double[:] N, double[:] spacing, double[:] offset, int[:,:,:] material_map, double[:,:,:] density_map, double[:,:,:] attinuation_lut, double[:,:,:] dose) nogil:


    cdef double weight, dist, r1, azimutal_angle
    cdef double compton, rayleigh, photo, total, scatter_angle, scatter_energy
    cdef int valid, force_interaction, index, material, i, n_indices, n_max
    n_max = <int> N[0]
    if n_max > <int>N[1]:
        n_max = <int> N[1]
    if n_max < <int> N[2]:
        n_max = <int> N[2]

    cdef double* particle = <double*>malloc(8*sizeof(double))
    for i in range(8):
        particle[i] = particles[i, particle_index]



    cdef double* stop = <double*> malloc(3*sizeof(double))

    cdef double* weight_p = <double*> malloc(sizeof(double))
    cdef int* index_p = <int*> malloc(sizeof(int))

    cdef double* l =<double*> malloc(n_max*n_max*sizeof(double))
    cdef int* ind=<int*> malloc(n_max*3*n_max*sizeof(int))

    force_interaction = 1
    valid = _siddon_func.is_intersecting(particle, N, spacing, offset)
    while valid == 1:

        if particle[6] < ENERGY_CUTOFF:
            break

        n_indices = _siddon_func.array_indices(particle, N, spacing, offset, &ind, &l)

        #############MEMORY LEAK
        if force_interaction == 1:
#            interaction_point_forced(particle, spacing, offset, ind, l, n_indices, &weight, &index, material_map, density_map, attinuation_lut, stop)
            interaction_point_forced(particle, spacing, offset, ind, l, n_indices, weight_p, index_p, material_map, density_map, attinuation_lut, stop)
        else:
#            interaction_point(particle, spacing, offset, ind, l, n_indices, &weight, &index, material_map, density_map, attinuation_lut, stop)
            interaction_point(particle, spacing, offset, ind, l, n_indices, weight_p, index_p, material_map, density_map, attinuation_lut, stop)
        index = index_p[0]
        weight = weight_p[0]

        if index < 0:
            break
        #############MEMORY LEAK END

        material = material_map[ind[index], ind[index + 1], ind[index+2]]
        compton = att_linear(attinuation_lut, material, 4, particle[6])
        photo = att_linear(attinuation_lut, material, 3, particle[6])
        rayleigh = att_linear(attinuation_lut, material, 2, particle[6])
        total = compton + rayleigh + photo
#
        r1 = random.random() * total

        if r1 <= photo:
            dose[ind[index], ind[index + 1], ind[index+2]] +=  weight * particle[6]
            valid = 0

        elif r1 <= (compton + photo):

            scatter_energy = compton_event_draw_energy_theta(particle[6], &scatter_angle)
            azimutal_angle = random.random() * PI * 2.
            particle[7] = weight
            dose[ind[index], ind[index + 1], ind[index+2]] +=  weight * (particle[6] - scatter_energy)
            for i in range(3):
                particle[i] = stop[i]
            particle[6] = scatter_energy
            rot_particle(particle, scatter_angle)
            valid = _siddon_func.is_intersecting(particle, N, spacing, offset)

        else:

            scatter_angle = rayleigh_event_draw_theta()
            azimutal_angle = random.random() * PI * 2.
            for i in range(3):
                particle[i] = stop[i]
            particle[7] = weight
            rot_particle(particle, scatter_angle)
            valid = _siddon_func.is_intersecting(particle, N, spacing, offset)

        if weight < WEIGHT_CUTOFF:
            valid = 0
#        force_interaction = 0

    free(l)
    free(ind)
    free(stop)
    free(particle)
    free(index_p)
    free(weight_p)
    return

def score_energy(double[:,:] particles, double[:] N, double[:] spacing, double[:] offset, int[:,:,:] material_map, double[:,:,:] density_map, double[:,:,:] attinuation_lut, double[:,:,:] dose):
    cdef long i
#    cdef double[:] particle = np.empty(8, dtype=np.double)
#    cdef openmp.omp_lock_t lock
#    cdef np.ndarray[np.double_t, ndim=1] particle
    for i in prange(particles.shape[1], schedule='static', nogil=True):
#    for i in range(particles.shape[1]):
#        particle = particles[:, i]
#        particle = particles[:,i]
#        particle = np.empty(8, dtype=np.double)
#        print ''
        transport_particle(particles, i, N, spacing, offset, material_map, density_map, attinuation_lut, dose)
    return


