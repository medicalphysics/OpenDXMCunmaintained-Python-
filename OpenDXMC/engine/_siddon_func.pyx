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
from libc.stdlib cimport malloc, free, realloc

cdef double ERRF = 1.e-9


cdef inline double max(double a, double b) nogil: return a if a >= b else b
cdef inline double min(double a, double b) nogil: return a if a <= b else b


cdef extern from "math.h":
    double sin(double) nogil
    double fabs(double) nogil
    double floor(double) nogil
    double ceil(double) nogil
    double sqrt(double) nogil
    int abs(int) nogil


cdef void bubble_sort(double* arr, int n) nogil:
    cdef int i, j
    for i in range(n):
        for j in range(1, n):
            if arr[j] < arr[j-1]:
                arr[j-1], arr[j] = arr[j], arr[j-1]

cdef int unique_array(double* ain, double* aout, int n) nogil:
    cdef int i, k
    bubble_sort(ain, n)
    aout[0] = ain[0]
    k = 1
    for i in range(1, n):
        if fabs(aout[k-1] - ain[i]) > ERRF:
            aout[k] = ain[i]
            k += 1
    return k - 1




cdef int merge_array(double* arr1, double* arr2, double* out, int n1, int n2, int n3) nogil:
    bubble_sort(arr1, n1)
    bubble_sort(arr2, n2)

    cdef int i, j, k
    i = 0
    j = 0
    k = 0
    while (i != n1) and (j != n2):
        if fabs(arr1[i] - arr2[j]) > ERRF:
            if arr1[i] < arr2[j]:
                out[k] = arr1[i]
                k += 1
                i += 1
            else:
                out[k] = arr2[j]
                k += 1
                j += 1
        else:
            out[k] = arr1[i]
            k += 1
            i += 1
            j += 1

    if i == n1:
        while j != n2:
            out[k] = arr2[j]
            j += 1
            k += 1
    else:
        while i != n1:
            out[k] = arr1[i]
            i += 1
            k += 1
    return k


cdef double plane(double[:] spacing, double[:] offset, int axis, int plane_number) nogil:
    return offset[axis] + plane_number * spacing[axis]


cdef int point_inside(double* particle, double t, double [:] N, double [:] spacing, double [:] offset) nogil:
    cdef int i
    cdef double stop, llim, ulim
    for i in xrange(3):
        ulim = offset[i] + N[i] * spacing[i] + ERRF
        llim = offset[i] - ERRF
        stop = particle[i] + t * particle[i+3]
        if not  (llim <= stop <= ulim):
            return 0
    return 1


cdef void get_stop(double* particle, double [:] N, double [:] spacing, double [:] offset, double* s) nogil:
    cdef double t_max, t_cand
    cdef int i
#    cdef double[:] s
#    s = np.empty(3, dtype=np.double)
    t_max = 0.0

    for i in range(3):
        if fabs(particle[i+3]) > ERRF:
            t_cand = (offset[i] - particle[i]) / particle[i+3]
            if point_inside(particle, t_cand, N, spacing, offset) == 1:
                t_max = max(t_max, t_cand)
            t_cand = (offset[i] + N[i] * spacing[i] - particle[i]) / particle[i+3]
            if point_inside(particle, t_cand, N, spacing, offset) == 1:
                t_max = max(t_max, t_cand)
    for i in range(3):
        s[i] = particle[i] + particle[i+3] * t_max
#    return s


cdef int is_intersecting(double* particle, double [:] N, double [:] spacing, double [:] offset) nogil:

    cdef double *stop = <double *>malloc(3 * sizeof(double))
    cdef int i

    get_stop(particle, N, spacing, offset, stop)

    for i in range(3):
        if fabs(particle[i] - stop[i]) > ERRF:
            free(stop)
            return 1
    free(stop)
    return 0


cdef double alpha(double* particle, double* stop, int plane, int axis, double [:] spacing, double [:] offset) nogil:
    return (offset[axis] + spacing[axis] * plane - particle[axis]) / (stop[axis] - particle[axis])


cdef double phi(double* particle, double* stop, double alpha, int axis, double [:] spacing, double [:] offset) nogil:
    return (particle[axis] + alpha * (stop[axis] - particle[axis]) - offset[axis]) / spacing[axis]


cdef int array_indices(double* particle, double [:] N, double [:] spacing, double [:] offset, int** indices, double** lenght) nogil:

    cdef double *stop = <double *>malloc(3 * sizeof(double))
    get_stop(particle, N, spacing, offset, stop)

#    cdef np.ndarray[np.double_t, ndim=2] a_e = np.empty((3, 2), dtype=np.double)
    cdef double *a_e = <double *>malloc(6 * sizeof(double))
#    cdef np.ndarray[np.int_t, ndim=2] ind_e = np.empty((3, 2), dtype=np.int)
    cdef int *ind_e = <int *>malloc(6 * sizeof(int))
#    cdef np.ndarray[np.int_t, ndim=1] ind = np.empty(3, dtype=np.int)
    cdef int *ind = <int *>malloc(3 * sizeof(int))
    cdef int i

    #finding valid dimensions
    for i in range(3):
        if fabs(particle[i] - stop[i]) > ERRF:
            ind[i] = 1
        else:
            ind[i] = 0

    #finding a_e
    cdef double a0, aN
    for i in range(3):
        if ind[i] == 1:
            a0 = alpha(particle, stop, 0, i, spacing, offset)
            aN = alpha(particle, stop, <int>N[i], i, spacing, offset)
            a_e[i] = min(a0, aN)
            a_e[i+3] = max(a0, aN)

    #finding extreme a
    cdef double amin, amax
    amin = 0.
    amax = 1.
    for i in range(3):
        if ind[i] == 1:
            amin = max(amin, a_e[i])
            amax = min(amax, a_e[i+3])

    #finding extreme indices:
    cdef int n = 0
    for i in range(3):
        if ind[i] == 1:
            if particle[i] < stop[i]:
                if amin == a_e[i]:
                    ind_e[i] = 0
                else:
                    ind_e[i] = <int>max(0, ceil(phi(particle, stop, amin, i, spacing, offset)))
                if amax == a_e[i+3]:
                    ind_e[i+3] = <int>N[i] - 1
                else:
                    ind_e[i+3] = <int>min(floor(phi(particle, stop, amax, i, spacing, offset)), N[i] - 1)
            else:
                if amin == a_e[i]:
                    ind_e[i+3] = <int>N[i] - 1
                else:
                    ind_e[i+3] = <int>min(floor(phi(particle, stop, amin, i, spacing, offset)), N[i] - 1)
                if amax == a_e[i+3]:
                    ind_e[i] = 0
                else:
                    ind_e[i] = <int>max(0, ceil(phi(particle, stop, amax, i, spacing, offset)))
            if ind_e[i+3] >= ind_e[i]:
                n += (ind_e[i+3] - ind_e[i] + 1)

    #finding a intersection values
#    cdef np.ndarray[np.double_t, ndim=1] a_ind = np.zeros(n+2, dtype=np.double)
    cdef double *a_ind = <double *>malloc((n+2) * sizeof(double))
    cdef double *a_xy = <double *>malloc((n+2) * sizeof(double))
    cdef int j, k
    k = 0
    for i in range(3):
        if ind[i] == 1:
            for j in range(ind_e[i], ind_e[i+3] + 1):
                a_ind[k] = alpha(particle, stop, j, i, spacing, offset)
                k += 1
    a_ind[k] = amin
    a_ind[k+1] = amax


#    cdef np.ndarray[np.int_t, ndim=2] indices


    cdef double a_red
    cdef double distance = 0
    #finding euclidian distance between start stop
    for i in range(3):
        distance += (stop[i]- particle[i]) * (stop[i]- particle[i])
    distance = sqrt(distance)
    #unique intersections
    j = unique_array(a_ind, a_xy, n+2)


#    if (indices is  NULL):
#    indices[0] = <int *>malloc(3 * j * sizeof(int)) 
#
#    if lenght is NULL:
#    lenght[0] = <double *>malloc(j * sizeof(double))

    cdef int l = 0
    for i in range(j):
        a_red = (a_xy[i + 1] + a_xy[i]) / 2.
        lenght[0][i] = <double>((a_xy[i + 1] - a_xy[i]) * distance)
        for k in range(3):
            indices[0][l] = <int>(floor(phi(particle, stop, a_red, k, spacing, offset)))
            l += 1
    free(stop)
    free(ind)
    free(a_e)
    free(ind_e)
    free(a_ind)
    free(a_xy)
    return j

def is_intersecting_py(particle, N, spacing, offset):
    for arg in [particle, N, spacing, offset]:
        assert arg.dtype == np.double
        assert arg is not None
    for arg in [N, spacing, offset]:
        assert arg.shape[0] == 3
    assert particle.shape[0] == 8

    cdef int i, r
    cdef double* particle_p = <double*>malloc(8*sizeof(double))
    for i in range(8):
        particle_p[i] = particle[i]
    r = is_intersecting(particle_p, N, spacing, offset)
    free(particle_p)
    return r

def array_indices_py(particle, N, spacing, offset):

    for arg in [particle, N, spacing, offset]:
        assert arg.dtype == np.double
        assert arg is not None
    for arg in [N, spacing, offset]:
        assert arg.shape[0] == 3
    assert particle.shape[0] == 8

    cdef int n_max = <int>N.max()
    cdef double* lenght = <double*>malloc(n_max*n_max *sizeof(double))
    cdef int* ind=<int*>malloc(3*n_max*n_max *sizeof(int))
    
    cdef int n, i, j
    cdef double* particle_p = <double*>malloc(8*sizeof(double))
    for i in range(8):
        particle_p[i] = particle[i]
    n = array_indices(particle_p, N, spacing, offset, &ind, &lenght)
#    print sizeof(lenght[0])
#    print sizeof(double)
#    cdef int n = <int> (sizeof(lenght)/sizeof(double))

    cdef np.ndarray[np.int_t, ndim=2] ind_np = np.empty((3, n), dtype=np.int)
    cdef np.ndarray[np.double_t, ndim=1] lenght_np = np.empty(n, dtype=np.double)
    for i in range(n):
        lenght_np[i] = lenght[i]
        for j in range(3):
            ind_np[j,i] = ind[i*3 + j]

    free(ind)
    free(lenght)
    free(particle_p)
    return ind_np, lenght_np








