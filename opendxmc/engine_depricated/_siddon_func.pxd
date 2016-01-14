# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:56:42 2015

@author: erlean
"""
#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: nonecheck=False
cdef double plane(double[:], double[:], int, int) nogil
cdef int is_intersecting(double*, double [:], double [:], double [:]) nogil
cdef int array_indices(double*, double [:], double [:], double [:], int**, double**) nogil
