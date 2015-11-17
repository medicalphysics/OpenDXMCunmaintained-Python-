from numpy cimport int64_t, uint8_t, uint64_t

from opendxmc.engine._random cimport RandomObject


cdef class Random:
    cdef RandomObject state

    cdef double _next_gauss
    cdef bint _has_next_gauss

    cdef void seed(self, uint64_t n) nogil
    cdef double random(self) nogil
    cdef int getrandbits(self, int k, uint8_t [::1] output) nogil
    cdef int _getrandbits(self, int k, uint8_t *output, size_t length) nogil
    cdef uint64_t _randbelow(self, uint64_t n) nogil
    cdef int randrange(self, int64_t start, int64_t stop, int64_t step,
                       int64_t *output) nogil
    cdef int randint(self, int64_t a, int64_t b, int64_t *output) nogil
    cdef void uniform(self, double a, double b, double *output) nogil
    cdef void standard_normal(self, double *output) nogil
    cdef void normal(self, double loc, double scale, double *output) nogil
    cdef int rayleigh(self, double scale, double *output) nogil