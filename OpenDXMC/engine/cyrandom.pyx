include "available_modules.pxi"

IF CYTIME_AVAILABLE == 1:
    from opendxmc.engine.cytime cimport time
    cdef inline unsigned long time_based_seed() nogil:
        return <unsigned long> (time() * 100000.0)
ELSE:
    from time import time
    cdef inline unsigned long time_based_seed() except? 0:
        # on Windows, unsigned long is only 32 bits.  Subtract 30 years
        # of seconds, so this value will fit.
        cdef double time_val = time()
        return <unsigned long> ((time_val - 946080000.0) * 100000.0)

from libc.math cimport log, sqrt

from opendxmc.engine._random cimport (
    genrand_int32, init_genrand, init_by_array, N as mt_N)

from numpy cimport int64_t, uint8_t, uint64_t
import cython
import numpy as np


cdef class Random:
    """Random number generator using Mersenne Twister.

    Parameters
    ----------
    seed : uint64
        Initial seed for the generator.  This parameter is optional.  If not
        provided, the current time will be used.
    """
    def __init__(self, seed=None):
        self._has_next_gauss = False

        if seed is None:
            init_genrand(&self.state, time_based_seed())
        else:
            self.seed(seed)

    def seed_py(self, uint64_t n):
        """Seed the random number generator.

        Parameters
        ----------
        n : uint64
            The seed value.
        """
        self.seed(n)

    cdef void seed(self, uint64_t n) nogil:
        """Seed the random number generator.

        Parameters
        ----------
        n : uint64
            The seed value.
        """
        # this is taken from the random_seed function in Python's
        # _randommodule.c, modified since the input is always a
        # 64 bit integer
        cdef size_t keyused
        cdef unsigned long key[2]

        # key[0] is the low 32 bits, and key[1] is the high 32 bits
        key[0] = n & (<uint64_t> 0xffffffff)
        key[1] = n >> 32
        keyused = 2 if key[1] > 0 else 1
        init_by_array(&self.state, key, keyused)

    def getstate(self):
        """Get the internal state of the RNG.
        """
        cdef Py_ssize_t i
        cdef list state

        version = 3
        state = []
        for i in range(mt_N):
            state.append(self.state.state[i])
        state.append(self.state.index)
        return (version, tuple(state), 0)

    def setstate(self, object state):
        """Set the internal state of the RNG.
        """
        cdef Py_ssize_t i
        cdef int version
        cdef tuple state_var

        # using object instead of tuple eliminates gcc warnings, but requires an
        # explicit type check
        if not isinstance(state, tuple):
            raise TypeError('state must be a tuple')
        version, state_var, _ = state
        if version != 3:
            raise ValueError('only Python 3 currently supported')
        if len(state_var) != mt_N + 1:
            raise ValueError('state tuple must be length {0}'.format(mt_N + 1))
        for i in range(mt_N):
            self.state.state[i] = state_var[i]
        self.state.index = state_var[-1]

    def random_py(self):
        """Return a random floating point value in the range [0, 1).

        Returns
        -------
        output : double
            The random value.
        """
        return self.random()

    @cython.cdivision(True)
    cdef double random(self) nogil:
        """Return a random floating point value in the range [0, 1).

        Returns
        -------
        output : double
            The random value.
        """
        # this is taken from the random_random function in Python's
        # _randommodule.c
        # random is the function named genrand_res53 in the original code;
        # generates a random number on [0,1) with 53-bit resolution; note that
        # 9007199254740992 == 2**53; I assume they're spelling "/2**53" as
        # multiply-by-reciprocal in the (likely vain) hope that the compiler
        # will optimize the division away at compile-time.  67108864 is 2**26.
        # In effect, a contains 27 random bits shifted left 26, and b fills in
        # the lower 26 bits of the 53-bit numerator.
        # The orginal code credited Isaku Wada for this algorithm, 2002/01/09.
        cdef unsigned long a, b
        a = genrand_int32(&self.state) >> 5
        b = genrand_int32(&self.state) >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def getrandbits_py(self, int k, output=None):
        """Generates an array with k random bits.

        Note that only the bytes necessary to store the requested number of
        random bits will be written to output.  If it is larger than what is
        required, those values will not be changed.

        Parameters
        ----------
        k : int
            The number of bits to generate.
        output : uint8 ndarray, optional
            The array where the output should be stored.  This parameter is
            optional.  If not present, a new array will be allocated.

        Returns
        -------
        output : uint8 ndarray
            The output array containing the generated bits.
        """
        cdef size_t nbytes
        cdef int ret

        if output is None:
            nbytes = bits2bytes(k)
            output = np.zeros((nbytes), 'u1')
        ret = self.getrandbits(k, output)
        if ret == -1:
            raise ValueError('number of bits less than zero')
        elif ret == -2:
            raise ValueError('output array too small')
        return output

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int getrandbits(self, int k, uint8_t [::1] output) nogil:
        """Generates k random bits.  The result is stored in the specified
        memoryview.

        Note that only the bytes necessary to store the requested number of
        random bits will be written to output.  If it is larger than what is
        required, those values will not be changed.

        Parameters
        ----------
        k : int
            The number of bits to generate.
        output : 1D uint8 array.
            The memoryview where the output will be stored.

        Returns
        -------
        status : int
            Equal to zero if the call completed successfully, -1 if k was a
            negative number, and -2 if the output buffer was too small.
        """
        return self._getrandbits(k, &output[0], output.shape[0])

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.final
    @cython.wraparound(False)
    cdef int _getrandbits(self, int k, uint8_t *output, size_t length) nogil:
        """Pointer-based version of getrandbits.  This allows calling functions
        to use nogil.
        """
        # this is taken from the random_getrandbits function in Python's
        # _randommodule.c
        cdef int i
        cdef size_t nbytes
        cdef unsigned long r

        if k < 0:
            return -1
        nbytes = bits2bytes(k)
        if nbytes > length:
            return -2
        for i in range(0, nbytes, 4):
            r = genrand_int32(&self.state)
            if k < 32:
                r >>= (32 - k)
            output[i+0] = <uint8_t> r
            output[i+1] = <uint8_t> (r >> 8)
            output[i+2] = <uint8_t> (r >> 16)
            output[i+3] = <uint8_t> (r >> 24)
            k -= 32
        return 0

    cdef uint64_t _randbelow(self, uint64_t n) nogil:
        """Return a random int in the range [0,n).
        """
        cdef int bits
        cdef size_t nbytes
        cdef uint64_t r = 0

        bits = nbits(n)
        nbytes = bits2bytes(bits)
        self._getrandbits(bits, <uint8_t*> &r, nbytes)
        while r >= n:
            r = 0
            self._getrandbits(bits, <uint8_t*> &r, nbytes)
        return r

    def randrange_py(self, start, stop, step):
        """Choose a random item from range(start, stop, step).

        This fixes the problem with randint() which includes the endpoint; in
        Python this is usually not what you want.

        Parameters
        ----------
        start : int64
            The start of the output range.
        stop : int64
            The end of the input range.
        step : int64
            The step size.

        Returns
        -------
        output : int64
            The random value.
        """
        cdef int ret
        cdef int64_t output

        ret = self.randrange(start, stop, step, &output)
        if ret == -1:
            msg = 'zero step size: ({0}, {1}, {2})'
            raise ValueError(msg.format(start, stop, step))
        elif ret == -2:
            msg = 'empty range:  ({0}, {1}, {2})'
            raise ValueError(msg.format(start, stop, step))
        return output

    @cython.cdivision(True)
    cdef int randrange(self, int64_t start, int64_t stop, int64_t step,
                       int64_t *output) nogil:
        """Choose a random item from range(start, stop, step).

        This fixes the problem with randint() which includes the endpoint; in
        Python this is usually not what you want.

        Parameters
        ----------
        start : int64
            The start of the output range.
        stop : int64
            The end of the input range.
        step : int64
            The step size.
        output : pointer to int64
            A pointer where the output will be written.

        Returns
        -------
        status : int
            Equal to 0 if no errors occurred, -1 if the step size is zero, or
            -2 if an empty range is defined.
        """
        cdef int64_t width, n

        width = stop - start
        if step > 0:
            n = (width + step - 1) / step
        elif step < 0:
            n = (width + step + 1) / step
        else:
            return -1
        if n <= 0:
            return -2
        output[0] = start + step * <int64_t> self._randbelow(n)
        return 0

    def randint_py(self, a, b):
        """Return random integer in range [a, b], including both end points.

        Parameters
        ----------
        a : int64
            The lower bound.
        b : int64
            The upper bound

        Returns
        -------
        output : int64
            The random value.
        """
        cdef int ret
        cdef int64_t output

        ret = self.randint(a, b, &output)
        if ret == -1:
            msg = 'zero step size: ({0}, {1})'
            raise ValueError(msg.format(a, b))
        elif ret == -2:
            msg = 'empty range:  ({0}, {1})'
            raise ValueError(msg.format(a, b))
        return output

    cdef int randint(self, int64_t a, int64_t b, int64_t *output) nogil:
        """Return random integer in range [a, b], including both end points.

        Parameters
        ----------
        a : int64
            The lower bound.
        b : int64
            The upper bound
        output : pointer to int64
            Pointer where the output will be stored

        Returns
        -------
        status : int
            Equal to 0 if no errors occurred, -1 if the step size is zero, or
            -2 if an empty range is defined.
        """
        return self.randrange(a, b + 1, 1, output)

    def uniform_py(self, double a, double b):
        """Return a random double precision floating point number in the
        range [a, b].

        Parameters
        ----------
        a : double
            The lower bound.
        b : double
            The upper bound

        Returns
        -------
        output : double
            The random value.
        """
        cdef double output
        self.uniform(a, b, &output)
        return output

    cdef void uniform(self, double a, double b, double *output) nogil:
        """Return a random double precision floating point number in the
        range [a, b].

        Parameters
        ----------
        a : double
            The lower bound.
        b : double
            The upper bound.
        output : pointer to double
            Pointer where the output will be stored.
        """
        output[0] = a + (b - a) * self.random()

    def standard_normal_py(self):
        """Return a random double precision floating point number drawn from a
        normal (Gaussian) distribution with a mean of zero and a standard
        deviation of one.

        Returns
        -------
        output : double
            The random value.
        """
        cdef double output
        self.standard_normal(&output)
        return output

    @cython.cdivision(True)
    cdef void standard_normal(self, double *output) nogil:
        """Return a random double precision floating point number drawn from a
        normal (Gaussian) distribution with a mean of zero and a standard
        deviation of one.

        Parameters
        ----------
        output : pointer to double
            Pointer where the output will be stored.
        """
        cdef double u = 0.0, v = 0.0, s = 2.0, x

        if self._has_next_gauss:
            self._has_next_gauss = False
            output[0] = self._next_gauss
            return
        # Marsaglia polar method
        # u and v are uniformly distributed on (-1, 1)
        while s == 0.0 or s >= 1.0:
            u = 2.0 * self.random() - 1.0
            v = 2.0 * self.random() - 1.0
            s = u * u + v * v
        x = sqrt(-2.0 / s * log(s))
        self._has_next_gauss = True
        self._next_gauss = u * x
        output[0] = v * x

    def normal_py(self, double loc, double scale):
        """Return a random double precision floating point number drawn from a
        normal (Gaussian) distribution.

        Parameters
        ----------
        loc : double
            The mean of the distribution.
        scale : double
            The standard deviation of the distribution.

        Returns
        -------
        output : double
            The random value.
        """
        cdef double output

        self.normal(loc, scale, &output)
        return output

    cdef void normal(self, double loc, double scale, double *output) nogil:
        """Return a random double precision floating point number drawn from a
        normal (Gaussian) distribution.

        Parameters
        ----------
        loc : double
            The mean of the distribution.
        scale : double
            The standard deviation of the distribution.
        output : pointer to double
            A pointer where the output will be stored
        """
        cdef double standard_normal
        self.standard_normal(&standard_normal)
        output[0] = loc + scale * standard_normal

    def rayleigh_py(self, double scale):
        """Returns a random double precision floating point number drawn from
        a Rayleigh distribution with the specified scale parameter.

        Parameters
        ----------
        scale : double
            The scale parameter of the distribution.

        Returns
        -------
        output : double
            The random value.
        """
        cdef int ret
        cdef double output

        ret = self.rayleigh(scale, &output)
        if ret == -1:
            raise ValueError('scale must be greater than zero')
        return output

    cdef int rayleigh(self, double scale, double *output) nogil:
        """Returns a random double precision floating point number drawn from
        a Rayleigh distribution with the specified scale parameter.

        Parameters
        ----------
        scale : double
            The scale parameter of the distribution.
        output : pointer to double
            A pointer where the result will be stored.

        Returns
        -------
        status : int
            Equal to zero if no error occurred, or -1 if the scale parameter is
            not positive.
        """
        if scale <= 0.0:
            return -1
        # self.random() returns a number in range [0, 1)
        # use (1.0 - self.random()) to change it to (0, 1], since log(0) is
        # undefined
        output[0] = scale * sqrt(-2.0 * log(1.0 - self.random()))
        return 0


def py_nbits(n):
    """Calculate the min number of bits to store n.
    """
    return nbits(n)


cdef inline int nbits(uint64_t n) nogil:
    """Calculate the min number of bits to store n.
    """
    # can be replaced with __builtin_clzl on GCC or _BitScanReverse64 on MSVC
    cdef int r = 0
    while n:
      r += 1
      n >>= 1
    return r


@cython.cdivision(True)
cdef inline size_t bits2bytes(int bits) nogil:
    """Convert a number of bits into an equivalent number of bytes.
    """
    return (((bits - 1) / 32) + 1) * 4
