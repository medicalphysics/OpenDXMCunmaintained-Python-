
cdef extern from "_random.c" nogil:

    ctypedef struct RandomObject:
        unsigned long *state
        int index

    cdef int N

    unsigned long genrand_int32(RandomObject *self)
    void init_genrand(RandomObject *self, unsigned long s)
    void init_by_array(RandomObject *self, const unsigned long *init_key,
                       size_t key_length)
