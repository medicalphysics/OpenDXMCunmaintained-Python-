cdef extern from "sys/time.h" nogil:

    struct timeval:
        int tv_sec  # seconds
        int tv_usec # microseconds

    ctypedef struct timezone:
        pass

    ctypedef long time_t

    struct timespec:
        time_t tv_sec  # seconds
        long   tv_nsec # nanoseconds

    int gettimeofday(timeval *tv, timezone *tz)
    int settimeofday(const timeval *tv, const timezone *tz)

    int nanosleep(const timespec *req, timespec *rem)