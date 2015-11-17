from ._time cimport (
    time_t, timeval, timespec, gettimeofday, nanosleep)


def py_sleep(double hold_time):
    """Sleep for the specified number of seconds.

    Parameters
    ----------
    hold_time : double
        The time to sleep in seconds.
    """
    with nogil:
        sleep(hold_time)


cdef void sleep(double hold_time) nogil:
    """Sleep for the specified number of seconds.

    Parameters
    ----------
    hold_time : double
        The time to sleep in seconds.
    """
    cdef timespec sleep_time
    cdef time_t tv_sec
    cdef long tv_nsec

    tv_sec = <time_t> hold_time
    tv_nsec = <long> ((hold_time - <double> tv_sec) * 1000000000.0)
    sleep_time.tv_sec = tv_sec
    sleep_time.tv_nsec = tv_nsec
    nanosleep(&sleep_time, NULL)


def py_time():
    """Get the current time as a floating point value.

    Returns
    -------
    time : double
        The current time in seconds.
    """
    return time()


cdef double time() nogil:
    """Get the current time as a floating point value.

    Returns
    -------
    time : double
        The current time in seconds.
    """
    cdef timeval tval
    cdef double result

    gettimeofday(&tval, NULL)
    result = <double> tval.tv_sec
    result += (<double> tval.tv_usec) / 1000000.0
    return result
