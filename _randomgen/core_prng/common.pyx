import numpy as np
cimport numpy as np
from common cimport *

np.import_array()

cdef np.ndarray int_to_array(object value, object name, object bits):
    len = bits // 64
    value = np.asarray(value)
    if value.shape == ():
        value = int(value)
        upper = int(2)**int(bits)
        if value < 0 or value >= upper:
            raise ValueError('{name} must be positive and '
                             'less than 2**{bits}.'.format(name=name, bits=bits))
        out = np.empty(len, dtype=np.uint64)
        for i in range(len):
            out[i] = value % 2**64
            value >>= 64
    else:
        out = value.astype(np.uint64)
        if out.shape != (len,):
            raise ValueError('{name} must have {len} elements when using '
                             'array form'.format(name=name, len=len))
    return out


cdef check_output(object out, object dtype, object size):
    if out is None:
        return
    cdef np.ndarray out_array = <np.ndarray>out
    if not (np.PyArray_CHKFLAGS(out_array, np.NPY_CARRAY) or
            np.PyArray_CHKFLAGS(out_array, np.NPY_FARRAY)):
        raise ValueError('Supplied output array is not contiguous, writable or aligned.')
    if out_array.dtype != dtype:
        raise TypeError('Supplied output array has the wrong type. '
                        'Expected {0}, got {0}'.format(dtype, out_array.dtype))
    if size is not None:
        # TODO: enable this !!! if tuple(size) != out_array.shape:
        raise ValueError('size and out cannot be simultaneously used')


cdef object double_fill(void *func, void *state, object size, object lock, object out):
    cdef random_double_0 random_func = (<random_double_0>func)
    cdef double *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None:
        with lock:
            return random_func(state)

    if out is not None:
        check_output(out, np.float64, size)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.double)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <double *>np.PyArray_DATA(out_array)
    with lock, nogil:
        for i in range(n):
            out_array_data[i] = random_func(state)
    return out_array

cdef object float_fill(void *func, void *state, object size, object lock, object out):
    cdef random_float_0 random_func = (<random_float_0>func)
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None:
        with lock:
            return random_func(state)

    if out is not None:
        check_output(out, np.float32, size)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float32)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <float *>np.PyArray_DATA(out_array)
    with lock, nogil:
        for i in range(n):
            out_array_data[i] = random_func(state)
    return out_array

cdef object float_fill_from_double(void *func, void *state, object size, object lock, object out):
    cdef random_double_0 random_func = (<random_double_0>func)
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None:
        with lock:
            return <float>random_func(state)

    if out is not None:
        check_output(out, np.float32, size)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float32)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <float *>np.PyArray_DATA(out_array)
    with lock, nogil:
        for i in range(n):
            out_array_data[i] = <float>random_func(state)
    return out_array

