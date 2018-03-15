#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True

from collections import namedtuple
from cpython cimport PyInt_AsLong, PyFloat_AsDouble
import sys
import numpy as np
cimport numpy as np
from common cimport *

np.import_array()

interface = namedtuple('interface', ['state_address', 'state', 'next_uint64',
                                     'next_uint32', 'next_double', 'brng'])


cdef double kahan_sum(double *darr, np.npy_intp n):
    cdef double c, y, t, sum
    cdef np.npy_intp i
    sum = darr[0]
    c = 0.0
    for i in range(1, n):
        y = darr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t
    return sum

cdef np.ndarray int_to_array(object value, object name, object bits, object uint_size):
    len = bits // uint_size
    value = np.asarray(value)
    if uint_size == 32:
        dtype = np.uint32
    elif uint_size == 64:
        dtype = np.uint64
    else:
        raise ValueError('Unknown uint_size')
    if value.shape == ():
        value = int(value)
        upper = int(2)**int(bits)
        if value < 0 or value >= upper:
            raise ValueError('{name} must be positive and '
                             'less than 2**{bits}.'.format(name=name, bits=bits))

        out = np.empty(len, dtype=dtype)
        for i in range(len):
            out[i] = value % 2**int(uint_size)
            value >>= int(uint_size)
    else:
        out = value.astype(dtype)
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


cdef object double_fill(void *func, brng_t *state, object size, object lock, object out):
    cdef random_double_fill random_func = (<random_double_fill>func)
    cdef double out_val
    cdef double *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None and out is None:
        with lock:
            random_func(state, 1, &out_val)
            return out_val

    if out is not None:
        check_output(out, np.float64, size)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.double)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <double *>np.PyArray_DATA(out_array)
    with lock, nogil:
        random_func(state, n, out_array_data)
    return out_array

cdef object float_fill(void *func, brng_t *state, object size, object lock, object out):
    cdef random_float_0 random_func = (<random_float_0>func)
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None and out is None:
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

cdef object float_fill_from_double(void *func, brng_t *state, object size, object lock, object out):
    cdef random_double_0 random_func = (<random_double_0>func)
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None and out is None:
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


cdef double POISSON_LAM_MAX = <double>np.iinfo('l').max - np.sqrt(np.iinfo('l').max)*10

cdef uint64_t MAXSIZE = <uint64_t>sys.maxsize

cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if np.any(np.signbit(val)):
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE:
        if np.any(np.less_equal(val, 0)):
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1 or cons == CONS_BOUNDED_0_1_NOTNAN:
        if np.any(np.less(val, 0)) or np.any(np.greater(val, 1)):
            raise ValueError(name + " <= 0 or " + name + " >= 1")
        if cons == CONS_BOUNDED_0_1_NOTNAN:
            if np.any(np.isnan(val)):
                raise ValueError(name + ' contains NaNs')
    elif cons == CONS_GT_1:
        if np.any(np.less_equal(val, 1)):
            raise ValueError(name + " <= 1")
    elif cons == CONS_GTE_1:
        if np.any(np.less(val, 1)):
            raise ValueError(name + " < 1")
    elif cons == CONS_POISSON:
        if np.any(np.greater(val, POISSON_LAM_MAX)):
            raise ValueError(name + " value too large")
        if np.any(np.less(val, 0.0)):
            raise ValueError(name + " < 0")

    return 0



cdef int check_constraint(double val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if np.signbit(val):
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE:
        if val <= 0:
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1 or cons == CONS_BOUNDED_0_1_NOTNAN:
        if val < 0 or val > 1:
            raise ValueError(name + " <= 0 or " + name + " >= 1")
        if cons == CONS_BOUNDED_0_1_NOTNAN:
            if np.isnan(val):
                raise ValueError(name + ' contains NaNs')
    elif cons == CONS_GT_1:
        if val <= 1:
            raise ValueError(name + " <= 1")
    elif cons == CONS_GTE_1:
        if val < 1:
            raise ValueError(name + " < 1")
    elif cons == CONS_POISSON:
        if val < 0:
            raise ValueError(name + " < 0")
        elif val > POISSON_LAM_MAX:
            raise ValueError(name + " value too large")

    return 0

cdef object cont_broadcast_1(void *func, brng_t *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             object out):

    cdef np.ndarray randoms
    cdef double a_val
    cdef double *randoms_data
    cdef np.broadcast it
    cdef random_double_1 f = (<random_double_1>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None and out is None:
        randoms = <np.ndarray>np.empty(size, np.double)
    elif out is None:
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_DOUBLE)
    else:
        randoms = <np.ndarray>out

    randoms_data = <double *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)
    it = np.PyArray_MultiIterNew2(randoms, a_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_2(void *func, brng_t *state, object size, object lock,
                 np.ndarray a_arr, object a_name, constraint_type a_constraint,
                 np.ndarray b_arr, object b_name, constraint_type b_constraint):
    cdef np.ndarray randoms
    cdef double a_val, b_val
    cdef double *randoms_data
    cdef np.broadcast it
    cdef random_double_2 f = (<random_double_2>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.double)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)


    randoms_data = <double *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            randoms_data[i] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_3(void *func, brng_t *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             np.ndarray b_arr, object b_name, constraint_type b_constraint,
                             np.ndarray c_arr, object c_name, constraint_type c_constraint):
    cdef np.ndarray randoms
    cdef double a_val, b_val, c_val
    cdef double *randoms_data
    cdef np.broadcast it
    cdef random_double_3 f = (<random_double_3>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if c_constraint != CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)
        randoms = <np.ndarray>np.empty(it.shape, np.double)

    randoms_data = <double *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
            randoms_data[i] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont(void *func, brng_t *state, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint,
                 object out):

    cdef np.ndarray a_arr, b_arr, c_arr
    cdef double _a = 0.0, _b = 0.0, _c = 0.0
    cdef bint is_scalar = True
    check_output(out, np.float64, size)
    if narg > 0:
        a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(a_arr) == 0
    if narg > 1:
        b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
    if narg == 3:
        c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_DOUBLE, np.NPY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(c_arr) == 0

    if not is_scalar:
        if narg == 1:
            return cont_broadcast_1(func, state, size, lock,
                                    a_arr, a_name, a_constraint,
                                    out)
        elif narg == 2:
            return cont_broadcast_2(func, state, size, lock,
                                    a_arr, a_name, a_constraint,
                                    b_arr, b_name, b_constraint)
        else:
            return cont_broadcast_3(func, state, size, lock,
                                    a_arr, a_name, a_constraint,
                                    b_arr, b_name, b_constraint,
                                    c_arr, c_name, c_constraint)

    if narg > 0:
        _a = PyFloat_AsDouble(a)
        if a_constraint != CONS_NONE and is_scalar:
            check_constraint(_a, a_name, a_constraint)
    if narg > 1:
        _b = PyFloat_AsDouble(b)
        if b_constraint != CONS_NONE:
            check_constraint(_b, b_name, b_constraint)
    if narg == 3:
        _c = PyFloat_AsDouble(c)
        if c_constraint != CONS_NONE and is_scalar:
            check_constraint(_c, c_name, c_constraint)

    if size is None and out is None:
        with lock:
            if narg == 0:
                return (<random_double_0>func)(state)
            elif narg == 1:
                return (<random_double_1>func)(state, _a)
            elif narg == 2:
                return (<random_double_2>func)(state, _a, _b)
            elif narg == 3:
                return (<random_double_3>func)(state, _a, _b, _c)

    cdef np.npy_intp i, n
    cdef np.ndarray randoms
    if out is None:
        randoms = <np.ndarray>np.empty(size)
    else:
        randoms = <np.ndarray>out
    n = np.PyArray_SIZE(randoms)

    cdef double *randoms_data =  <double *>np.PyArray_DATA(randoms)
    cdef random_double_0 f0;
    cdef random_double_1 f1;
    cdef random_double_2 f2;
    cdef random_double_3 f3;

    with lock, nogil:
        if narg == 0:
            f0 = (<random_double_0>func)
            for i in range(n):
                randoms_data[i] = f0(state)
        elif narg == 1:
            f1 = (<random_double_1>func)
            for i in range(n):
                randoms_data[i] = f1(state, _a)
        elif narg == 2:
            f2 = (<random_double_2>func)
            for i in range(n):
                randoms_data[i] = f2(state, _a, _b)
        elif narg == 3:
            f3 = (<random_double_3>func)
            for i in range(n):
                randoms_data[i] = f3(state, _a, _b, _c)

    if out is None:
        return randoms
    else:
        return out

cdef object discrete_broadcast_d(void *func, brng_t *state, object size, object lock,
                                 np.ndarray a_arr, object a_name, constraint_type a_constraint):

    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_d f = (<random_uint_d>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.int64)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_INT64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew2(randoms, a_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_dd(void *func, brng_t *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                  np.ndarray b_arr, object b_name, constraint_type b_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_dd f = (<random_uint_dd>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.int64)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_INT64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            randoms_data[i] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_di(void *func, brng_t *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                  np.ndarray b_arr, object b_name, constraint_type b_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_di f = (<random_uint_di>func)
    cdef np.npy_intp i, n


    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.int64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 2))[0]
            (<int64_t*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_iii(void *func, brng_t *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                  np.ndarray b_arr, object b_name, constraint_type b_constraint,
                                  np.ndarray c_arr, object c_name, constraint_type c_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_iii f = (<random_uint_iii>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if c_constraint != CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.int64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 3))[0]
            randoms_data[i] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_i(void *func, brng_t *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_i f = (<random_uint_i>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_INT64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew2(randoms, a_arr)
    with lock, nogil:
        for i in range(n):
            a_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

# Needs double <vec>, double-double <vec>, double-int64_t<vec>, int64_t <vec>, int64_t-int64_t-int64_t
cdef object disc(void *func, brng_t *state, object size, object lock,
                 int narg_double, int narg_int64,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):

    cdef double _da = 0, _db = 0
    cdef int64_t _ia = 0, _ib = 0 , _ic = 0
    cdef bint is_scalar = True
    if narg_double > 0:
        a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(a_arr) == 0
        if narg_double > 1:
            b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
        elif narg_int64 == 1:
            b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_INT64, np.NPY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
    else:
        if narg_int64 > 0:
            a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_INT64, np.NPY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(a_arr) == 0
        if narg_int64 > 1:
            b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_INT64, np.NPY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
        if narg_int64 > 2 :
            c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_INT64, np.NPY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(c_arr) == 0

    if not is_scalar:
        if narg_int64 == 0:
            if narg_double == 1:
                return discrete_broadcast_d(func, state, size, lock,
                                            a_arr, a_name, a_constraint)
            elif narg_double == 2:
                return discrete_broadcast_dd(func, state, size, lock,
                                             a_arr, a_name, a_constraint,
                                             b_arr, b_name, b_constraint)
        elif narg_int64 == 1:
            if narg_double == 0:
                return discrete_broadcast_i(func, state, size, lock,
                                            a_arr, a_name, a_constraint)
            elif narg_double == 1:
                return discrete_broadcast_di(func, state, size, lock,
                                             a_arr, a_name, a_constraint,
                                             b_arr, b_name, b_constraint)
        else:
            raise NotImplementedError("No vector path available")


    if narg_double > 0:
        _da = PyFloat_AsDouble(a)
        if a_constraint != CONS_NONE and is_scalar:
            check_constraint(_da, a_name, a_constraint)

        if narg_double > 1:
            _db = PyFloat_AsDouble(b)
            if b_constraint != CONS_NONE and is_scalar:
                check_constraint(_db, b_name, b_constraint)
        elif narg_int64 == 1:
            _ib = PyInt_AsLong(b)
            if b_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ib, b_name, b_constraint)
    else:
        if narg_int64 > 0:
            _ia = PyInt_AsLong(a)
            if a_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ia, a_name, a_constraint)
        if narg_int64 > 1:
            _ib = PyInt_AsLong(b)
            if b_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ib, b_name, b_constraint)
        if narg_int64 > 2 :
            _ic = PyInt_AsLong(c)
            if c_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ic, c_name, c_constraint)

    if size is None:
        with lock:
            if narg_int64 == 0:
                if narg_double == 0:
                    return (<random_uint_0>func)(state)
                elif narg_double == 1:
                    return (<random_uint_d>func)(state, _da)
                elif narg_double == 2:
                    return (<random_uint_dd>func)(state, _da, _db)
            elif narg_int64 == 1:
                if narg_double == 0:
                    return (<random_uint_i>func)(state, _ia)
                if narg_double == 1:
                    return (<random_uint_di>func)(state, _da, _ib)
            else:
                return (<random_uint_iii>func)(state, _ia, _ib, _ic)

    cdef np.npy_intp i, n
    cdef np.ndarray randoms = <np.ndarray>np.empty(size, np.int64)
    cdef np.int64_t *randoms_data
    cdef random_uint_0 f0;
    cdef random_uint_d fd;
    cdef random_uint_dd fdd;
    cdef random_uint_di fdi;
    cdef random_uint_i fi;
    cdef random_uint_iii fiii;

    n = np.PyArray_SIZE(randoms)
    randoms_data =  <np.int64_t *>np.PyArray_DATA(randoms)

    with lock, nogil:
        if narg_int64 == 0:
            if narg_double == 0:
                f0 = (<random_uint_0>func)
                for i in range(n):
                    randoms_data[i] = f0(state)
            elif narg_double == 1:
                fd = (<random_uint_d>func)
                for i in range(n):
                    randoms_data[i] = fd(state, _da)
            elif narg_double == 2:
                fdd = (<random_uint_dd>func)
                for i in range(n):
                    randoms_data[i] = fdd(state, _da, _db)
        elif narg_int64 == 1:
            if narg_double == 0:
                fi = (<random_uint_i>func)
                for i in range(n):
                    randoms_data[i] = fi(state, _ia)
            if narg_double == 1:
                fdi = (<random_uint_di>func)
                for i in range(n):
                    randoms_data[i] = fdi(state, _da, _ib)
        else:
            fiii = (<random_uint_iii>func)
            for i in range(n):
                randoms_data[i] = fiii(state, _ia, _ib, _ic)

    return randoms


cdef object cont_broadcast_1_f(void *func, brng_t *state, object size, object lock,
                                   np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                   object out):

    cdef np.ndarray randoms
    cdef float a_val
    cdef float *randoms_data
    cdef np.broadcast it
    cdef random_float_1 f = (<random_float_1>func)
    cdef np.npy_intp i, n

    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None and out is None:
        randoms = <np.ndarray>np.empty(size, np.float32)
    elif out is None:
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr),
                                       np.PyArray_DIMS(a_arr),
                                       np.NPY_FLOAT32)
    else:
        randoms = <np.ndarray>out

    randoms_data = <float *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)
    it = np.PyArray_MultiIterNew2(randoms, a_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<float*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_f(void *func, brng_t *state, object size, object lock,
                   object a, object a_name, constraint_type a_constraint,
                   object out):

    cdef np.ndarray a_arr, b_arr, c_arr
    cdef float _a
    cdef bint is_scalar = True
    cdef int requirements = np.NPY_ALIGNED | np.NPY_FORCECAST
    check_output(out, np.float32, size)
    a_arr = <np.ndarray>np.PyArray_FROMANY(a, np.NPY_FLOAT32, 0, 0, requirements)
    # a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_FLOAT32, np.NPY_ALIGNED)
    is_scalar = np.PyArray_NDIM(a_arr) == 0

    if not is_scalar:
        return cont_broadcast_1_f(func, state, size, lock, a_arr, a_name, a_constraint, out)

    _a = <float>PyFloat_AsDouble(a)
    if a_constraint != CONS_NONE:
        check_constraint(_a, a_name, a_constraint)

    if size is None and out is None:
        with lock:
            return (<random_float_1>func)(state, _a)

    cdef np.npy_intp i, n
    cdef np.ndarray randoms
    if out is None:
        randoms = <np.ndarray>np.empty(size, np.float32)
    else:
        randoms = <np.ndarray>out
    n = np.PyArray_SIZE(randoms)

    cdef float *randoms_data =  <float *>np.PyArray_DATA(randoms)
    cdef random_float_1 f1 = <random_float_1>func;

    with lock, nogil:
        for i in range(n):
            randoms_data[i] = f1(state, _a)

    if out is None:
        return randoms
    else:
        return out