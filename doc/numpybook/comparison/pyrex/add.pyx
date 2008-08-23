# -*- Mode: Python -*-  Not really, but close enough

cimport c_numpy
from c_numpy cimport import_array, ndarray, npy_intp, npy_cdouble, \
     npy_cfloat, NPY_DOUBLE, NPY_CDOUBLE, NPY_FLOAT, \
     NPY_CFLOAT

#We need to initialize NumPy
import_array()

def zadd(object ao, object bo):
    cdef ndarray c, a, b
    cdef npy_intp i

    a = c_numpy.PyArray_ContiguousFromAny(ao, NPY_CDOUBLE, 1, 1)
    b = c_numpy.PyArray_ContiguousFromAny(bo, NPY_CDOUBLE, 1, 1)
    
    c = c_numpy.PyArray_SimpleNew(a.nd, a.dimensions,
                                  a.descr.type_num)

    for i from 0 <= i < a.dimensions[0]:
        (<npy_cdouble *>c.data)[i].real = (<npy_cdouble *>a.data)[i].real + \
                                      (<npy_cdouble *>b.data)[i].real
        (<npy_cdouble *>c.data)[i].imag = (<npy_cdouble *>a.data)[i].imag + \
                                      (<npy_cdouble *>b.data)[i].imag
    return c

def cadd(object ao, object bo):
    cdef ndarray c, a, b
    cdef npy_intp i

    a = c_numpy.PyArray_ContiguousFromAny(ao, NPY_CFLOAT, 1, 1)
    b = c_numpy.PyArray_ContiguousFromAny(bo, NPY_CFLOAT, 1, 1)
    
    c = c_numpy.PyArray_SimpleNew(a.nd, a.dimensions,
                                  a.descr.type_num)

    for i from 0 <= i < a.dimensions[0]:
        (<npy_cfloat *>c.data)[i].real = (<npy_cfloat *>a.data)[i].real + \
                                      (<npy_cfloat *>b.data)[i].real
        (<npy_cfloat *>c.data)[i].imag = (<npy_cfloat *>a.data)[i].imag + \
                                      (<npy_cfloat *>b.data)[i].imag
    return c


def dadd(object ao, object bo):
    cdef ndarray c, a, b
    cdef npy_intp i

    a = c_numpy.PyArray_ContiguousFromAny(ao, NPY_DOUBLE, 1, 1)
    b = c_numpy.PyArray_ContiguousFromAny(bo, NPY_DOUBLE, 1, 1)
    
    c = c_numpy.PyArray_SimpleNew(a.nd, a.dimensions,
                                  a.descr.type_num)

    for i from 0 <= i < a.dimensions[0]:
        (<double *>c.data)[i] = (<double *>a.data)[i] + \
                                (<double *>b.data)[i]
    return c


def sadd(object ao, object bo):
    cdef ndarray c, a, b
    cdef npy_intp i

    a = c_numpy.PyArray_ContiguousFromAny(ao, NPY_FLOAT, 1, 1)
    b = c_numpy.PyArray_ContiguousFromAny(bo, NPY_FLOAT, 1, 1)
    
    c = c_numpy.PyArray_SimpleNew(a.nd, a.dimensions,
                                  a.descr.type_num)

    for i from 0 <= i < a.dimensions[0]:
        (<float *>c.data)[i] = (<float *>a.data)[i] + \
                                (<float *>b.data)[i]
    return c
