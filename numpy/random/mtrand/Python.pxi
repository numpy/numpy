# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


cdef extern from "Python.h":
    ctypedef int size_t
    char* PyString_AsString(object string)
    object PyString_FromString(char* c_string)

    void* PyMem_Malloc(size_t n)
    void* PyMem_Realloc(void* buf, size_t n)
    void PyMem_Free(void* buf)

    void Py_DECREF(object obj)
    void Py_XDECREF(object obj)
    void Py_INCREF(object obj)
    void Py_XINCREF(object obj)
   
cdef extern from "string.h":
    void *memcpy(void *s1, void *s2, int n)

cdef extern from "math.h":
    double fabs(double x)
