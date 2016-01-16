from cpython.object cimport PyObject
from cpython.ref cimport PyTypeObject


cdef extern from "Python.h":
    ctypedef int Py_intptr_t


cdef extern from "numpy/arrayobject.h":
    # XXX Remove later
    cdef void** PyArray_API

    cdef enum NPY_TYPES:
        NPY_BOOL
        NPY_BYTE
        NPY_UBYTE
        NPY_SHORT
        NPY_USHORT
        NPY_INT
        NPY_UINT
        NPY_LONG
        NPY_ULONG
        NPY_LONGLONG
        NPY_ULONGLONG
        NPY_FLOAT
        NPY_DOUBLE
        NPY_LONGDOUBLE
        NPY_CFLOAT
        NPY_CDOUBLE
        NPY_CLONGDOUBLE
        NPY_OBJECT
        NPY_STRING
        NPY_UNICODE
        NPY_VOID
        NPY_NTYPES
        NPY_NOTYPE

        NPY_INT8
        NPY_INT16
        NPY_INT32
        NPY_INT64
        NPY_INT128
        NPY_INT256
        NPY_UINT8
        NPY_UINT16
        NPY_UINT32
        NPY_UINT64
        NPY_UINT128
        NPY_UINT256
        NPY_FLOAT16
        NPY_FLOAT32
        NPY_FLOAT64
        NPY_FLOAT80
        NPY_FLOAT96
        NPY_FLOAT128
        NPY_FLOAT256
        NPY_COMPLEX32
        NPY_COMPLEX64
        NPY_COMPLEX128
        NPY_COMPLEX160
        NPY_COMPLEX192
        NPY_COMPLEX256
        NPY_COMPLEX512

        NPY_INTP

    PyArray_Descr* PyArray_DescrFromType (int)

    ctypedef struct PyArrayObject:
        # For use in situations where ndarray can't replace PyArrayObject*,
        # like PyArrayObject**.
        pass

    ctypedef enum NPY_CASTING:
            NPY_NO_CASTING
            NPY_EQUIV_CASTING
            NPY_SAFE_CASTING
            NPY_SAME_KIND_CASTING
            NPY_UNSAFE_CASTING

    ctypedef Py_intptr_t npy_intp

    ctypedef struct PyArray_ArrFuncs:
        pass

    ctypedef struct PyArray_ArrayDescr:
        PyArray_Descr *base
        PyObject *shape

    cdef PyTypeObject *PyArrayDescr_Type

    ctypedef struct PyArray_Descr:
        Py_ssize_t ob_refcnt
        PyTypeObject *ob_type
        PyTypeObject *typeobj
        char kind
        char type
        char byteorder
        int flags
        int type_num
        int elsize
        int alignment
        PyArray_ArrayDescr *subarray
        PyObject *fields
        PyObject *names
        PyArray_ArrFuncs *f


cdef extern from "numpy/ufuncobject.h":

    ctypedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)

    ctypedef struct  PyUFuncObject:
        int nin, nout, nargs
        int identity
        PyUFuncGenericFunction *functions
        void **data
        int ntypes
        int check_return
        char *name
        char *types
        char *doc
        void *ptr
        PyObject *obj
        PyObject *userloops
