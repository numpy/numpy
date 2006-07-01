# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


cdef extern from "numpy/arrayobject.h":

    cdef enum PyArray_TYPES:
        PyArray_BOOL
        PyArray_BYTE
        PyArray_UBYTE
        PyArray_SHORT
        PyArray_USHORT 
        PyArray_INT
        PyArray_UINT 
        PyArray_LONG
        PyArray_ULONG
        PyArray_LONGLONG
        PyArray_ULONGLONG
        PyArray_FLOAT
        PyArray_DOUBLE 
        PyArray_LONGDOUBLE
        PyArray_CFLOAT
        PyArray_CDOUBLE
        PyArray_CLONGDOUBLE
        PyArray_OBJECT
        PyArray_STRING
        PyArray_UNICODE
        PyArray_VOID
        PyArray_NTYPES
        PyArray_NOTYPE

    cdef enum requirements:
        CONTIGUOUS
        FORTRAN
        OWNDATA
        FORCECAST
        ENSURECOPY
        ENSUREARRAY
        ELEMENTSTRIDES
        ALIGNED
        NOTSWAPPED
        WRITEABLE
        UPDATEIFCOPY
        ARR_HAS_DESCR

        BEHAVED_FLAGS
        BEHAVED_NS_FLAGS
        CARRAY_FLAGS
        CARRAY_FLAGS_RO
        FARRAY_FLAGS
        FARRAY_FLAGS_RO
        DEFAULT_FLAGS

        IN_ARRAY
        OUT_ARRAY
        INOUT_ARRAY
        IN_FARRAY
        OUT_FARRAY
        INOUT_FARRAY

        UPDATE_ALL_FLAGS 

    ctypedef struct cdouble:
        double real
        double imag

    ctypedef struct cfloat:
        double real
        double imag

    ctypedef int intp 

    ctypedef extern class numpy.dtype [object PyArray_Descr]:
        cdef int type_num, elsize, alignment
        cdef char type, kind, byteorder, hasobject
        cdef object fields, typeobj

    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef object base
        cdef dtype descr
        cdef int flags

    object PyArray_ZEROS(int ndims, intp* dims, PyArray_TYPES type_num, int fortran)
    object PyArray_EMPTY(int ndims, intp* dims, PyArray_TYPES type_num, int fortran)
    dtype PyArray_DescrFromTypeNum(PyArray_TYPES type_num)
    object PyArray_SimpleNew(int ndims, intp* dims, PyArray_TYPES type_num)
    int PyArray_Check(object obj)
    object PyArray_ContiguousFromAny(object obj, PyArray_TYPES type, 
        int mindim, int maxdim)
    intp PyArray_SIZE(ndarray arr)
    intp PyArray_NBYTES(ndarray arr)
    void *PyArray_DATA(ndarray arr)
    object PyArray_FromAny(object obj, dtype newtype, int mindim, int maxdim,
		    int requirements, object context)
    object PyArray_FROMANY(object obj, PyArray_TYPES type_num, int min,
                           int max, int requirements)
    object PyArray_NewFromDescr(object subtype, dtype newtype, int nd,
                                intp* dims, intp* strides, void* data,
                                int flags, object parent)

    void import_array()
