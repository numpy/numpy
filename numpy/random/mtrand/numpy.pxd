# :Author:    Travis Oliphant

cdef extern from "numpy/npy_no_deprecated_api.h": pass

cdef extern from "numpy/arrayobject.h":

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

    cdef enum requirements:
        NPY_ARRAY_C_CONTIGUOUS
        NPY_ARRAY_F_CONTIGUOUS
        NPY_ARRAY_OWNDATA
        NPY_ARRAY_FORCECAST
        NPY_ARRAY_ENSURECOPY
        NPY_ARRAY_ENSUREARRAY
        NPY_ARRAY_ELEMENTSTRIDES
        NPY_ARRAY_ALIGNED
        NPY_ARRAY_NOTSWAPPED
        NPY_ARRAY_WRITEABLE
        NPY_ARRAY_UPDATEIFCOPY
        NPY_ARR_HAS_DESCR

        NPY_ARRAY_BEHAVED
        NPY_ARRAY_BEHAVED_NS
        NPY_ARRAY_CARRAY
        NPY_ARRAY_CARRAY_RO
        NPY_ARRAY_FARRAY
        NPY_ARRAY_FARRAY_RO
        NPY_ARRAY_DEFAULT

        NPY_ARRAY_IN_ARRAY
        NPY_ARRAY_OUT_ARRAY
        NPY_ARRAY_INOUT_ARRAY
        NPY_ARRAY_IN_FARRAY
        NPY_ARRAY_OUT_FARRAY
        NPY_ARRAY_INOUT_FARRAY

        NPY_ARRAY_UPDATE_ALL

    cdef enum defines:
        NPY_MAXDIMS

    ctypedef struct npy_cdouble:
        double real
        double imag

    ctypedef struct npy_cfloat:
        double real
        double imag

    ctypedef int npy_intp

    ctypedef extern class numpy.dtype [object PyArray_Descr]: pass

    ctypedef extern class numpy.ndarray [object PyArrayObject]: pass

    ctypedef extern class numpy.flatiter [object PyArrayIterObject]:
        cdef int  nd_m1
        cdef npy_intp index, size
        cdef ndarray ao
        cdef char *dataptr

    ctypedef extern class numpy.broadcast [object PyArrayMultiIterObject]:
        cdef int numiter
        cdef npy_intp size, index
        cdef int nd
        cdef npy_intp *dimensions
        cdef void **iters

    object PyArray_ZEROS(int ndims, npy_intp* dims, NPY_TYPES type_num, int fortran)
    object PyArray_EMPTY(int ndims, npy_intp* dims, NPY_TYPES type_num, int fortran)
    dtype PyArray_DescrFromTypeNum(NPY_TYPES type_num)
    object PyArray_SimpleNew(int ndims, npy_intp* dims, NPY_TYPES type_num)
    int PyArray_Check(object obj)
    object PyArray_ContiguousFromAny(object obj, NPY_TYPES type,
        int mindim, int maxdim)
    object PyArray_ContiguousFromObject(object obj, NPY_TYPES type,
        int mindim, int maxdim)
    npy_intp PyArray_SIZE(ndarray arr)
    npy_intp PyArray_NBYTES(ndarray arr)
    object PyArray_FromAny(object obj, dtype newtype, int mindim, int maxdim,
                            int requirements, object context)
    object PyArray_FROMANY(object obj, NPY_TYPES type_num, int min,
                           int max, int requirements)
    object PyArray_NewFromDescr(object subtype, dtype newtype, int nd,
                                npy_intp* dims, npy_intp* strides, void* data,
                                int flags, object parent)

    object PyArray_FROM_OTF(object obj, NPY_TYPES type, int flags)
    object PyArray_EnsureArray(object)

    object PyArray_MultiIterNew(int n, ...)

    char *PyArray_MultiIter_DATA(broadcast multi, int i)
    void PyArray_MultiIter_NEXTi(broadcast multi, int i)
    void PyArray_MultiIter_NEXT(broadcast multi)

    object PyArray_IterNew(object arr)
    void PyArray_ITER_NEXT(flatiter it)

    void import_array()

# include functions that were once macros in the new api

    int PyArray_NDIM(ndarray arr)
    char * PyArray_DATA(ndarray arr)
    npy_intp * PyArray_DIMS(ndarray arr)
    npy_intp * PyArray_STRIDES(ndarray arr)
    npy_intp PyArray_DIM(ndarray arr, int idim)
    npy_intp PyArray_STRIDE(ndarray arr, int istride)
    object PyArray_BASE(ndarray arr)
    dtype PyArray_DESCR(ndarray arr)
    int PyArray_FLAGS(ndarray arr)
    npy_intp PyArray_ITEMSIZE(ndarray arr)
    int PyArray_TYPE(ndarray arr)
    int PyArray_CHKFLAGS(ndarray arr, int flags)
    object PyArray_GETITEM(ndarray arr, char *itemptr)
