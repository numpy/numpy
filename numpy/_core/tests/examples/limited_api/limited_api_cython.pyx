#cython: language_level=3

"""
Make sure cython can compile using the NumPy C-API in limited API mode
(see meson.build).

"""

cimport numpy as cnp

cnp.import_array()


def nonzero(cnp.ndarray arr):
    """Count non-zero elements of a float64 array using the NpyIter API."""
    cdef cnp.NpyIter *it
    cdef cnp.NpyIter_IterNextFunc iternext
    cdef char **dataptr
    cdef cnp.npy_intp *strideptr
    cdef cnp.npy_intp *innersizeptr
    cdef cnp.npy_intp nonzero_count = 0
    cdef cnp.npy_intp stride, count
    cdef char *data

    if cnp.PyArray_SIZE(arr) == 0:
        return 0

    it = cnp.NpyIter_New(
        arr,
        cnp.NPY_ITER_READONLY | cnp.NPY_ITER_EXTERNAL_LOOP | cnp.NPY_ITER_REFS_OK,
        cnp.NPY_KEEPORDER, cnp.NPY_NO_CASTING, <cnp.dtype>NULL)
    if it == NULL:
        raise RuntimeError("NpyIter_New failed")
    iternext = cnp.NpyIter_GetIterNext(it, NULL)
    if iternext == NULL:
        cnp.NpyIter_Deallocate(it)
        raise RuntimeError("NpyIter_GetIterNext failed")
    dataptr = cnp.NpyIter_GetDataPtrArray(it)
    strideptr = cnp.NpyIter_GetInnerStrideArray(it)
    innersizeptr = cnp.NpyIter_GetInnerLoopSizePtr(it)

    while True:
        data = dataptr[0]
        stride = strideptr[0]
        count = innersizeptr[0]
        while count > 0:
            if (<double *>data)[0] != 0.0:
                nonzero_count += 1
            data += stride
            count -= 1
        if not iternext(it):
            break

    cnp.NpyIter_Deallocate(it)
    return nonzero_count


def iter_next(cnp.ndarray arr):
    """Sum float64 array elements using PyArray_ITER_NEXT / _DATA / _NOTDONE."""
    cdef cnp.flatiter it = cnp.PyArray_IterNew(arr)
    cdef double total = 0.0
    while cnp.PyArray_ITER_NOTDONE(it):
        total += (<double *>cnp.PyArray_ITER_DATA(it))[0]
        cnp.PyArray_ITER_NEXT(it)
    return total


def iter_goto1d(cnp.ndarray arr, cnp.npy_intp index):
    """Get element at a flat index using PyArray_ITER_GOTO1D."""
    cdef cnp.flatiter it = cnp.PyArray_IterNew(arr)
    cnp.PyArray_ITER_GOTO1D(it, index)
    return (<double *>cnp.PyArray_ITER_DATA(it))[0]


def iter_reset(cnp.ndarray arr):
    """Sum float64 array elements after PyArray_ITER_RESET."""
    cdef cnp.flatiter it = cnp.PyArray_IterNew(arr)
    cdef double total = 0.0
    while cnp.PyArray_ITER_NOTDONE(it):
        cnp.PyArray_ITER_NEXT(it)
    cnp.PyArray_ITER_RESET(it)
    while cnp.PyArray_ITER_NOTDONE(it):
        total += (<double *>cnp.PyArray_ITER_DATA(it))[0]
        cnp.PyArray_ITER_NEXT(it)
    return total


def iter_goto(cnp.ndarray arr, tuple coord):
    """Get element at a coordinate using PyArray_ITER_GOTO."""
    cdef int nd = cnp.PyArray_NDIM(arr)
    cdef cnp.npy_intp destination[32]
    cdef cnp.flatiter it
    cdef int i
    if len(coord) != nd:
        raise ValueError("coordinate length mismatch")
    for i in range(nd):
        destination[i] = coord[i]
    it = cnp.PyArray_IterNew(arr)
    cnp.PyArray_ITER_GOTO(it, destination)
    return (<double *>cnp.PyArray_ITER_DATA(it))[0]


def multi_iter_next(cnp.ndarray a, cnp.ndarray b):
    """Sum broadcast (a + b) using PyArray_MultiIter_NEXT / _DATA."""
    cdef cnp.broadcast multi = cnp.PyArray_MultiIterNew2(a, b)
    cdef double total = 0.0
    while cnp.PyArray_MultiIter_NOTDONE(multi):
        total += (<double *>cnp.PyArray_MultiIter_DATA(multi, 0))[0]
        total += (<double *>cnp.PyArray_MultiIter_DATA(multi, 1))[0]
        cnp.PyArray_MultiIter_NEXT(multi)
    return total


def multi_iter_goto1d(cnp.ndarray a, cnp.ndarray b, cnp.npy_intp index):
    """Get (a, b) at a flat index using PyArray_MultiIter_GOTO1D."""
    cdef cnp.broadcast multi = cnp.PyArray_MultiIterNew2(a, b)
    cdef double va, vb
    cnp.PyArray_MultiIter_GOTO1D(multi, index)
    va = (<double *>cnp.PyArray_MultiIter_DATA(multi, 0))[0]
    vb = (<double *>cnp.PyArray_MultiIter_DATA(multi, 1))[0]
    return va, vb


def multi_iter_nexti(cnp.ndarray a, cnp.ndarray b, int steps):
    """Advance only iterator 0 by N steps using PyArray_MultiIter_NEXTi."""
    cdef cnp.broadcast multi = cnp.PyArray_MultiIterNew2(a, b)
    cdef int _i
    for _i in range(steps):
        cnp.PyArray_MultiIter_NEXTi(multi, 0)
    return (<double *>cnp.PyArray_MultiIter_DATA(multi, 0))[0]


def get_datetime_value(obj):
    """Underlying int64 of a datetime64 scalar via get_datetime64_value."""
    return cnp.get_datetime64_value(obj)


def get_timedelta_value(obj):
    """Underlying int64 of a timedelta64 scalar via get_timedelta64_value."""
    return cnp.get_timedelta64_value(obj)


def get_datetime_unit(obj):
    """Unit (NPY_DATETIMEUNIT base) of a datetime64 scalar."""
    return <int>cnp.get_datetime64_unit(obj)


def is_datetime64(obj):
    """Cython is_datetime64_object (isinstance-like check)."""
    return cnp.is_datetime64_object(obj)


def is_timedelta64(obj):
    """Cython is_timedelta64_object (isinstance-like check)."""
    return cnp.is_timedelta64_object(obj)
