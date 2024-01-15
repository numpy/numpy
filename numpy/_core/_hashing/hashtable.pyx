from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from cpython.object cimport PyTypeObject
from decimal import Decimal

cimport cython
from cpython.mem cimport (
    PyMem_Free,
    PyMem_Malloc,
)
from cpython.ref cimport (
    Py_INCREF,
    PyObject,
)
from libc.stdlib cimport (
    free,
    malloc,
)

cdef extern from "Python.h":
    # Note: importing extern-style allows us to declare these as nogil
    # functions, whereas `from cpython cimport` does not.
    bint PyBool_Check(object obj) nogil
    bint PyFloat_Check(object obj) nogil
    bint PyComplex_Check(object obj) nogil
    bint PyObject_TypeCheck(object obj, PyTypeObject* type) nogil

    # Note that following functions can potentially raise an exception,
    # thus they cannot be declared 'nogil'.
    object PyUnicode_EncodeLocale(object obj, const char *errors) nogil
    object PyUnicode_DecodeLocale(const char *str, const char *errors) nogil


import numpy as np

cimport numpy as cnp
from numpy cimport ndarray

cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyFloatingArrType_Type

cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyComplexFloatingArrType_Type

cdef extern from "numpy/npy_common.h":
    int64_t NPY_MIN_INT64

from numpy._core._hashing.khash cimport (
    KHASH_TRACE_DOMAIN,
    are_equivalent_float32_t,
    are_equivalent_float64_t,
    are_equivalent_khcomplex64_t,
    are_equivalent_khcomplex128_t,
    kh_needed_n_buckets,
    kh_python_hash_equal,
    kh_python_hash_func,
    khiter_t,
)

from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# All numeric types except complex
ctypedef fused numeric_t:
    int8_t
    int16_t
    int32_t
    int64_t

    uint8_t
    uint16_t
    uint32_t
    uint64_t

    float32_t
    float64_t

# All numeric types + object, doesn't include complex
ctypedef fused numeric_object_t:
    numeric_t
    object

cdef extern from "numpy/npy_common.h":
    int64_t NPY_MIN_INT64

cdef:
    float64_t INF = <float64_t>np.inf
    float64_t NEGINF = -INF

    int64_t NPY_NAT = NPY_MIN_INT64

    type cDecimal = Decimal  # for faster isinstance checks

def get_hashtable_trace_domain():
    return KHASH_TRACE_DOMAIN


def object_hash(obj):
    return kh_python_hash_func(obj)


def objects_are_equal(a, b):
    return kh_python_hash_equal(a, b)

SIZE_HINT_LIMIT = (1 << 20) + 7


cdef Py_ssize_t _INIT_VEC_CAP = 128

cdef inline const char* get_c_string_buf_and_size(str py_string,
                                                  Py_ssize_t *length) except NULL:
    """
    Extract internal char* buffer of unicode or bytes object `py_string` with
    getting length of this internal buffer saved in `length`.

    Notes
    -----
    Python object owns memory, thus returned char* must not be freed.
    `length` can be NULL if getting buffer length is not needed.

    Parameters
    ----------
    py_string : str
    length : Py_ssize_t*

    Returns
    -------
    buf : const char*
    """
    # Note PyUnicode_AsUTF8AndSize() can
    #  potentially allocate memory inside in unlikely case of when underlying
    #  unicode object was stored as non-utf8 and utf8 wasn't requested before.
    return PyUnicode_AsUTF8AndSize(py_string, length)


cdef inline const char* get_c_string(str py_string) except NULL:
    return get_c_string_buf_and_size(py_string, NULL)

cpdef bint checknull(object val, bint inf_as_na=False):
    """
    Return boolean describing of the input is NA-like, defined here as any
    of:
     - None
     - nan
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - Decimal("NaN")

    Parameters
    ----------
    val : object
    inf_as_na : bool, default False
        Whether to treat INF and -INF as NA values.

    Returns
    -------
    bool
    """
    if val is None:
        return True
    elif is_float_object(val) or is_complex_object(val):
        if val != val:
            return True
        elif inf_as_na:
            return val == INF or val == NEGINF
        return False
    elif cnp.is_timedelta64_object(val):
        return cnp.get_timedelta64_value(val) == NPY_NAT
    elif cnp.is_datetime64_object(val):
        return cnp.get_datetime64_value(val) == NPY_NAT
    else:
        return is_decimal_na(val)


cdef inline bint is_float_object(object obj) noexcept nogil:
    """
    Cython equivalent of `isinstance(val, (float, np.floating))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_float : bool
    """
    return (PyFloat_Check(obj) or
            (PyObject_TypeCheck(obj, &PyFloatingArrType_Type)))


cdef inline bint is_complex_object(object obj) noexcept nogil:
    """
    Cython equivalent of `isinstance(val, (complex, np.complexfloating))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_complex : bool
    """
    return (PyComplex_Check(obj) or
            PyObject_TypeCheck(obj, &PyComplexFloatingArrType_Type))


cdef bint is_decimal_na(object val):
    """
    Is this a decimal.Decimal object Decimal("NAN").
    """
    return isinstance(val, cDecimal) and val != val


include "hashtable_class_helper.pxi"
include "hashtable_func_helper.pxi"


# map derived hash-map types onto basic hash-map types:
if np.dtype(np.intp) == np.dtype(np.int64):
    IntpHashTable = Int64HashTable
    unique_label_indices = _unique_label_indices_int64
elif np.dtype(np.intp) == np.dtype(np.int32):
    IntpHashTable = Int32HashTable
    unique_label_indices = _unique_label_indices_int32
else:
    raise ValueError(np.dtype(np.intp))


cdef class Factorizer:
    cdef readonly:
        Py_ssize_t count

    def __cinit__(self, size_hint: int):
        self.count = 0

    def get_count(self) -> int:
        return self.count

    def factorize(self, values, na_sentinel=-1, na_value=None, mask=None) -> np.ndarray:
        raise NotImplementedError


cdef class ObjectFactorizer(Factorizer):
    cdef public:
        PyObjectHashTable table
        ObjectVector uniques

    def __cinit__(self, size_hint: int):
        self.table = PyObjectHashTable(size_hint)
        self.uniques = ObjectVector()

    def factorize(
        self, ndarray[object] values, na_sentinel=-1, na_value=None, mask=None
    ) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray[np.intp]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = ObjectFactorizer(3)
        >>> fac.factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
        cdef:
            ndarray[intp_t] labels

        if mask is not None:
            raise NotImplementedError("mask not supported for ObjectFactorizer.")

        if self.uniques.external_view_exists:
            uniques = ObjectVector()
            uniques.extend(self.uniques.to_array())
            self.uniques = uniques
        labels = self.table.get_labels(values, self.uniques,
                                       self.count, na_sentinel, na_value)
        self.count = len(self.uniques)
        return labels
