"""
numerictypes: Define the numeric type objects

This module is designed so "from numerictypes import \\*" is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    sctypeDict

  Type objects (not all will be available, depends on platform):
      see variable sctypes for which ones you have

    Bit-width names

    int8 int16 int32 int64
    uint8 uint16 uint32 uint64
    float16 float32 float64 float96 float128
    complex64 complex128 complex192 complex256
    datetime64 timedelta64

    c-based names

    bool

    object_

    void, str_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    double, cdouble,
    longdouble, clongdouble,

   As part of the type-hierarchy:    xx -- is bit-width

   generic
     +-> bool                                   (kind=b)
     +-> number
     |   +-> integer
     |   |   +-> signedinteger     (intxx)      (kind=i)
     |   |   |     byte
     |   |   |     short
     |   |   |     intc
     |   |   |     intp
     |   |   |     int_
     |   |   |     longlong
     |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
     |   |         ubyte
     |   |         ushort
     |   |         uintc
     |   |         uintp
     |   |         uint
     |   |         ulonglong
     |   +-> inexact
     |       +-> floating          (floatxx)    (kind=f)
     |       |     half
     |       |     single
     |       |     double
     |       |     longdouble
     |       \\-> complexfloating  (complexxx)  (kind=c)
     |             csingle
     |             cdouble
     |             clongdouble
     +-> flexible
     |   +-> character
     |   |     bytes_                           (kind=S)
     |   |     str_                             (kind=U)
     |   |
     |   \\-> void                              (kind=V)
     \\-> object_ (not used much)               (kind=O)

"""
import numbers
import warnings

from numpy._utils import set_module

import numpy as np
from . import multiarray as ma
from .multiarray import (
    busday_count,
    busday_offset,
    busdaycalendar,
    datetime_as_string,
    datetime_data,
    dtype,
    is_busday,
    ndarray,
)

# we add more at the bottom
__all__ = [
    'ScalarType', 'typecodes', 'issubdtype', 'datetime_data',
    'datetime_as_string', 'busday_offset', 'busday_count',
    'is_busday', 'busdaycalendar', 'isdtype'
]

# we don't need all these imports, but we need to keep them for compatibility
# for users using np._core.numerictypes.UPPER_TABLE
# we don't export these for import *, but we do want them accessible
# as numerictypes.bool, etc.
from builtins import bool, bytes, complex, float, int, object, str  # noqa: F401, UP029

from ._string_helpers import (  # noqa: F401
    LOWER_TABLE,
    UPPER_TABLE,
    english_capitalize,
    english_lower,
    english_upper,
)
from ._type_aliases import allTypes, sctypeDict, sctypes

# We use this later
generic = allTypes['generic']

genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16',
                   'int32', 'uint32', 'int64', 'uint64',
                   'float16', 'float32', 'float64', 'float96', 'float128',
                   'complex64', 'complex128', 'complex192', 'complex256',
                   'object']


@set_module('numpy')
def issctype(rep):
    """
    Determines whether the given object represents a scalar data-type.

    Parameters
    ----------
    rep : any
        If `rep` is an instance of a scalar dtype, True is returned. If not,
        False is returned.

    Returns
    -------
    out : bool
        Boolean result of check whether `rep` is a scalar dtype.

    See Also
    --------
    issubsctype, issubdtype, obj2sctype, sctype2char

    Examples
    --------
    >>> from numpy._core.numerictypes import issctype
    >>> issctype(np.int32)
    True
    >>> issctype(list)
    False
    >>> issctype(1.1)
    False

    Strings are also a scalar type:

    >>> issctype(np.dtype(np.str_))
    True

    """
    if not isinstance(rep, (type, dtype)):
        return False
    try:
        res = obj2sctype(rep)
        if res and res != object_:
            return True
        else:
            return False
    except Exception:
        return False


def obj2sctype(rep, default=None):
    """
    Return the scalar dtype or NumPy equivalent of Python type of an object.

    Parameters
    ----------
    rep : any
        The object of which the type is returned.
    default : any, optional
        If given, this is returned for objects whose types can not be
        determined. If not given, None is returned for those objects.

    Returns
    -------
    dtype : dtype or Python type
        The data type of `rep`.

    See Also
    --------
    sctype2char, issctype, issubsctype, issubdtype

    Examples
    --------
    >>> from numpy._core.numerictypes import obj2sctype
    >>> obj2sctype(np.int32)
    <class 'numpy.int32'>
    >>> obj2sctype(np.array([1., 2.]))
    <class 'numpy.float64'>
    >>> obj2sctype(np.array([1.j]))
    <class 'numpy.complex128'>

    >>> obj2sctype(dict)
    <class 'numpy.object_'>
    >>> obj2sctype('string')

    >>> obj2sctype(1, default=list)
    <class 'list'>

    """
    # prevent abstract classes being upcast
    if isinstance(rep, type) and issubclass(rep, generic):
        return rep
    # extract dtype from arrays
    if isinstance(rep, ndarray):
        return rep.dtype.type
    # fall back on dtype to convert
    try:
        res = dtype(rep)
    except Exception:
        return default
    else:
        return res.type


@set_module('numpy')
def issubclass_(arg1, arg2):
    """
    Determine if a class is a subclass of a second class.

    `issubclass_` is equivalent to the Python built-in ``issubclass``,
    except that it returns False instead of raising a TypeError if one
    of the arguments is not a class.

    Parameters
    ----------
    arg1 : class
        Input class. True is returned if `arg1` is a subclass of `arg2`.
    arg2 : class or tuple of classes.
        Input class. If a tuple of classes, True is returned if `arg1` is a
        subclass of any of the tuple elements.

    Returns
    -------
    out : bool
        Whether `arg1` is a subclass of `arg2` or not.

    See Also
    --------
    issubsctype, issubdtype, issctype

    Examples
    --------
    >>> np.issubclass_(np.int32, int)
    False
    >>> np.issubclass_(np.int32, float)
    False
    >>> np.issubclass_(np.float64, float)
    True

    """
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False


@set_module('numpy')
def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    Parameters
    ----------
    arg1, arg2 : dtype or dtype specifier
        Data-types.

    Returns
    -------
    out : bool
        The result.

    See Also
    --------
    issctype, issubdtype, obj2sctype

    Examples
    --------
    >>> from numpy._core import issubsctype
    >>> issubsctype('S8', str)
    False
    >>> issubsctype(np.array([1]), int)
    True
    >>> issubsctype(np.array([1]), float)
    False

    """
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))


class _PreprocessDTypeError(Exception):
    pass




def _preprocess_dtype(obj):
    """
    Preprocess dtype argument to allow only NumPy dtypes and scalar
    types.
    """
    if isinstance(obj, ma.dtype):
        return obj
    if isinstance(obj, type):
        try:
            dtype = np.dtype(obj)
        except TypeError:
            pass
        else:
            # If the discovered dtype has the input type as
            # it's type, then it should be a valid input.
            # (rejects e.g. the abstract types)
            if dtype.type is obj:
                return dtype

    raise TypeError(
        "dtype argument must be a NumPy dtype or concrete scalar type, "
        f"but it is a {type(obj)}."
    ) from None


_kind_to_dtypes = {
    "bool": (np.dtypes.BoolDType,),
    "signed integer": (np.dtypes.SignedIntegerAbstractDType,),
    "unsigned integer": (np.dtypes.UnsignedIntegerAbstractDType,),
    "integral": (np.dtypes.IntegerAbstractDType,),
    "real floating": (np.dtypes.FloatAbstractDType,),
    "complex floating": (np.dtypes.ComplexAbstractDType,),
    # The Array API "numeric" kind excludes bool, so do not use
    # NumericAbstractDType directly here.
    "numeric": (
        np.dtypes.IntegerAbstractDType,
        np.dtypes.InexactAbstractDType,
    ),
}


def _preprocess_kind(kind):
    if isinstance(kind, str):
        try:
            return _kind_to_dtypes[kind]
        except KeyError:
            raise ValueError(
                "kind argument is a string, but"
                f" {kind!r} is not a known kind name."
            ) from None
    if isinstance(kind, type):
        if issubclass(kind, np.dtype):
            return kind

        dt = np.dtype(kind)
        if type(dt).type is kind:
            # OK, the scalar type seems to map to a DType cleanly
            # (works for all NumPy scalar types)
            return type(dt)
        elif dt.type is kind:
            # Should be impossible, but just in case act as if the user
            # passed in the concrete dtype.
            kind = dt

    if isinstance(kind, np.dtype):
        # Not really OK, this would be an identity or equal check
        # Deprecated in NumPy 2.6, 2026-06
        warnings.warn(
            "isdtype() with a dtype instance as second argument is deprecated "
            "as it has no clear meaning.  Use `==` instead which also checks "
            "byte-order or use a string/class from `numpy.dtypes.<kind>`. "
            "Scalar classes (`np.int64`, etc.) can generally be used as well. "
            "(Deprecated NumPy 2.6).",
            DeprecationWarning, stacklevel=2
        )
        return type(kind)

    raise TypeError(
        "kind argument must be a DType class or string, "
        f"but it is a {type(kind)}."
    ) from None


@set_module('numpy')
def isdtype(dtype, kind):
    """
    Determine if a provided dtype is of a specified data type ``kind``.

    Parameters
    ----------
    dtype : dtype
        The input dtype.
    kind : DType class, str, or tuple of DType classes/strs.
        The supported strings kinds are:
        * ``'bool'`` : boolean DType
        * ``'signed integer'`` : signed integer data types
        * ``'unsigned integer'`` : unsigned integer data types
        * ``'integral'`` : integer data types
        * ``'real floating'`` : real-valued floating-point data types
        * ``'complex floating'`` : complex floating-point data types
        * ``'numeric'`` : numeric data types

        Otherwise should be a dtype class from the `numpy.dtypes` namespace.
        Concrete scalar types such as ``np.float64`` are also acceptable.

    Returns
    -------
    out : bool

    See Also
    --------
    issubdtype

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0], dtype=np.float32)
    >>> np.isdtype(arr.dtype, np.float64)
    False
    >>> np.isdtype(arr.dtype, "real floating")
    True
    >>> np.isdtype(arr.dtype, ("real floating", "complex floating"))
    True
    >>> np.isdtype(np.dtype(">f8"), np.dtypes.Float64DType)
    True

    Notes
    -----
    After normalization of the input arguments, this function is equivalent to
    ``isinstance(dtype, np.dtypes.<kind>)``.

    User defined dtypes can pass a ``isdtype(dt, "real floating")`` check if
    they register via ``numpy.dtypes.FloatAbstractDType.register(type(dt))``.

    """
    dtype = _preprocess_dtype(dtype)

    if isinstance(kind, tuple):
        kind = tuple(_preprocess_kind(k) for k in kind)
    else:
        kind = _preprocess_kind(kind)

    return isinstance(dtype, kind)


@set_module('numpy')
def issubdtype(arg1, arg2):
    r"""
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    .. note::
        This function relies on the scalar type hierarchy. This works in
        practice for NumPy dtypes and some user-defines ones but does not
        generalize necessarily.
        ``isdtype()`` has slightly clearer semantics although NumPy 2.6 is
        requires to work with user defined dtypes in general.

    Parameters
    ----------
    arg1, arg2 : dtype_like
        `dtype` or object coercible to one

    Returns
    -------
    out : bool

    See Also
    --------
    isdtype : Similar function with clearer defined semantics.
    :ref:`arrays.scalars` : Overview of the numpy type hierarchy.

    Examples
    --------
    `issubdtype` can be used to check the type of arrays:

    >>> ints = np.array([1, 2, 3], dtype=np.int32)
    >>> np.issubdtype(ints.dtype, np.integer)
    True
    >>> np.issubdtype(ints.dtype, np.floating)
    False

    >>> floats = np.array([1, 2, 3], dtype=np.float32)
    >>> np.issubdtype(floats.dtype, np.integer)
    False
    >>> np.issubdtype(floats.dtype, np.floating)
    True

    Similar types of different sizes are not subdtypes of each other:

    >>> np.issubdtype(np.float64, np.float32)
    False
    >>> np.issubdtype(np.float32, np.float64)
    False

    but both are subtypes of `floating`:

    >>> np.issubdtype(np.float64, np.floating)
    True
    >>> np.issubdtype(np.float32, np.floating)
    True

    For convenience, dtype-like objects are allowed too:

    >>> np.issubdtype('S1', np.bytes_)
    True
    >>> np.issubdtype('i4', np.signedinteger)
    True

    Abstract DType classes from :mod:`numpy.dtypes` are also accepted, in
    which case the DType class hierarchy is consulted directly:

    >>> np.issubdtype(np.dtype('int64'), np.dtypes.IntegerAbstractDType)
    True

    """
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type

    return issubclass(arg1, arg2)


@set_module('numpy')
def sctype2char(sctype):
    """
    Return the string representation of a scalar dtype.

    Parameters
    ----------
    sctype : scalar dtype or object
        If a scalar dtype, the corresponding string character is
        returned. If an object, `sctype2char` tries to infer its scalar type
        and then return the corresponding string character.

    Returns
    -------
    typechar : str
        The string character corresponding to the scalar type.

    Raises
    ------
    ValueError
        If `sctype` is an object for which the type can not be inferred.

    See Also
    --------
    obj2sctype, issctype, issubsctype, mintypecode

    Examples
    --------
    >>> from numpy._core.numerictypes import sctype2char
    >>> for sctype in [np.int32, np.double, np.cdouble, np.bytes_, np.ndarray]:
    ...     print(sctype2char(sctype))
    l # may vary
    d
    D
    S
    O

    >>> x = np.array([1., 2-1.j])
    >>> sctype2char(x)
    'D'
    >>> sctype2char(list)
    'O'

    """
    sctype = obj2sctype(sctype)
    if sctype is None:
        raise ValueError("unrecognized type")
    if sctype not in sctypeDict.values():
        # for compatibility
        raise KeyError(sctype)
    return dtype(sctype).char


def _scalar_type_key(typ):
    """A ``key`` function for `sorted`."""
    dt = dtype(typ)
    return (dt.kind.lower(), dt.itemsize)


ScalarType = [int, float, complex, bool, bytes, str, memoryview]
ScalarType += sorted(dict.fromkeys(sctypeDict.values()), key=_scalar_type_key)
ScalarType = tuple(ScalarType)


# Now add the types we've determined to this module
for key in allTypes:
    globals()[key] = allTypes[key]
    __all__.append(key)

del key

typecodes = {'Character': 'c',
             'Integer': 'bhilqnp',
             'UnsignedInteger': 'BHILQNP',
             'Float': 'efdg',
             'Complex': 'FDG',
             'AllInteger': 'bBhHiIlLqQnNpP',
             'AllFloat': 'efdgFDG',
             'Datetime': 'Mm',
             'All': '?bhilqnpBHILQNPefdgFDGSUVOMm'}

# backwards compatibility --- deprecated name
# Formal deprecation: Numpy 1.20.0, 2020-10-19 (see numpy/__init__.py)
typeDict = sctypeDict

def _register_types():
    numbers.Integral.register(integer)
    numbers.Complex.register(inexact)
    numbers.Real.register(floating)
    numbers.Number.register(number)


_register_types()
