## Automatically adapted for numpy Sep 19, 2005 by convertcode.py

__all__ = ['iscomplexobj','isrealobj','imag','iscomplex',
           'isreal','nan_to_num','real','real_if_close',
           'typename','asfarray','mintypecode','asscalar',
           'common_type']

import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, asanyarray, array, isnan, \
		obj2sctype, zeros
from ufunclike import isneginf, isposinf

_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'

def mintypecode(typechars,typeset='GDFgdf',default='d'):
    """ Return a minimum data type character from typeset that
    handles all typechars given

    The returned type character must be the smallest size such that
    an array of the returned type can handle the data from an array of
    type t for each t in typechars (or if typechars is an array,
    then its dtype.char).

    If the typechars does not intersect with the typeset, then default
    is returned.

    If t in typechars is not a string then t=asarray(t).dtype.char is
    applied.
    """
    typecodes = [(type(t) is type('') and t) or asarray(t).dtype.char\
                 for t in typechars]
    intersection = [t for t in typecodes if t in typeset]
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    l = []
    for t in intersection:
        i = _typecodes_by_elsize.index(t)
        l.append((i,t))
    l.sort()
    return l[0][1]

def asfarray(a, dtype=_nx.float_):
    """asfarray(a,dtype=None) returns a as a float array."""
    dtype = _nx.obj2sctype(dtype)
    if not issubclass(dtype, _nx.inexact):
        dtype = _nx.float_
    return asanyarray(a,dtype=dtype)

def real(val):
    """Return the real part of val.

    Useful if val maybe a scalar or an array.
    """
    return asanyarray(val).real

def imag(val):
    """Return the imaginary part of val.

    Useful if val maybe a scalar or an array.
    """
    return asanyarray(val).imag

def iscomplex(x):
    """Return a boolean array where elements are True if that element
    is complex (has non-zero imaginary part).

    For scalars, return a boolean.
    """
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, _nx.complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return +res  # convet to array-scalar if needed

def isreal(x):
    """Return a boolean array where elements are True if that element
    is real (has zero imaginary part)

    For scalars, return a boolean.
    """
    return imag(x) == 0

def iscomplexobj(x):
    """Return True if x is a complex type or an array of complex numbers.

    Unlike iscomplex(x), complex(3.0) is considered a complex object.
    """
    return issubclass( asarray(x).dtype.type, _nx.complexfloating)

def isrealobj(x):
    """Return True if x is not a complex type.

    Unlike isreal(x), complex(3.0) is considered a complex object.
    """
    return not issubclass( asarray(x).dtype.type, _nx.complexfloating)

#-----------------------------------------------------------------------------

def _getmaxmin(t):
    import getlimits
    f = getlimits.finfo(t)
    return f.max, f.min

def nan_to_num(x):
    """
    Returns a copy of replacing NaN's with 0 and Infs with large numbers

    The following mappings are applied:
        NaN -> 0
        Inf -> limits.double_max
       -Inf -> limits.double_min
    """
    try:
        t = x.dtype.type
    except AttributeError:
        t = obj2sctype(type(x))
    if issubclass(t, _nx.complexfloating):
        y = nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    else:
        try:
            y = x.copy()
        except AttributeError:
            y = array(x)
    if not issubclass(t, _nx.integer):
        if not y.shape:
            y = array([x])
            scalar = True
        else:
            scalar = False
        are_inf = isposinf(y)
        are_neg_inf = isneginf(y)
        are_nan = isnan(y)
        maxf, minf = _getmaxmin(y.dtype.type)
        y[are_nan] = 0
        y[are_inf] = maxf
        y[are_neg_inf] = minf
        if scalar:
            y = y[0]
    return y

#-----------------------------------------------------------------------------

def real_if_close(a,tol=100):
    """If a is a complex array, return it as a real array if the imaginary
    part is close enough to zero.

    "Close enough" is defined as tol*(machine epsilon of a's element type).
    """
    a = asanyarray(a)
    if not issubclass(a.dtype.type, _nx.complexfloating):
        return a
    if tol > 1:
        import getlimits
        f = getlimits.finfo(a.dtype.type)
        tol = f.eps * tol
    if _nx.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a


def asscalar(a):
    """Convert an array of size 1 to its scalar equivalent.
    """
    return a.item()

#-----------------------------------------------------------------------------

_namefromtype = {'S1' : 'character',
                 '?' : 'bool',
                 'b' : 'signed char',
                 'B' : 'unsigned char',
                 'h' : 'short',
                 'H' : 'unsigned short',
                 'i' : 'integer',
                 'I' : 'unsigned integer',
                 'l' : 'long integer',
                 'L' : 'unsigned long integer',
                 'q' : 'long long integer',
                 'Q' : 'unsigned long long integer',
                 'f' : 'single precision',
                 'd' : 'double precision',
                 'g' : 'long precision',
                 'F' : 'complex single precision',
                 'D' : 'complex double precision',
                 'G' : 'complex long double precision',
                 'S' : 'string',
                 'U' : 'unicode',
                 'V' : 'void',
                 'O' : 'object'
                 }

def typename(char):
    """Return an english description for the given data type character.
    """
    return _namefromtype[char]

#-----------------------------------------------------------------------------

#determine the "minimum common type" for a group of arrays.
array_type = [[_nx.single, _nx.double, _nx.longdouble],
              [_nx.csingle, _nx.cdouble, _nx.clongdouble]]
array_precision = {_nx.single : 0,
                   _nx.double : 1,
                   _nx.longdouble : 2,
                   _nx.csingle : 0,
                   _nx.cdouble : 1,
                   _nx.clongdouble : 2}
def common_type(*arrays):
    """Given a sequence of arrays as arguments, return the best inexact
    scalar type which is "most" common amongst them.

    The return type will always be a inexact scalar type, even if all
    the arrays are integer arrays.
    """
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 1
        else:
            p = array_precision.get(t, None)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]
