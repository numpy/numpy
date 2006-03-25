## Automatically adapted for numpy Sep 19, 2005 by convertcode.py

__all__ = ['iscomplexobj','isrealobj','imag','iscomplex',
           'isreal','nan_to_num','real','real_if_close',
           'typename','asfarray','mintypecode','asscalar',
           'common_type']

import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, array, isnan, obj2sctype
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
    a = asarray(a,dtype=dtype)
    return a

def real(val):
    return asarray(val).real

def imag(val):
    return asarray(val).imag

def iscomplex(x):
    return imag(x) != _nx.zeros_like(x)

def isreal(x):
    return imag(x) == _nx.zeros_like(x)

def iscomplexobj(x):
    return issubclass( asarray(x).dtype.type, _nx.complexfloating)

def isrealobj(x):
    return not issubclass( asarray(x).dtype.type, _nx.complexfloating)

#-----------------------------------------------------------------------------

def _getmaxmin(t):
    import getlimits
    f = getlimits.finfo(t)
    return f.max, f.min

def nan_to_num(x):
    # mapping:
    #    NaN -> 0
    #    Inf -> limits.double_max
    #   -Inf -> limits.double_min
    try:
        t = x.dtype.type
    except AttributeError:
        t = obj2sctype(type(x))
    if issubclass(t, _nx.complexfloating):
        y = nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    elif issubclass(t, _nx.integer):
        y = array(x)
    else:
        y = array(x)
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
    a = asarray(a)
    if a.dtype.char not in 'FDG':
        return a
    if tol > 1:
        import getlimits
        f = getlimits.finfo(a.dtype.type)
        tol = f.eps * tol
    if _nx.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a


def asscalar(a):
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

#determine the "minimum common type code" for a group of arrays.
array_kind = {'i':0, 'l': 0, 'f': 0, 'd': 0, 'g':0, 'F': 1, 'D': 1, 'G':1}
array_precision = {'i': 1, 'l': 1,
                   'f': 0, 'd': 1, 'g':2,
                   'F': 0, 'D': 1, 'G':2}
array_type = [['f', 'd', 'g'], ['F', 'D', 'G']]
def common_type(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, array_kind[t])
        precision = max(precision, array_precision[t])
    return array_type[kind][precision]
