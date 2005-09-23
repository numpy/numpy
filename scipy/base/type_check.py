## Automatically adapted for scipy Sep 19, 2005 by convertcode.py

import types
import numeric as _nx
from numeric import ndarray, array, isinf, isnan, isfinite, signbit, \
     ufunc, ScalarType

__all__ = ['ScalarType','iscomplexobj','isrealobj','imag','iscomplex',
           'isscalar','isneginf','isposinf','isinf','isfinite',
           'isreal','nan_to_num','real','real_if_close',
           'typename','common_type',
           'asfarray','mintypecode']

_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb'

def mintypecode(typecodes,typeset='DFdf',default='d',savespace=0):
    """ Return a typecode in typeset such that for each
    t in typecodes
    array(typecode=typecode)[:] = array(typecode=t)
    is valid, looses no information, and array(typecode=typecode)
    element size is minimal unless when typecodes does not
    intersect with typeset then default is returned.
    As a special case, if savespace is False then 'D' is returned
    whenever typecodes contain 'F' and 'd'.
    If t in typecodes is not a string then t=t.typecode() is applied.
    """
    typecodes = [(type(t) is type('') and t) or asarray(t).dtypechar\
                 for t in typecodes]
    intersection = [t for t in typecodes if t in typeset]
    if not intersection:
       return default
    if not savespace and 'F' in intersection and 'd' in intersection:
       return 'D'
    l = []
    for t in intersection:
       i = _typecodes_by_elsize.index(t)
       l.append((i,t))
    l.sort()
    return l[0][1]

def asfarray(a, dtype=None):
    """asfarray(a,dtype=None) returns a as a float array."""
    a = asarray(a,dtype)
    if typecode is None and a.dtypechar not in 'GDFgfd':
       return a.astype('d')
    return a

   
def isscalar(num):
    if isinstance(num, _nx.generic):
        return True
    else:
        return type(num) in ScalarType

def real(val):
    aval = asarray(val).real

def imag(val):
    aval = asarray(val).imag

def iscomplex(x):
    return imag(x) != _nx.zeros(asarray(x).shape)

def isreal(x):
    return imag(x) == _nx.zeros(asarray(x).shape)

def iscomplexobj(x):
    return issubclass( asarray(x).dtype, _nx.complexfloating)

def isrealobj(x):
    return not issubclass( asarray(x).dtype, _nx.complexfloating)

#-----------------------------------------------------------------------------

def isposinf(val):
    return isinf(val) & ~signbit(val)
    
def isneginf(val):
    return isinf(val) & signbit(val)

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
        t = x.dtype
    except AttributeError:
        t = obj2dtype(type(x))
    if issubclass(t, _nx.complexfloating):
        y = nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    else:   
        y = array(x)
        are_inf = isposinf(x)
        are_neg_inf = isneginf(x)
        are_nan = isnan(x)
        maxf, minf = _getmaxmin(y.dtype)
        y[are_nan] = 0
        y[are_inf] = maxf
        y[are_neg_inf] = minf
    return y

#-----------------------------------------------------------------------------

def real_if_close(a,tol=1e-13):
    a = asarray(a)
    if a.dtypechar in ['F','D','G'] and _nx.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a


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
                 'f' : 'single',
                 'd' : 'float',
                 'g' : 'longfloat',
                 'F' : 'complex single',
                 'D' : 'complex float',
                 'G' : 'complex longfloat',
                 'S' : 'string',
                 'U' : 'unicode',
                 'V' : 'void',
                 'O' : 'object'
                 }

def typename(char):
    """Return an english name for the given typecode character.
    """
    return _namefromtype[char]

#-----------------------------------------------------------------------------

#determine the "minimum common type code" for a group of arrays.
array_kind = {'i':0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
array_type = [['f', 'd'], ['F', 'D']]
def common_type(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtypechar
        kind = max(kind, array_kind[t])
        precision = max(precision, array_precision[t])
    return array_type[kind][precision]

if __name__ == '__main__':
    print "Nothing..."
