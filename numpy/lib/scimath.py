"""
Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.
"""

__all__ = ['sqrt', 'log', 'log2', 'logn','log10', 'power', 'arccos',
           'arcsin', 'arctanh']

import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.lib.type_check import isreal


#__all__.extend([key for key in dir(nx.umath)
#                if key[0] != '_' and key not in __all__])

_ln2 = nx.log(2.0)

def _tocomplex(arr):
    if isinstance(arr.dtype, (nt.single, nt.byte, nt.short, nt.ubyte,
                              nt.ushort)):
        return arr.astype(nt.csingle)
    else:
        return arr.astype(nt.cdouble)

def _fix_real_lt_zero(x):
    x = asarray(x)
    if any(isreal(x) & (x<0)):
        x = _tocomplex(x)
    return x

def _fix_real_abs_gt_1(x):
    x = asarray(x)
    if any(isreal(x) & (abs(x)>1)):
        x = _tocomplex(x)
    return x

def sqrt(x):
    x = _fix_real_lt_zero(x)
    return nx.sqrt(x)

def log(x):
    x = _fix_real_lt_zero(x)
    return nx.log(x)

def log10(x):
    x = _fix_real_lt_zero(x)
    return nx.log10(x)

def logn(n, x):
    """ Take log base n of x.
    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return nx.log(x)/nx.log(n)

def log2(x):
    """ Take log base 2 of x.
    """
    x = _fix_real_lt_zero(x)
    return nx.log(x)/_ln2

def power(x, p):
    x = _fix_real_lt_zero(x)
    return nx.power(x, p)

def arccos(x):
    x = _fix_real_abs_gt_1(x)
    return nx.arccos(x)

def arcsin(x):
    x = _fix_real_abs_gt_1(x)
    return nx.arcsin(x)

def arctanh(x):
    x = _fix_real_abs_gt_1(x)
    return nx.arctanh(x)
