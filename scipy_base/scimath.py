"""
Wrapper functions to more user-friendly calling of certain math functions
whose output is different than the input in certain domains of the input.
"""

__all__ = ['sqrt', 'log', 'log2','logn','log10', 'power', 'arccos',
           'arcsin', 'arctanh']

import numerix as _nx
from numerix import *

from type_check import isreal, asarray
from function_base import any

__all__.extend([key for key in dir(_nx.fastumath) \
                if key[0]!='_' and key not in __all__])

def _tocomplex(arr):
    if arr.typecode() in ['f', 's', 'b', '1','w']:
        return arr.astype('F')
    else:
        return arr.astype('D')

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
    return fastumath.sqrt(x)

def log(x):
    x = _fix_real_lt_zero(x)
    return fastumath.log(x)

def log10(x):
    x = _fix_real_lt_zero(x)
    return fastumath.log10(x)    

def logn(n,x):
    """ Take log base n of x.
    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return fastumath.log(x)/fastumath.log(n)

def log2(x):
    """ Take log base 2 of x.
    """
    x = _fix_real_lt_zero(x)
    return fastumath.log(x)/fastumath.log(2)

def power(x, p):
    x = _fix_real_lt_zero(x)
    return fastumath.power(x, p)


def arccos(x):
    x = _fix_real_abs_gt_1(x)
    return fastumath.arccos(x)

def arcsin(x):
    x = _fix_real_abs_gt_1(x)
    return fastumath.arcsin(x)

def arctanh(x):
    x = _fix_real_abs_gt_1(x)
    return fastumath.arctanh(x)
