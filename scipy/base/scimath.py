## Automatically adapted for scipy Sep 19, 2005 by convertcode.py

"""
Wrapper functions to more user-friendly calling of certain math functions
whose output is different than the input in certain domains of the input.
"""

__all__ = ['sqrt', 'log', 'log2','logn','log10', 'power', 'arccos',
           'arcsin', 'arctanh']

import numeric as _nx
from numeric import *

from type_check import isreal, asscalar

__all__.extend([key for key in dir(_nx.umath) \
                if key[0]!='_' and key not in __all__])

_ln2 = log(2.0)

def _tocomplex(arr):
    if arr.dtypechar in ['f', 'h', 'B', 'b','H']:
        return arr.astype('F')
    else:
        return arr.astype('D')

def _fix_real_lt_zero(x):
    x = asarray(x)
    if any(isreal(x) & (x<0)):
        x = _tocomplex(x)
    return asscalar(x)

def _fix_real_abs_gt_1(x):
    x = asarray(x)
    if any(isreal(x) & (abs(x)>1)):
        x = _tocomplex(x)
    return x
    
def sqrt(x):
    x = _fix_real_lt_zero(x)
    return _nx.sqrt(x)

def log(x):
    x = _fix_real_lt_zero(x)
    return _nx.log(x)

def log10(x):
    x = _fix_real_lt_zero(x)
    return _nx.log10(x)

def logn(n,x):
    """ Take log base n of x.
    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return log(x)/log(n)

def log2(x):
    """ Take log base 2 of x.
    """
    x = _fix_real_lt_zero(x)
    return log(x)/_ln2

def power(x, p):
    x = _fix_real_lt_zero(x)
    return _nx.power(x, p)


def arccos(x):
    x = _fix_real_abs_gt_1(x)
    return arccos(x)

def arcsin(x):
    x = _fix_real_abs_gt_1(x)
    return arcsin(x)

def arctanh(x):
    x = _fix_real_abs_gt_1(x)
    return arctanh(x)
