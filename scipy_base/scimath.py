"""
Wrapper functions to more user-friendly calling of certain math functions
whose output is different than the input in certain domains of the input.
"""

__all__ = ['sqrt', 'log', 'log2','logn','log10', 'power', 'arccos',
           'arcsin', 'arctanh']

import Numeric

from type_check import isreal, asarray
from function_base import any
import fastumath
from fastumath import *

__all__.extend([key for key in dir(fastumath) \
                if key[0]!='_' and key not in __all__])

def _tocomplex(arr):
    if arr.typecode() in ['f', 's', 'b', '1','w']:
        return arr.astype('F')
    else:
        return arr.astype('D')

def sqrt(x):
    x = asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.sqrt(x)

def log(x):
    x = asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.log(x)

def log10(x):
    x = asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.log10(x)    

def logn(n,x):
    """ Take log base n of x.
    """
    x = asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    if isreal(n) and (n<0):
        n = _tocomplex(n)
    return fastumath.log(x)/fastumath.log(n)

def log2(x):
    """ Take log base 2 of x.
    """
    x = asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.log(x)/fastumath.log(2)


def power(x, p):
    x = asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.power(x, p)
    
def arccos(x):
    x = asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return fastumath.arccos(x)

def arcsin(x):
    x = asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return fastumath.arcsin(x)

def arctanh(x):
    x = asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return fastumath.arctanh(x)
