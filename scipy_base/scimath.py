"""
Wrapper functions to more user-friendly calling of certain math functions whose output is
different than the input under certain conditions.
"""

__all__ = ['sqrt', 'log', 'log2','logn','log10', 'power', 'arccos', 'arcsin', 'arctanh']

from type_check import isreal
from function_base import any
import scipy_base.fastumath
import Numeric
from scipy_base.fastumath import *
toextend = scipy_base.fastumath.__dict__.keys()
for key in toextend:
    if key not in __all__ and key[0] != '_':
       __all__.append(key)

def _tocomplex(arr):
    if arr.typecode() in ['f', 's', 'b', '1']:
        return arr.astype('F')
    else:
        return arr.astype('D')

def sqrt(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return scipy_base.fastumath.sqrt(x)

def log(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return scipy_base.fastumath.log(x)

def log10(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return scipy_base.fastumath.log10(x)    

def logn(n,x):
    """ Take log base n of x.
    """
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    if isreal(n) and (n<0):
        n = _tocomplex(n)
    return scipy_base.fastumath.log(x)/scipy_base.fastumath.log(n)

def log2(x):
    """ Take log base 2 of x.
    """
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return scipy_base.fastumath.log(x)/scipy_base.fastumath.log(2)


def power(x, p):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return scipy_base.fastumath.power(x, p)
    
def arccos(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return scipy_base.fastumath.arccos(x)

def arcsin(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return scipy_base.fastumath.arcsin(x)

def arctanh(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return scipy_base.fastumath.arctanh(x)


