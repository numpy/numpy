"""
Wrapper functions to more user-friendly calling of certain math functions whose output is
different than the input under certain conditions.
"""

__all__ = ['sqrt', 'log', 'log10', 'power', 'arccos', 'arcsin', 'arctanh']

from convenience import any, isreal
import fastumath
import Numeric
from fastumath import *
toextend = fastumath.__dict__.keys()
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
    return fastumath.sqrt(x)

def log(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.log(x)

def log10(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.log10(x)    

def logn(n,s):
    """ logn(n,s) -- Take log base n of s.
    """
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    if isreal(n) and (n<0):
        n = _tocomplex(n)
    return fastumath.log(s)/fastumath.log(n)

def log2(s):
    """ log2(s) -- Take log base 2 of s.
    """
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.log(s)/fastumath.log(2)


def power(x, p):
    x = Numeric.asarray(x)
    if isreal(x) and any(x<0):
        x = _tocomplex(x)
    return fastumath.power(x, p)
    
def arccos(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return fastumath.arccos(x)

def arcsin(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return fastumath.arcsin(x)

def arctanh(x):
    x = Numeric.asarray(x)
    if isreal(x) and any(abs(x)>1):
        x = _tocomplex(x)
    return fastumath.arctanh(x)


