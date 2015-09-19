"""
Pure python implementations of ufuncs for Object type.
Makes use of math and cmath module for some ufuncs.

"""
from __future__ import division, absolute_import, print_function
import math
from numpy.core.umath import pi as PI
import numpy as np
import sys

LOG2 = math.log(2)

if sys.version_info > (3,):
    def isint(x):
        return type(x) in (bool, int)
else:
    def isint(x):
        return type(x) in (bool, int, long)

def handled_by_np(x):
    return type(x) in (float, complex)

def _notimplemented_msg(fname, etype, etype2=None):
    if etype2 is not None:
        typestr = "({a}, {b})".format(a=etype, b=etype2)
    else:
        typestr = str(etype)
    return ("object-array fallback for ufunc '{name}' is not implemented for "
            "arguments of type {type}").format(name=fname, type=typestr)


def logical_and(x, y):
    return bool(x and y)

def logical_or(x, y):
    return bool(x or y)

def logical_xor(x, y):
    return bool(x or y) and not bool(x and y)

def logical_not(x):
    return bool(not x)

def maximum(x, y):
    if isint(y) and handled_by_np(x):
        return np.maximum(x, float(y))
    if isint(x) and handled_by_np(y):
        return np.maximum(float(x), y)
    return max(x, y)

def minimum(x, y):
    if isint(y) and handled_by_np(x):
        return np.minimum(x, float(y))
    if isint(x) and handled_by_np(y):
        return np.minimum(float(x), y)
    return min(x, y)

def fmax(x, y):
    if isint(y) and handled_by_np(x):
        return np.fmax(x, float(y))
    if isint(x) and handled_by_np(y):
        return np.fmax(float(x), y)
    return max(x, y)

def fmin(x, y):
    if isint(y) and handled_by_np(x):
        return np.fmin(x, float(y))
    if isint(x) and handled_by_np(y):
        return np.fmin(float(x), y)
    return min(x, y)

def iscomplex(x):
    if isint(x):
        return False
    raise TypeError(_notimplemented_msg('iscomplex', type(x)))

def isreal(x):
    if isint(x):
        return True
    raise TypeError(_notimplemented_msg('isreal', type(x)))

def isnan(x):
    if isint(x):
        return False
    raise TypeError(_notimplemented_msg('isnan', type(x)))

def isinf(x):
    if isint(x):
        return False
    raise TypeError(_notimplemented_msg('isinf', type(x)))

def isfinite(x):
    if isint(x):
        return True
    raise TypeError(_notimplemented_msg('isfinite', type(x)))

def conjugate(x):
    if isint(x):
        return x
    raise TypeError(_notimplemented_msg('conjugate', type(x)))

def degrees(x):
    return x*180/PI
rad2deg = degrees

def radians(x):
    return x*PI/180
deg2rad = radians

def square(x):
    return x*x

def reciprocal(x):
    if isint(x):
        if x == 1:
            return 1
        if x == -1:
            return -1
        if x == 0:
            return np.reciprocal(0)
        return 0
    raise TypeError(_notimplemented_msg('reciprocal', type(x)))

def cbrt(x):
    if isint(x):
        if x < 0:
            return -(-x)**(1.0/3)
        return x**(1.0/3)
    raise TypeError(_notimplemented_msg('cbrt', type(x)))

def _ones_like(x):
    return 1

def sign(x):
    if isint(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
        return x
    raise TypeError(_notimplemented_msg('sign', type(x)))

def hypot(x, y):
    if isint(x) or isint(y):
        if isint(x):
            x = float(x)
        if isint(y):
            y = float(y)
        return np.hypot(x, y)
    raise TypeError(_notimplemented_msg('hypot', type(x)))

def signbit(x):
    if isint(x):
        return math.copysign(1, x) < 0
    raise TypeError(_notimplemented_msg('signbit', type(x)))

def nextafter(x, y):
    if isint(x) or isint(y):
        if isint(x):
            x = float(x)
        if isint(y):
            y = float(y)
        return np.nextafter(x, y)
    raise TypeError(_notimplemented_msg('nextafter', type(x)))

def copysign(x, y):
    if isint(x) or isint(y):
        return math.copysign(x, y)
    raise TypeError(_notimplemented_msg('copysign', type(x)))

def logaddexp(x, y):
    if isint(x) or isint(y):
        if isint(x):
            x = float(x)
        if isint(y):
            y = float(y)
        return np.logaddexp(x, y)
    raise TypeError(_notimplemented_msg('logaddexp', type(x)))

def logaddexp2(x, y):
    if isint(x) or isint(y):
        if isint(x):
            x = float(x)
        if isint(y):
            y = float(y)
        return np.logaddexp2(x, y)
    raise TypeError(_notimplemented_msg('logaddexp2', type(x)))

def fmod(x, y): #needs work
    if isint(x) or isint(y):
        if isint(y) and handled_by_np(x):
            return np.fmod(x, float(y))
        if isint(x) and handled_by_np(y):
            return np.fmod(float(x), y)
        if isint(x) and isint(y):
            if y == 0:
                return 0
            signchange = (-1 if x < 0 else 1)*(-1 if y < 0 else 1)
            return signchange*(x % y)
    raise TypeError(_notimplemented_msg('fmod', type(x)))

def remainder(x, y): #needs work
    if isint(x) or isint(y):
        if isint(y) and handled_by_np(x):
            return np.remainder(x, float(y))
        if isint(x) and handled_by_np(y):
            return np.remainder(float(x), y)
        if isint(x) and isint(y):
            if y == 0:
                return 0
            return x % y
    raise TypeError(_notimplemented_msg('remainder', type(x)))

def ldexp(x, y): #same as fmod
    if isint(x) or isint(y):
        return x*(2**y)
    raise TypeError(_notimplemented_msg('ldexp', type(x)))

def trunc(x):
    if isint(x):
        return x
    raise TypeError(_notimplemented_msg('trunc', type(x)))

def ceil(x):
    if isint(x):
        return x
    raise TypeError(_notimplemented_msg('ceil', type(x)))

def floor(x):
    if isint(x):
        return x
    raise TypeError(_notimplemented_msg('floor', type(x)))

def rint(x):
    if isint(x):
        return x
    raise TypeError(_notimplemented_msg('rint', type(x)))

def _make_float_func(name):
    nfunc = getattr(np, name)
    def mathfunc(x):
        if isint(x):
            return nfunc(float(x))
        raise TypeError(_notimplemented_msg(name, type(x)))
    return mathfunc

_floatfuncs = ['fabs', 'arctan2', 'modf', 'frexp', 'arccos', 'arccosh',
               'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cos', 'sin', 'tan',
               'cosh', 'sinh', 'tanh', 'exp', 'exp2', 'expm1', 'log', 'log10',
               'log2', 'log1p', 'sqrt', 'spacing', ]

for _npy_name in _floatfuncs:
    globals()[_npy_name] = _make_float_func(_npy_name)
