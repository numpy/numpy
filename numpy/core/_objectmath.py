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
    def islong(x):
        return (x is int) and (x >= 2**64) #biggest numpy integer
else:
    def islong(x):
        return (x in (int, long)) and (x >= 2**64)

def numpy_knows(x):
    return np.min_scalar_type(x) != np.dtype("O") #better way?

def _notimplemented_msg(fname, etype, etype2=None):
    if etype2 is not None:
        typestr = "({a}, {b})".format(a=etype, b=etype2)
    else:
        typestr = str(etype)
    return ("object-array fallback for ufunc '{name}' is not implemented for "
            "arguments of type {type}").format(name=fname, type=typestr)


def logical_and(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.logical_and(x, y).item()
    return bool(x and y)

def logical_or(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.logical_or(x, y).item()
    return bool(x or y)

def logical_xor(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.logical_xor(x, y).item()
    return bool(x or y) and not bool(x and y)

def logical_not(x):
    if numpy_knows(x):
        return np.logical_not(x).item()
    return bool(not x)

def maximum(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.maximum(x, y).item()
    return max(x, y)

def minimum(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.minimum(x, y).item()
    return min(x, y)

def fmax(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.fmax(x, y).item()
    return min(x, y)

def fmin(x, y):
    if numpy_knows(x) and numpy_knows(y):
        return np.fmin(x, y).item()
    return min(x, y)

def iscomplex(x):
    if numpy_knows(x):
        return np.iscomplex(x).item()
    if islong(x) or islong(y):
        return False
    raise TypeError

def isreal(x):
    if numpy_knows(x):
        return np.isreal(x).item()
    if islong(x):
        return True
    raise TypeError

def isfinite(x):
    if numpy_knows(x):
        return np.isfinite(x).item()
    if islong(x):
        return True
    return not _u_isinf(x) and not _u_isnan(x)

def conjugate(x):
    if numpy_knows(x):
        return np.conjugate(x).item()
    if islong(x):
        return x
    raise TypeError

def exp2(x):
    if numpy_knows(x):
        return np.exp2(x).item()
    if islong(x):
        return 2**x
    raise TypeError

def log2(x):
    if numpy_knows(x):
        return np.log2(x).item()
    if islong(x):
        return math.log(x, 2)/math.log(2)
    raise TypeError

def degrees(x):
    return x*180/PI
rad2deg = degrees

def radians(x):
    return x*PI/180
deg2rad = radians

def square(x):
    return x*x

def reciprocal(x):
    if numpy_knows(x):
        return np.reciprocal(x).item()
    if islong(x):
        return 0 if x != 1 else 1
    raise TypeError

def cbrt(x):
    if numpy_knows(x):
        return np.cbrt(x).item()
    if islong(x):
        if x < 0:
            return -(-x)**(1.0/3)
        return x**(1.0/3)
    raise TypeError

def _ones_like(x):
    return 1

def sign(x):
    if numpy_knows(x):
        return np.sign(x).item()
    if islong(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
        return x
    raise TypeError

def hypot(x, y):
    if numpy_knows(x):
        return np.hypot(x, y).item()
    if islong(x) or islong(y):
        return math.hypot(x, y)
    raise TypeError

# the following only have implementation for builtin types

def nextafter(x):
    if islong(x):
        x = float(x)
    if numpy_knows(x):
        return np.nextafter(x).item()
    raise TypeError

def spacing(x):
    if islong(x):
        x = float(x)
    if numpy_knows(x):
        return np.spacing(x).item()
    raise TypeError

def signbit(x):
    if numpy_knows(x):
        return np.signbit(x).item()
    if islong(x):
        return math.copysign(1, x) < 0
    raise TypeError

def copysign(x, y):
    if numpy_knows(x):
        return np.copysign(x, y).item()
    if islong(x) or islong(y):
        return math.copysign(x, y)
    raise TypeError

def logaddexp(x, y):
    if islong(x):
        x = float(x)
    if islong(y):
        y = float(y)
    if numpy_knows(x):
        return np.logaddexp(x, y).item()
    raise TypeError

def logaddexp2(x, y):
    if islong(x):
        x = float(x)
    if islong(y):
        y = float(y)
    if numpy_knows(x):
        return np.logaddexp2(x, y).item()
    raise TypeError

def fmod(x, y): # correct? or try to return a long?
    if islong(x):
        x = float(x) 
    if islong(y):
        y = float(y)
    if numpy_knows(x):
        return np.fmod(x, y).item()
    raise TypeError

def ldexp(x, y): #same as fmod
    if islong(x):
        x = float(x) 
    if islong(y):
        y = float(y)
    if numpy_knows(x):
        return np.ldexp(x, y).item()
    raise TypeError

def expm1(x):
    if islong(x):
        x = float(x) 
    if numpy_knows(x):
        return np.expm1(x).item()
    raise TypeError

def log1p(x):
    if islong(x):
        x = float(x) 
    if numpy_knows(x):
        return np.log1p(x).item()
    raise TypeError

def isnan(x):
    if numpy_knows(x):
        return np.isnan(x).item()
    if islong(x):
        return False
    raise TypeError

def isinf(x):
    if numpy_knows(x):
        return np.isinf(x).item()
    if islong(x):
        return False
    raise TypeError

def trunc(x):
    if numpy_knows(x):
        return np.trunc(x).item()
    if islong(x):
        return math.trunc(x)
    raise TypeError

def ceil(x):
    if numpy_knows(x):
        return np.ceil(x).item()
    if islong(x):
        return math.ceil(x)
    raise TypeError

def floor(x):
    if numpy_knows(x):
        return np.floor(x).item()
    if islong(x):
        return math.floor(x)
    raise TypeError

def rint(x):
    if numpy_knows(x):
        return np.rint(x).item()
    if islong(x):
        return x
    raise TypeError

def _make_float_func(nname, fname):
    nfunc = getattr(np, nname)
    mfunc = getattr(math, fname)
    def mathfunc(x):
        if islong(x):
            x = float(x) 
        if numpy_knows(x):
            return nfunc(x)
        raise TypeError
    return mathfunc

_mathfuncs = [ ('fabs',     'fabs'),
               ('arctan2',  'atan2'),
               ('modf',     'modf'),
               ('frexp',    'frexp'),
               ('arccos',   'acos'), 
               ('arccosh',  'acosh'), 
               ('arcsin',   'asin'), 
               ('arcsinh',  'asinh'), 
               ('arctan',   'atan'),
               ('arctanh',  'atanh'), 
               ('cos',      'cos'), 
               ('sin',      'sin'), 
               ('tan',      'tan'), 
               ('cosh',     'cosh'), 
               ('sinh',     'sinh'),
               ('tanh',     'tanh'),
               ('exp',      'exp'),
               ('log',      'log'),
               ('log10',    'log10'),
               ('sqrt',     'sqrt'),
              ]

for _npy_name, _math_name in _mathfuncs:
    globals()[_npy_name] = _make_float_func(_npy_name, _math_name)

# sometimes one ufunc wants to call another ufunc. To avoid going through all
# of numpy's machinery again, this function creates a version of the ufunc that
# dispatches the call appropriately
def loopback_unary_ufunc(ufuncname):
    npufunc = getattr(np, ufuncname)
    objimpl = globals()[ufuncname]

    def _ufunc(x):
        if isinstance(x, (np.ndarray, np.generic)):
            return npufunc(x)
        methfunc = getattr(x, ufuncname, None)
        if methfunc and callable(methfunc):
            return methfunc()
        return objimpl(x)
    return _ufunc

for uname in ['isnan', 'isinf', 'log', 'exp', 'sqrt']:
    globals()['_u_'+uname] = loopback_unary_ufunc(uname)
