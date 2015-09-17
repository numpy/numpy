"""
Pure python implementations of ufuncs for Object type.
Makes use of math and cmath module for some ufuncs.

"""
from __future__ import division, absolute_import, print_function
import math, cmath
from numpy.core.umath import pi as PI
import numpy as np
import sys

LOG2 = math.log(2)

if sys.version_info > (3,):
    # can be improved
    long = int

def _notimplemented_msg(fname, etype, etype2=None):
    if etype2 is not None:
        typestr = "({a}, {b})".format(a=etype, b=etype2)
    else:
        typestr = str(etype)
    return ("object-array fallback for ufunc '{name}' is not implemented for "
            "arguments of type {type}").format(name=fname, type=typestr)

def logical_and(x, y):
    return True if bool(x and y) else False

def logical_or(x, y):
    return True if bool(x or y) else False

def logical_xor(x, y):
    return True if bool((x or y) and not (x and y)) else False

def logical_not(x):
    return True if bool(not x) else False

def maximum(x, y):
    return max(x, y)

def minimum(x, y):
    return min(x, y)

def fmax(x, y):
    return x if (x >= y or _u_isnan(y)) else y

def fmin(x, y):
    return x if (x <= y or _u_isnan(y)) else y

def iscomplex(x):
    return isinstance(x, complex)

def isreal(x):
    return not isinstance(x, complex)

def isfinite(x):
    return True if (not _u_isinf(x) and not _u_isnan(x)) else False

def conjugate(x):
    return x.conjugate() if hasattr(x, 'conjugate') else x

def exp2(x):
    return 2.0**x

def log2(x):
    if type(x) in (bool, int, long, float):
        return math.log(x, 2)
    if type(x) is complex:
        return cmath.log(x, 2)
    return _u_log(x)/_u_log(type(x)(2)) # or notimplemented?

def degrees(x):
    return x*180/PI
rad2deg = degrees

def radians(x):
    return x*PI/180
deg2rad = radians

def square(x):
    return x*x

def reciprocal(x):
    return type(x)(1.0/x)

def cbrt(x):
    return x**(1.0/3)

def _ones_like(x):
    return 1

def sign(x):
    # np.sign does not handle bool
    if type(x) in (int, long, float):
        if x > 0:
            return type(x)(1)
        if x < 0:
            return type(x)(-1)
        if x == 0:
            return type(x)(0)
        return x
    if type(x) is complex:
        if math.isnan(x.real) or math.isnan(x.imag):
            return complex(np.nan)
        if x.real > 0:
            return complex(1)
        if x.real < 0:
            return complex(-1)
        if x.imag > 0:
            return complex(1)
        if x.imag < 0:
            return complex(-1)
        if x == 0:
            return complex(0)
        return x #should never happen

    raise TypeError(_notimplemented_msg('sign', type(x)))


def hypot(x, y):
    if type(x) in (bool, int, long, float):
        return math.hypot(x, y)
    return _u_sqrt(x**2 + y**2)

# the following only have implementation for builtin types

def nextafter(x):
    raise TypeError(_notimplemented_msg('nextafter', type(x)))

def spacing(x):
    raise TypeError(_notimplemented_msg('spacing', type(x)))

def signbit(x):
    if type(x) in (bool, int, long, float):
        return math.copysign(1, x) < 0
    raise TypeError(_notimplemented_msg('signbit', type(x)))

def copysign(x, y):
    if (    type(x) in (bool, int, long, float) and 
            type(y) in (bool, int, long, float) ):
        return math.copysign(x, y)
    raise TypeError(_notimplemented_msg('copysign', type(x)))

def logaddexp(x,y):
    if type(x) is complex or type(x) is complex:
        return cmath.log(cmath.exp(x) + cmath.exp(y))
    if (    type(x) not in (bool, int, long, float) or
            type(y) not in (bool, int, long, float) ):
        raise TypeError(_notimplemented_msg('logaddexp', type(x)))

    if x == y:
        return x + LOG2
    if x > y:
        return x + math.log1p(math.exp(y-x))
    if x < y:
        return y + math.log1p(math.exp(x-y))
    return x-y

def logaddexp2(x,y):
    if type(x) is complex or type(x) is complex:
        return cmath.log(2**x + 2**y)/LOG2
    if type(x) not in (bool, int, long, float):
        raise TypeError(_notimplemented_msg('logaddexp2', type(x)))

    if x == y:
        return x + 1
    if x > y:
        return x + math.log1p(math.pow(2, y-x))/LOG2
    if x < y:
        return y + math.log1p(math.pow(2, x-y))/LOG2
    return x-y

def fmod(x, y):
    if (    type(x) in (bool, int, long, float) and 
            type(y) in (bool, int, long, float) ):
        return math.fmod(x, y)
    raise TypeError(_notimplemented_msg('fmod', type(x), type(y)))

def ldexp(x, y):
    if (    type(x) in (bool, int, long, float) and 
            type(y) in (bool, int, long, float) ):
        return math.ldexp(x, y)
    raise TypeError(_notimplemented_msg('ldexp', type(x), type(y)))

def expm1(x):
    if type(x) in (bool, int, long, float):
        return math.expm1(x)
    if type(x) is complex:
        return cmath.exp(x)-1
    raise TypeError(_notimplemented_msg('expm1', type(x)))

def log1p(x):
    if type(x) in (bool, int, long, float):
        return math.log1p(x)
    if type(x) is complex:
        return cmath.log(x + 1)
    raise TypeError(_notimplemented_msg('log1p', type(x)))

def isnan(x):
    if type(x) in (bool, int, long, float):
        return math.isnan(x)
    if type(x) is complex:
        return cmath.isnan(x)
    raise TypeError(_notimplemented_msg('isnan', type(x)))

def isinf(x):
    if type(x) in (bool, int, long, float):
        return math.isinf(x)
    if type(x) is complex:
        return cmath.isinf(x)
    raise TypeError(_notimplemented_msg('isinf', type(x)))

def _makecMathFunc(fname):
    mfunc = getattr(math, fname)
    cfunc = getattr(cmath, fname)
    def cmathfunc(x):
        if type(x) in (bool, int, long, float):
            return mfunc(x)
        if type(x) is complex:
            return cfunc(x)
        raise TypeError(_notimplemented_msg(fname, type(x)))
    return cmathfunc

def _makeMathFunc(fname):
    mfunc = getattr(math, fname)
    def mathfunc(x):
        if type(x) in (bool, int, long, float):
            return mfunc(x)
        raise TypeError(_notimplemented_msg(fname, type(x)))
    return mathfunc

def trunc(x):
    if type(x) in (bool, int, long, float):
        if _u_isinf(x) or _u_isnan(x):
            return x
        return math.trunc(x)
    raise TypeError(_notimplemented_msg('trunc', type(x)))

def ceil(x):
    if type(x) in (bool, int, long, float):
        if _u_isinf(x) or _u_isnan(x):
            return x
        return math.ceil(x)
    raise TypeError(_notimplemented_msg('ceil', type(x)))

def floor(x):
    if type(x) in (bool, int, long, float):
        if _u_isinf(x) or _u_isnan(x):
            return x
        return math.floor(x)
    raise TypeError(_notimplemented_msg('floor', type(x)))

def _roundnan(x):
        if _u_isinf(x) or _u_isnan(x):
            return x
        return round(x)

def rint(x):
    if type(x) in (bool, int, long, float):
        return _roundnan(x)
    if type(x) is complex:
        return complex(_roundnan(x.real), _roundnan(x.imag))
    raise TypeError(_notimplemented_msg('rint', type(x)))


_mathfuncs = [('fabs',     'fabs'),
              ('arctan2',  'atan2'),
              ('modf',     'modf'),
              ('frexp',    'frexp'),
              ]

for _npy_name, _math_name in _mathfuncs:
    globals()[_npy_name] = _makeMathFunc(_math_name)

_cmathfuncs = [('arccos',   'acos'), 
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

for _npy_name, _cmath_name in _cmathfuncs:
    globals()[_npy_name] = _makecMathFunc(_cmath_name)

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
