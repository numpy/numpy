"""
Pure python implementations of ufuncs for Object type.
Makes use of math and cmath module for some ufuncs.

"""
from __future__ import division, absolute_import, print_function
import math, cmath
from numpy.core.umath import pi as PI
import numpy as np

LOG2 = math.log(2)

def _notimplemented_msg(fname, etype, etype2=None):
    return ("fallback for ufunc '{}' is not implemented for objects of type "
            "{}{}").format(fname, etype, '' if etype2 is None else (' '+etype2))

def logical_and(x, y):
    return True if (x and y) else False

def logical_or(x, y):
    return True if (x or y) else False

def logical_xor(x, y):
    return True if ((x or y) and not (x and y)) else False

def logical_not(x):
    return True if (not x) else False

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
    if isinstance(x, (int, long, float)):
        return math.log(x, 2)
    if isinstance(x, complex):
        return cmath.log(x, 2)
    return _u_log(x)/_u_log(type(x)(2)) # or notimplemented?

def degrees(x):
    return x*180/PI

def radians(x):
    return x*PI/180

def square(x):
    return x*x

def reciprocal(x):
    return type(x)(1.0/x)

def cbrt(x):
    return x**(1.0/3)

def sign(x):
    if x > 0:
        return type(x)(1)
    if x < 0:
        return type(x)(-1)
    if x == 0:
        return type(x)(0)
    return x

def hypot(x, y):
    if isinstance(x, (int, long, float)):
        return math.hypot(x, y)
    return _u_sqrt(x**2 + y**2)

# the following only have implementation for builtin types

def rint(x):
    if isinstance(x, (int, long, float)):
        return int(round(x))
    if isinstance(x, complex):
        return complex(int(round(x.real)), int(round(x.imag)))
    raise TypeError(_notimplemented_msg('rint', type(x)))

def nextafter(x):
    raise TypeError(_notimplemented_msg('nextafter', type(x)))

def spacing(x):
    raise TypeError(_notimplemented_msg('spacing', type(x)))

def signbit(x):
    if isinstance(x, (int, long, float)):
        return math.copysign(1, x) < 0
    raise TypeError(_notimplemented_msg('signbit', type(x)))

def logaddexp(x,y):
    if isinstance(x, complex):
        return cmath.log(cmath.exp(x) + cmath.exp(y))
    if not isinstance(x, (int, long, float)):
        raise TypeError(_notimplemented_msg('logaddexp', type(x)))

    if x == y:
        return x + LOG2
    if x > y:
        return x + math.log1p(math.exp(y-x))
    if x < y:
        return y + math.log1p(math.exp(x-y))
    return x-y

def logaddexp2(x,y):
    if isinstance(x, complex):
        return cmath.log(2**x + 2**y)/LOG2
    if not isinstance(x, (int, long, float)):
        raise TypeError(_notimplemented_msg('logaddexp2', type(x)))

    if x == y:
        return x + 1
    if x > y:
        return x + math.log1p(math.pow(2, y-x))/LOG2
    if x < y:
        return y + math.log1p(math.pow(2, x-y))/LOG2
    return x-y

def fmod(x, y):
    if isinstance(x, (int, long, float)):
        return math.fmod(x, y)
    raise TypeError(_notimplemented_msg('fmod', type(x), type(y)))

def ldexp(x, y):
    if isinstance(x, (int, long, float)):
        return math.ldexp(x, y)
    raise TypeError(_notimplemented_msg('ldexp', type(x), type(y)))

def expm1(x):
    if isinstance(x, (int, long, float)):
        return math.expm1(x)
    if isinstance(x, complex):
        return cmath.exp(x)-1
    raise TypeError(_notimplemented_msg('expm1', type(x)))

def log1p(x):
    if isinstance(x, (int, long, float)):
        return math.log1p(x)
    if isinstance(x, complex):
        return cmath.log(x + 1)
    raise TypeError(_notimplemented_msg('log1p', type(x)))

def isnan(x):
    try:
        return math.isnan(x)
    except:
        pass
    try:
        return cmath.isnan(x)
    except:
        pass
    return False

def isinf(x):
    try:
        return math.isinf(x)
    except:
        pass
    try:
        return cmath.isinf(x)
    except:
        pass
    return False

def _makecMathFunc(fname):
    mfunc = getattr(math, fname)
    cfunc = getattr(cmath, fname)
    def cmathfunc(x):
        if isinstance(x, (int, long, float)):
            return mfunc(x)
        if isinstance(x, complex):
            return cfunc(x)
        raise TypeError(_notimplemented_msg(fname, type(x)))
    return cmathfunc

def _makeMathFunc(fname):
    mfunc = getattr(math, fname)
    def mathfunc(x):
        if isinstance(x, (int, long, float)):
            return mfunc(x)
        raise TypeError(_notimplemented_msg(fname, type(x)))
    return mathfunc

_mathfuncs = [('ceil',     'ceil'),
              ('trunc',    'trunc'),
              ('floor',    'floor'),
              ('fabs',     'fabs'),
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
