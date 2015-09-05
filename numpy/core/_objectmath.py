"""
Pure python implementations of ufuncs for Object type.
Makes use of math and cmath module for some ufuncs.

"""
from __future__ import division, absolute_import, print_function
import math, cmath

LOG2 = math.log(2)

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
    return x if (x >= y or math.isnan(y)) else y

def fmin(x, y):
    return x if (x <= y or math.isnan(y)) else y

def iscomplex(x):
    return isinstance(x, complex)

def isreal(x):
    return not isinstance(x, complex)

def isfinite(x):
    if isinstance(x, complex):
        return not cmath.isinf(x) and not cmath.isnan(x)
    return not math.isinf(x) and not math.isnan(x)

def conjugate(x):
    return x.conjugate() if hasattr(x, 'conjugate') else x

def exp2(x):
    if isinstance(x, complex):
        return cmath.pow(2, x)
    return math.pow(2, x)

def log2(x):
    if isinstance(x, complex):
        return cmath.log(x, 2)
    return math.log(x, 2)

def degrees(x):
    return x*180/np.pi

def radians(x):
    return x*np.pi/180

def square(x):
    return x*x

def reciprocal(x):
    return type(x)(1.0/x)

def rint(x):
    return int(round(x))

def nextafter(x):
    raise Exception("Not done")

def spacing(x):
    raise Exception("Not done")

def cbrt(x):
    raise Exception("Not done")

def signbit(x):
    if isinstance(x, complex):
        return cmath.copysign(1, x) < 0
    return math.copysign(1, x) < 0

def logaddexp(x,y):
    if x == y:
        return x + LOG2
    if x > y:
        return x + math.log1p(math.exp(y-x))
    if x < y:
        return y + math.log1p(math.exp(x-y))
    return x-y

def logaddexp2(x,y):
    if x == y:
        return x + 1
    if x > y:
        return x + math.log1p(math.pow(2, y-x))/LOG2
    if x < y:
        return y + math.log1p(math.pow(2, x-y))/LOG2
    return x-y

def sign(x):
    if x > 0:
        return type(x)(1)
    if x < 0:
        return type(x)(-1)
    if x == 0:
        return type(x)(0)
    return x

def fmod(a, b):
    return math.fmod(a, b)

def hypot(a, b):
    return math.hypot(a, b)

def expm1(a):
    if isinstance(x, complex):
        return cmath.exp(x)-1
    return math.expm1(x)

def isnan(a):
    try:
        return cmath.isnan(a)
    except:
        return False

def isinf(a):
    try:
        return cmath.isinf(a)
    except:
        return False


def _makecMathFunc(fname):
    mfunc = getattr(math, fname)
    cfunc = getattr(cmath, fname)
    def cmathfunc(x):
        if isinstance(x, complex):
            return cfunc(x)
        return mfunc(x)
    return cmathfunc

def _makeMathFunc(fname):
    mfunc = getattr(math, fname)
    def mathfunc(x):
        return mfunc(x)
    return mathfunc

_mathfuncs = [('ceil',     'ceil'),
              ('trunc',    'trunc'),
              ('floor',    'floor'),
              ('fabs',     'fabs'),
              ('arctan2',  'atan2'),
              ('hypot',    'hypot'),
              ('ldexp',    'ldexp'),
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

