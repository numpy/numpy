import types
import Numeric
from Numeric import *
from fastumath import *

import limits

# For now, we'll use Tim Peter's IEEE module for NaN, Inf, -Inf comparison
# stuff.  Perhaps move to cephes in the future??
from ieee_754 import PINF,MINF,NAN

__all__ = ['ScalarType','array_iscomplex','array_isreal','imag','iscomplex',
           'isscalar','isneginf','isposinf','isnan','isinf','isfinite',
           'isreal','isscalar','nan_to_num','real','real_if_close',
           'typename']

ScalarType = [types.IntType, types.LongType, types.FloatType, types.ComplexType]

toChar = lambda x: Numeric.array(x, Numeric.Character)
toInt8 = lambda x: Numeric.array(x, Numeric.Int8)# or use variable names such as Byte
toInt16 = lambda x: Numeric.array(x, Numeric.Int16)
toInt32 = lambda x: Numeric.array(x, Numeric.Int32)
toInt = lambda x: Numeric.array(x, Numeric.Int)
toFloat32 = lambda x: Numeric.array(x, Numeric.Float32)
toFloat64 = lambda x: Numeric.array(x, Numeric.Float64)
toComplex32 = lambda x: Numeric.array(x, Numeric.Complex32)
toComplex64 = lambda x: Numeric.array(x, Numeric.Complex64)

cast = {Numeric.Character: toChar,
        Numeric.Int8: toInt8,
        Numeric.Int16: toInt16,
        Numeric.Int32: toInt32,
        Numeric.Int: toInt,
        Numeric.Float32: toFloat32,
        Numeric.Float64: toFloat64,
        Numeric.Complex32: toComplex32,
        Numeric.Complex64: toComplex64}

def isscalar(num):
    if isinstance(num, ArrayType):
        return len(num.shape) == 0 and num.typecode() != 'O'
    return type(num) in ScalarType

def real(val):
    aval = asarray(val)
    if aval.typecode() in ['F', 'D']:
        return aval.real
    else:
        return aval

def imag(val):
    aval = asarray(val)
    if aval.typecode() in ['F', 'D']:
        return aval.imag
    else:
        return array(0,aval.typecode())*aval

def iscomplex(x):
    return imag(x) != Numeric.zeros(asarray(x).shape)

def isreal(x):
    return imag(x) == Numeric.zeros(asarray(x).shape)

def array_iscomplex(x):
    return asarray(x).typecode() in ['F', 'D']

def array_isreal(x):
    return not asarray(x).typecode() in ['F', 'D']

#-----------------------------------------------------------------------------

def isnan(val):
    # fast, but apparently not portable (according to notes by Tim Peters)
    #return val != val
    # very slow -- should really use cephes methods or *something* different
    import ieee_754
    vals = ravel(val)
    if array_iscomplex(vals):
        r = array(map(ieee_754.isnan,real(vals)))        
        i = array(map(ieee_754.isnan,imag(vals)))
        results = Numeric.logical_or(r,i)
    else:        
        results = array(map(ieee_754.isnan,vals))
    return results

def isposinf(val):
    # complex not handled currently (and potentially ambiguous)
    #return Numeric.logical_and(isinf(val),val > 0)
    return val == PINF
    
def isneginf(val):
    # complex not handled currently (and potentially ambiguous)
    #return Numeric.logical_and(isinf(val),val < 0)
    return val == MINF
    
def isinf(val):
    return Numeric.logical_or(isposinf(val),isneginf(val))

def isfinite(val):
    vals = asarray(val)
    if array_iscomplex(vals):
        r = isfinite(real(vals))
        i = isfinite(imag(vals))
        results = Numeric.logical_and(r,i)
    else:    
        fin = Numeric.logical_not(isinf(val))
        an = Numeric.logical_not(isnan(val))
        results = Numeric.logical_and(fin,an)
    return results        
def nan_to_num(x):
    # mapping:
    #    NaN -> 0
    #    Inf -> limits.double_max
    #   -Inf -> limits.double_min
    # complex not handled currently
    import limits
    try:
        t = x.typecode()
    except AttributeError:
        t = type(x)
    if t in [types.ComplexType,'F','D']:    
        y = nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    else:    
        x = Numeric.asarray(x)
        are_inf = isposinf(x)
        are_neg_inf = isneginf(x)
        are_nan = isnan(x)
        choose_array = are_neg_inf + are_nan * 2 + are_inf * 3
        y = Numeric.choose(choose_array,
                   (x,limits.double_min, 0., limits.double_max))
    return y

#-----------------------------------------------------------------------------

def real_if_close(a,tol=1e-13):
    a = Numeric.asarray(a)
    if a.typecode() in ['F','D'] and Numeric.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a


#-----------------------------------------------------------------------------

_namefromtype = {'c' : 'character',
                 '1' : 'signed char',
                 'b' : 'unsigned char',
                 's' : 'short',
                 'i' : 'integer',
                 'l' : 'long integer',
                 'f' : 'float',
                 'd' : 'double',
                 'F' : 'complex float',
                 'D' : 'complex double',
                 'O' : 'object'
                 }

def typename(char):
    """Return an english name for the given typecode character.
    """
    return _namefromtype[char]

#-----------------------------------------------------------------------------

#determine the "minimum common type code" for a group of arrays.
array_kind = {'i':0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
array_type = [['f', 'd'], ['F', 'D']]
def common_type(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.typecode()
        kind = max(kind, array_kind[t])
        precision = max(precision, array_precision[t])
    return array_type[kind][precision]

#-----------------------------------------------------------------------------
# Test Routines
#-----------------------------------------------------------------------------

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

if __name__ == '__main__':
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'double epsilon:',double_epsilon
    print 'double tiny:',double_tiny
