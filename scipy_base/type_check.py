
import types
import Numeric
from fastumath import isinf, isnan, isfinite
from Numeric import ArrayType, array, multiarray

__all__ = ['ScalarType','iscomplexobj','isrealobj','imag','iscomplex',
           'isscalar','isneginf','isposinf','isnan','isinf','isfinite',
           'isreal','nan_to_num','real','real_if_close',
           'typename','cast','common_type','typecodes', 'asarray']

def asarray(a, typecode=None, savespace=None):
   """asarray(a,typecode=None, savespace=0) returns a as a NumPy array.
   Unlike array(), no copy is performed if a is already an array.
   """
   if type(a) is ArrayType:
      if typecode is None or typecode == a.typecode():
         if savespace is None or a.spacesaver()==savespace:
            return a
      else:
         r = a.astype(typecode)
         if not (savespace is None or a.spacesaver()==savespace):
            r.savespace(savespace)
         return r
   return multiarray.array(a,typecode,copy=0,savespace=savespace or 0)

ScalarType = [types.IntType, types.LongType, types.FloatType, types.ComplexType]

typecodes = Numeric.typecodes
typecodes['AllInteger'] = '1silbwu'

try:
   Char = Numeric.Character
except AttributeError:
   Char = 'c'

toChar = lambda x: asarray(x).astype(Char)
toInt8 = lambda x: asarray(x).astype(Numeric.Int8)# or use variable names such as Byte
toUInt8 = lambda x: asarray(x).astype(Numeric.UnsignedInt8)
_unsigned = 0
if hasattr(Numeric,'UnsignedInt16'):
   toUInt16 = lambda x: asarray(x).astype(Numeric.UnsignedInt16)
   toUInt32 = lambda x: asarray(x).astype(Numeric.UnsignedInt32)
   _unsigned = 1
   
toInt16 = lambda x: asarray(x).astype(Numeric.Int16)
toInt32 = lambda x: asarray(x).astype(Numeric.Int32)
toInt = lambda x: asarray(x).astype(Numeric.Int)
toFloat32 = lambda x: asarray(x).astype(Numeric.Float32)
toFloat64 = lambda x: asarray(x).astype(Numeric.Float64)
toComplex32 = lambda x: asarray(x).astype(Numeric.Complex32)
toComplex64 = lambda x: asarray(x).astype(Numeric.Complex64)

# This is for pre Numeric 21.x compatiblity. Adding it is harmless.
if  not hasattr(Numeric,'Character'):
    Numeric.Character = 'c'
        
cast = {Numeric.Character: toChar,
        Numeric.UnsignedInt8: toUInt8,
        Numeric.Int8: toInt8,
        Numeric.Int16: toInt16,
        Numeric.Int32: toInt32,
        Numeric.Int: toInt,
        Numeric.Float32: toFloat32,
        Numeric.Float64: toFloat64,
        Numeric.Complex32: toComplex32,
        Numeric.Complex64: toComplex64}

if _unsigned:
   cast[Numeric.UnsignedInt16] = toUInt16
   cast[Numeric.UnsignedInt32] = toUInt32
   

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

def iscomplexobj(x):
    return asarray(x).typecode() in ['F', 'D']

def isrealobj(x):
    return not asarray(x).typecode() in ['F', 'D']

#-----------------------------------------------------------------------------

##def isnan(val):
##    # fast, but apparently not portable (according to notes by Tim Peters)
##    #return val != val
##    # very slow -- should really use cephes methods or *something* different
##    import ieee_754
##    vals = ravel(val)
##    if array_iscomplex(vals):
##        r = array(map(ieee_754.isnan,real(vals)))        
##        i = array(map(ieee_754.isnan,imag(vals)))
##        results = Numeric.logical_or(r,i)
##    else:        
##        results = array(map(ieee_754.isnan,vals))
##    if isscalar(val):
##        results = results[0]
##    return results

def isposinf(val):
    return isinf(val) & (val > 0)
    
def isneginf(val):
    return isinf(val) & (val < 0)
    
##def isinf(val):
##    return Numeric.logical_or(isposinf(val),isneginf(val))

##def isfinite(val):
##    vals = asarray(val)
##    if iscomplexobj(vals):
##        r = isfinite(real(vals))
##        i = isfinite(imag(vals))
##        results = Numeric.logical_and(r,i)
##    else:    
##        fin = Numeric.logical_not(isinf(val))
##        an = Numeric.logical_not(isnan(val))
##        results = Numeric.logical_and(fin,an)
##    return results        

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
        x = asarray(x)
        are_inf = isposinf(x)
        are_neg_inf = isneginf(x)
        are_nan = isnan(x)
        choose_array = are_neg_inf + are_nan * 2 + are_inf * 3
        y = Numeric.choose(choose_array,
                   (x,limits.double_min, 0., limits.double_max))
    return y

#-----------------------------------------------------------------------------

def real_if_close(a,tol=1e-13):
    a = asarray(a)
    if a.typecode() in ['F','D'] and Numeric.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a


#-----------------------------------------------------------------------------

_namefromtype = {'c' : 'character',
                 '1' : 'signed char',
                 'b' : 'unsigned char',
                 's' : 'short',
                 'w' : 'unsigned short',
                 'i' : 'integer',
                 'u' : 'unsigned integer',
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

if __name__ == '__main__':
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'double epsilon:',double_epsilon
    print 'double tiny:',double_tiny
