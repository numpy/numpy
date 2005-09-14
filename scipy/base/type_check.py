import types
import numerix as _nx
from numerix import ArrayType, array, isinf, isnan, isfinite, UfuncType

__all__ = ['ScalarType','iscomplexobj','isrealobj','imag','iscomplex',
           'isscalar','isneginf','isposinf','isnan','isinf','isfinite',
           'isreal','nan_to_num','real','real_if_close',
           'typename','cast','common_type','typecodes', 'asarray',
           'asfarray','ArrayType','UfuncType','mintypecode']

_typecodes_by_elsize = 'DFdfluiwsb1c'
def mintypecode(typecodes,typeset='DFdf',default='d',savespace=0):
   """ Return a typecode in typeset such that for each
   t in typecodes
     array(typecode=typecode)[:] = array(typecode=t)
   is valid, looses no information, and array(typecode=typecode)
   element size is minimal unless when typecodes does not
   intersect with typeset then default is returned.
   As a special case, if savespace is False then 'D' is returned
   whenever typecodes contain 'F' and 'd'.
   If t in typecodes is not a string then t=t.typecode() is applied.
   """
   typecodes = [(type(t) is type('') and t) or asarray(t).typecode()\
                for t in typecodes]
   intersection = [t for t in typecodes if t in typeset]
   if not intersection:
      return default
   if not savespace and 'F' in intersection and 'd' in intersection:
      return 'D'
   l = []
   for t in intersection:
      i = _typecodes_by_elsize.index(t)
      l.append((i,t))
   l.sort()
   return l[0][1]

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
   return array(a,typecode,copy=0,savespace=savespace or 0)

def asfarray(a, typecode=None, savespace=None):
   """asfarray(a,typecode=None, savespace=0) returns a as a NumPy float array."""
   a = asarray(a,typecode,savespace)
   if typecode is None and a.typecode() not in 'CFfd':
      return a.astype('d')
   return a

ScalarType = [types.IntType, types.LongType, types.FloatType, types.ComplexType]

typecodes = _nx.typecodes
typecodes['AllInteger'] = '1silbwu'

try:
   Char = _nx.Character
except AttributeError:
   Char = 'c'

toChar = lambda x: asarray(x).astype(Char)
toInt8 = lambda x: asarray(x).astype(_nx.Int8)# or use variable names such as Byte
toUInt8 = lambda x: asarray(x).astype(_nx.UnsignedInt8)
_unsigned = 0
if hasattr(_nx,'UnsignedInt16'):
   toUInt16 = lambda x: asarray(x).astype(_nx.UnsignedInt16)
   toUInt32 = lambda x: asarray(x).astype(_nx.UnsignedInt32)
   _unsigned = 1
   
toInt16 = lambda x: asarray(x).astype(_nx.Int16)
toInt32 = lambda x: asarray(x).astype(_nx.Int32)
toInt = lambda x: asarray(x).astype(_nx.Int)
toFloat32 = lambda x: asarray(x).astype(_nx.Float32)
toFloat64 = lambda x: asarray(x).astype(_nx.Float64)
toComplex32 = lambda x: asarray(x).astype(_nx.Complex32)
toComplex64 = lambda x: asarray(x).astype(_nx.Complex64)

# This is for pre _nx 21.x compatiblity. Adding it is harmless.
if  not hasattr(_nx,'Character'):
    _nx.Character = 'c'
        
cast = {_nx.Character: toChar,
        _nx.UnsignedInt8: toUInt8,
        _nx.Int8: toInt8,
        _nx.Int16: toInt16,
        _nx.Int32: toInt32,
        _nx.Int: toInt,
        _nx.Float32: toFloat32,
        _nx.Float64: toFloat64,
        _nx.Complex32: toComplex32,
        _nx.Complex64: toComplex64}

if _unsigned:
   cast[_nx.UnsignedInt16] = toUInt16
   cast[_nx.UnsignedInt32] = toUInt32
   

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
    return imag(x) != _nx.zeros(asarray(x).shape)

def isreal(x):
    return imag(x) == _nx.zeros(asarray(x).shape)

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
##        results = _nx.logical_or(r,i)
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
##    return _nx.logical_or(isposinf(val),isneginf(val))

##def isfinite(val):
##    vals = asarray(val)
##    if iscomplexobj(vals):
##        r = isfinite(real(vals))
##        i = isfinite(imag(vals))
##        results = _nx.logical_and(r,i)
##    else:    
##        fin = _nx.logical_not(isinf(val))
##        an = _nx.logical_not(isnan(val))
##        results = _nx.logical_and(fin,an)
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
        y = _nx.choose(choose_array,
                   (x,limits.double_min, 0., limits.double_max))
    return y

#-----------------------------------------------------------------------------

def real_if_close(a,tol=1e-13):
    a = asarray(a)
    if a.typecode() in ['F','D'] and _nx.allclose(a.imag, 0, atol=tol):
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
