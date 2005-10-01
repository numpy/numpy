# Borrowed and adapted from numarray

"""numerictypes: Define the numeric type objects

This module is designed so 'from numeric3types import *' is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    typeDict

  Type objects (not all will be available, depends on platform):
      see variable arraytypes for which ones you have

    Bit-width names
    
    int8 int16 int32 int64 int128
    uint8 uint16 uint32 uint64 uint128
    float16 float32 float64 float96 float128 float256
    complex32 complex64 complex128 complex192 complex256 complex512

    c-based names 

    bool_

    object_

    void, str_, unicode_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    float_, complex_,
    longfloat, clongfloat,

    As part of the type-hierarchy:    xx -- is bit-width
    
     generic
       bool_
       numeric
         integer
           signedinteger   (intxx)
             byte
             short
             intc
             intp           int0
             int_
             longlong
           unsignedinteger  (uintxx)
             ubyte
             ushort
             uintc
             uintp          uint0
             uint_
             ulonglong
         floating           (floatxx)
             single          
             float_  (double)
             longfloat
         complexfloating    (complexxx)
             csingle        
             complex_ (cfloat, cdouble)
             clongfloat
   
       flexible
         character
           str_     (string)
           unicode_ 
         void
   
       object_

$Id: numerictypes.py,v 1.17 2005/09/09 22:20:06 teoliphant Exp $
"""

import multiarray
typeinfo = multiarray.typeinfo
ndarray = multiarray.ndarray
array = multiarray.array
_I = typeinfo

pybool = bool
pyint = int
pyfloat = float
pylong = long
pycomplex = complex
pyobject = object
pyunicode = unicode

import types as _types

typeDict = {}      # Contains all leaf-node numeric types with aliases


def _evalname(name):
    k = 0
    for ch in name:
        if ch in '0123456789':
            break
        k+=1
    try:
        bits = pyint(name[k:])
    except ValueError:
        bits = 0
    base = name[:k]
    return base, bits

def bitname(obj):
    """Return a bit-width name for a given type object"""
    name = obj.__name__[:-8]
    base = ''
    char = ''
    try:
        info = _I[name.upper()]
        assert(info[-1] == obj)  # sanity check
        bits = info[2]
        
    except KeyError:     # bit-width name
        base, bits = _evalname(name)
        char = base[0]

    if name=='bool':
        char = 'b'
    elif name=='string':
        char = 'S'
        base = 'string'
    elif name=='unicode':
        char = 'U'
        base = 'unicode'
    elif name=='void':
        char = 'V'
        base = 'void'
    elif name=='object':
        char = 'O'
        base = 'object'
        bits = 0

    bytes = bits / 8        

    if char != '' and bytes != 0:
        char = "%s%d" % (char, bytes)

    return base, bits, char  

revdict = {}
_tocheck = _I.keys()
_thisdict = globals()  # this will insert into module name space

for a in _tocheck:
    name = a.lower()
    if isinstance(_I[a],type(())):
        typeobj = _I[a][-1]
        # define C-name and insert typenum and typechar references also
        _thisdict[name] = typeobj
        typeDict[name] = typeobj
        typeDict[_I[a][0]] = typeobj
        typeDict[_I[a][1]] = typeobj

        # insert bit-width version for this class (if relevant)
        base, bit, char = bitname(typeobj)
        revdict[typeobj] = (_I[a][:-1], (base, bit, char), a)
        if base != '':
            _thisdict["%s%d" % (base, bit)] = typeobj
            typeDict["%s%d" % (base, bit)] = typeobj
        if char != '':
            typeDict[char] = typeobj
        
    else:  # generic class
        _thisdict[name] = _I[a]

#
# Rework the Python names (so that float and complex and int are consistent
#                            with Python usage)
#
complex_ = cdouble
int0 = intp
uint0 = uintp
single = float
csingle = cfloat
float_ = double
intc = int
uintc = uint
int_ = long
uint = ulong
cfloat = cdouble
longfloat = longdouble
clongfloat = clongdouble

bool_ = bool
unicode_ = unicode
str_ = string
object_ = object

object = pyobject
unicode = pyunicode
int = pyint
long = pylong
float = pyfloat
complex = pycomplex
bool = pybool

del ulong, pyobject, pyunicode, pyint, pylong, pyfloat, pycomplex, pybool

del _thisdict, _tocheck, a, name, typeobj
del base, bit, char


# Now, construct dictionary to lookup character codes from types

_dtype2char_dict = {}
for name in typeinfo.keys():
    tup = typeinfo[name]
    if isinstance(tup,type(())):
        _dtype2char_dict[tup[-1]] = tup[0]

arraytypes = {'int': [],
              'uint':[],
              'float':[],
              'complex':[],
              'others':[bool_,object_,str_,unicode_,void]}

_ibytes = [1,2,4,8,16,32,64]
_fbytes = [2,4,8,10,12,16,32,64]
              
for bytes in _ibytes:
    bits = 8*bytes
    try:
        arraytypes['int'].append(eval("int%d" % bits))
    except NameError:
        pass
    try:
        arraytypes['uint'].append(eval("uint%d" % bits))
    except NameError:
        pass

for bytes in _fbytes:
    bits = 8*bytes
    try:
        arraytypes['float'].append(eval("float%d" % bits))
    except NameError:
        pass
    try:
        arraytypes['complex'].append(eval("complex%d" % (2*bits,)))
    except NameError:
        pass

del bytes, bits

genericTypeRank = ['bool','int8','uint8','int16','uint16',
                   'int32', 'uint32', 'int64', 'uint64', 'int128',
                   'uint128','float16',
                   'float32','float64', 'float80', 'float96', 'float128',
                   'float256'
                   'complex32', 'complex64', 'complex128', 'complex160',
                   'complex192', 'complex256', 'complex512', 'object']

def maximum_dtype(t):
    """returns the type of highest precision of the same general kind as 't'"""
    g = obj2dtype(t)
    if g is None:
        return t
    t = g
    name = t.__name__[:-8]
    base, bits = _evalname(name)
    if bits == 0:
        return t
    else:
        return arraytypes[base][-1]

def _python_type(t):
    """returns the type corresponding to a certain Python type"""
    if not isinstance(t, _types.TypeType):
        t = type(t)
    if t == _types.IntType:
        return int_
    elif t == _types.FloatType:
        return float_
    elif t == _types.ComplexType:
        return complex_
    elif t == _types.BooleanType:
        return bool_
    elif t == _types.StringType:
        return str_
    elif t == _types.UnicodeType:
        return unicode_
    elif t == _types.BufferType:
        return void
    else:
        return object_

def isdtype(rep):
    """Determines whether the given object represents
    a numeric array type."""
    try:
        char = dtype2char(rep)
        return True
    except (KeyError, ValueError):
        return False
    
def obj2dtype(rep, default=None):
    try:
        if issubclass(rep, generic):
            return rep
    except TypeError:
        pass
        
    if isinstance(rep, type):
        return _python_type(rep)
    if isinstance(rep, ndarray):
        return rep.dtype    
    res = typeDict.get(rep, default)
    return res

def dtype2char(dtype):
    dtype = obj2dtype(dtype)
    if dtype is None:
        raise ValueError, "unrecognized type"
    return _dtype2char_dict[dtype]


del _ibytes, _fbytes, multiarray

# Create dictionary of casting functions that wrap sequences
# indexed by type or type character

# This dictionary allows look up based on any alias for a type
class _castdict(dict):
    def __getitem__(self, obj):
        return dict.__getitem__(self, obj2dtype(obj))

cast = _castdict()
ScalarType = [_types.IntType, _types.LongType, _types.FloatType,
              _types.StringType, _types.UnicodeType, _types.ComplexType,
              _types.BufferType]
ScalarType.extend(_dtype2char_dict.keys())
for key in _dtype2char_dict.keys():
    cast[key] = lambda x, k=key : array(x,copy=0).astype(k)

del key




