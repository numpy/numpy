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

    bool

    object

    void, string, unicode 

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int, uint,
    longlong, ulonglong,

    single, csingle,
    float, complex,
    longfloat, clongfloat,

    As part of the type-hierarchy:    xx -- is bit-width
    
     generic
       bool
       numeric
         integer
           signedinteger   (intxx)
             byte
             short
             int
             intp           int0
             longint
             longlong
           unsignedinteger  (uintxx)
             ubyte
             ushort
             uint
             uintp          uint0
             ulongint
             ulonglong
         floating           (floatxx)
             single          
             float  (double)
             longfloat
         complexfloating    (complexxx)
             csingle        
             complex (cfloat, cdouble)
             clongfloat
   
       flexible
         character
           string
           unicode
         void
   
       object

$Id: numerictypes.py,v 1.17 2005/09/09 22:20:06 teoliphant Exp $
"""

import multiarray
typeinfo = multiarray.typeinfo
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
complex = cdouble
int0 = intp
uint0 = uintp
single = float
csingle = cfloat
float = double
intc = int
uintc = uint
int = long
uint = ulong
cfloat = cdouble
longfloat = longdouble
clongfloat = clongdouble
long = pylong

del pylong, ulong

del _thisdict, _tocheck, a, name, typeobj
del base, bit, char


arraytypes = {'int': [],
              'uint':[],
              'float':[],
              'complex':[],
              'others':[bool,object,string,unicode,void]}

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

def maximum_type(t):
    """returns the type of highest precision of the same general kind as 't'"""
    if not issubclass(t, generic):
        return t
    else:
        name = t.__name__[:-8]
        base, bits = _evalname(name)
        if bits == 0:
            return t
        else:
            return arraytypes[base][-1]

def python_type(t):
    """returns the type corresponding to a certain Python type"""
    if not isinstance(t, _types.TypeType):
        t = type(t)
    if t == _types.IntType:
        return long
    elif t == _types.FloatType:
        return float
    elif t == _types.ComplexType:
        return complex
    elif t == _types.StringType:
        return string
    elif t == _types.UnicodeType:
        return unicode
    else:
        return object

def istype(rep):
    """Determines whether the given object represents
    a numeric array type."""
    return issubclass(rep, generic) or typeDict.has_key(rep)


def totype(rep, default=None):
    try:
        if issubclass(rep, generic):
            return rep
        if isinstance(rep, type):
            return python_type(rep)
        res = typeDict.get(rep, default)
    except:
        res = default
    return res


del _ibytes, _fbytes, multiarray





