# Borrowed and adapted from numarray

"""numerictypes: Define the numeric type objects

This module is designed so 'from numerictypes import *' is safe.
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

# we add more at the bottom
__all__ = ['typeDict', 'arraytypes', 'ScalarType', 'obj2dtype', 'cast', 'nbytes']

from multiarray import typeinfo, ndarray, array
import types as _types

# we don't export these for import *, but we do want them accessible
# as numerictypes.bool, etc.
from __builtin__ import bool, int, long, float, complex, object, unicode, str

typeDict = {}      # Contains all leaf-node numeric types with aliases
allTypes = {}      # Collect the types we will add to the module here

def _evalname(name):
    k = 0
    for ch in name:
        if ch in '0123456789':
            break
        k += 1
    try:
        bits = int(name[k:])
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
        info = typeinfo[name.upper()]
        assert(info[-1] == obj)  # sanity check
        bits = info[2]

    except KeyError:     # bit-width name
        base, bits = _evalname(name)
        char = base[0]

    if name == 'bool':
        char = 'b'
        base = 'bool'
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

def _add_types():
    for a in typeinfo.keys():
        name = a.lower()
        if isinstance(typeinfo[a], type(())):
            typeobj = typeinfo[a][-1]
                        
            # define C-name and insert typenum and typechar references also
            allTypes[name] = typeobj
            typeDict[name] = typeobj
            typeDict[typeinfo[a][0]] = typeobj
            typeDict[typeinfo[a][1]] = typeobj

            # insert bit-width version for this class (if relevant)
            base, bit, char = bitname(typeobj)

#            typeobj.bytes=bit/8

            revdict[typeobj] = (typeinfo[a][:-1], (base, bit, char), a)
            if base != '':
                allTypes["%s%d" % (base, bit)] = typeobj
                typeDict["%s%d" % (base, bit)] = typeobj
            if char != '':
                typeDict[char] = typeobj

        else:  # generic class
            allTypes[name] = typeinfo[a]
_add_types()


# We use these later
void = allTypes['void']
generic = allTypes['generic']

#
# Rework the Python names (so that float and complex and int are consistent
#                            with Python usage)
#
def _set_up_aliases():
    type_pairs = [('complex_', 'cdouble'),
                  ('int0', 'intp'),
                  ('uint0', 'uintp'),
                  ('single', 'float'),
                  ('csingle', 'cfloat'),
                  ('float_', 'double'),
                  ('intc', 'int'),
                  ('uintc', 'uint'),
                  ('int_', 'long'),
                  ('uint', 'ulong'),
                  ('cfloat', 'cdouble'),
                  ('longfloat', 'longdouble'),
                  ('clongfloat', 'clongdouble'),
                  ('bool_', 'bool'),
                  ('unicode_', 'unicode'),
                  ('str_', 'string'),
                  ('object_', 'object')]
    for alias, t in type_pairs:
        allTypes[alias] = allTypes[t]
    # Remove aliases overriding python types
    for t in ['ulong', 'object', 'unicode', 'int', 'long', 'float',
              'complex', 'bool']:
        try:
            del allTypes[t]
        except KeyError:
            pass
_set_up_aliases()

# Now, construct dictionary to lookup character codes from types

_dtype2char_dict = {}
def _construct_char_code_lookup():
    for name in typeinfo.keys():
        tup = typeinfo[name]
        if isinstance(tup, type(())):
            _dtype2char_dict[tup[-1]] = tup[0]
_construct_char_code_lookup()

arraytypes = {'int': [],
              'uint':[],
              'float':[],
              'complex':[],
              'others':[bool,object,str,unicode,void]}

def _add_array_type(typename, bits):
    try:
        t = allTypes['%s%d' % (typename, bits)]
    except KeyError:
        pass
    else:
        arraytypes[typename].append(t)

def _set_array_types():
    ibytes = [1, 2, 4, 8, 16, 32, 64]
    fbytes = [2, 4, 8, 10, 12, 16, 32, 64]
    for bytes in ibytes:
        bits = 8*bytes
        _add_array_type('int', bits)
        _add_array_type('uint', bits)
    for bytes in fbytes:
        bits = 8*bytes
        _add_array_type('float', bits)
        _add_array_type('complex', bits)
_set_array_types()

genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16',
                   'int32', 'uint32', 'int64', 'uint64', 'int128',
                   'uint128', 'float16',
                   'float32', 'float64', 'float80', 'float96', 'float128',
                   'float256',
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

_python_types = {int : 'int_',
                 float: 'float_',
                 complex: 'complex_',
                 bool: 'bool_',
                 str: 'string',
                 unicode: 'unicode_',
                 _types.BufferType: 'void',
                }
def _python_type(t):
    """returns the type corresponding to a certain Python type"""
    if not isinstance(t, _types.TypeType):
        t = type(t)
    return allTypes[_python_types.get(t, 'object_')]

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


# Create dictionary of casting functions that wrap sequences
# indexed by type or type character

# This dictionary allows look up based on any alias for a type
class _typedict(dict):
    def __getitem__(self, obj):
        return dict.__getitem__(self, obj2dtype(obj))

cast = _typedict()
ScalarType = [_types.IntType, _types.FloatType,
              _types.ComplexType, _types.LongType, _types.BooleanType,
              _types.StringType, _types.UnicodeType, _types.BufferType]
ScalarType.extend(_dtype2char_dict.keys())
ScalarType = tuple(ScalarType)
for key in _dtype2char_dict.keys():
    cast[key] = lambda x, k=key : array(x, copy=False).astype(k)

nbytes = _typedict()
def _construct_nbytes_lookup():
    for name, val in typeinfo.iteritems():
        if not isinstance(val, tuple):
            continue
        nbytes[val[-1]] = val[2] / 8
_construct_nbytes_lookup()

# Now add the types we've determined to this module
for key in allTypes:
    globals()[key] = allTypes[key]
    __all__.append(key)

del key

