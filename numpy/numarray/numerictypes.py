"""numerictypes: Define the numeric type objects

This module is designed so 'from numerictypes import *' is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    typeDict

  Numeric type objects:
    Bool
    Int8 Int16 Int32 Int64
    UInt8 UInt16 UInt32 UInt64
    Float32 Double64
    Complex32 Complex64

  Numeric type classes:
    NumericType
      BooleanType
      SignedType
      UnsignedType
      IntegralType
        SignedIntegralType
        UnsignedIntegralType
      FloatingType
      ComplexType

$Id: numerictypes.py,v 1.55 2005/12/01 16:22:03 jaytmiller Exp $
"""

__all__ = ['NumericType','HasUInt64','typeDict','IsType',
           'BooleanType', 'SignedType', 'UnsignedType', 'IntegralType',
           'SignedIntegralType', 'UnsignedIntegralType', 'FloatingType',
           'ComplexType', 'AnyType', 'ObjectType', 'Any', 'Object',
           'Bool', 'Int8', 'Int16', 'Int32', 'Int64', 'Float32',
           'Float64', 'UInt8', 'UInt16', 'UInt32', 'UInt64',
           'Complex32', 'Complex64', 'Byte', 'Short', 'Int','Long',
           'Float', 'Complex', 'genericTypeRank', 'pythonTypeRank',
           'pythonTypeMap', 'scalarTypeMap', 'genericCoercions',
           'typecodes', 'genericPromotionExclusions','MaximumType',
           'getType','scalarTypes', 'typefrom']

MAX_ALIGN = 8
MAX_INT_SIZE = 8

import numpy
LP64 = numpy.intp(0).itemsize == 8

HasUInt64 = 1
try:
    numpy.int64(0)
except:
    HasUInt64 = 0

#from typeconv import typeConverters as _typeConverters
#import numinclude
#from _numerictype import _numerictype, typeDict

# Enumeration of numarray type codes
typeDict = {}

_tAny       = 0
_tBool      = 1
_tInt8      = 2
_tUInt8     = 3
_tInt16     = 4
_tUInt16    = 5
_tInt32     = 6
_tUInt32    = 7
_tInt64     = 8
_tUInt64    = 9
_tFloat32   = 10
_tFloat64   = 11
_tComplex32 = 12
_tComplex64 = 13
_tObject    = 14

def IsType(rep):
    """Determines whether the given object or string, 'rep', represents
    a numarray type."""
    return isinstance(rep, NumericType) or rep in typeDict

def _register(name, type, force=0):
    """Register the type object.  Raise an exception if it is already registered
    unless force is true.
    """
    if name in typeDict and not force:
        raise ValueError("Type %s has already been registered" % name)
    typeDict[name] = type
    return type


class NumericType(object):
    """Numeric type class

    Used both as a type identification and the repository of
    characteristics and conversion functions.
    """
    def __new__(type, name, bytes, default, typeno):
        """__new__() implements a 'quasi-singleton pattern because attempts
        to create duplicate types return the first created instance of that
        particular type parameterization,  i.e. the second time you try to
        create "Int32",  you get the original Int32, not a new one.
        """
        if name in typeDict:
            self = typeDict[name]
            if self.bytes != bytes or self.default != default or \
                   self.typeno != typeno:
                raise ValueError("Redeclaration of existing NumericType "\
                                 "with different parameters.")
            return self
        else:
            self = object.__new__(type)
            self.name = "no name"
            self.bytes = None
            self.default = None
            self.typeno = -1
            return self

    def __init__(self, name, bytes, default, typeno):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name
        self.bytes = bytes
        self.default = default
        self.typeno = typeno
        self._conv = None
        _register(self.name, self)

    def __getnewargs__(self):
        """support the pickling protocol."""
        return (self.name, self.bytes, self.default, self.typeno)

    def __getstate__(self):
        """support pickling protocol... no __setstate__ required."""
        False

class BooleanType(NumericType):
    pass

class SignedType:
    """Marker class used for signed type check"""
    pass

class UnsignedType:
    """Marker class used for unsigned type check"""
    pass

class IntegralType(NumericType):
    pass

class SignedIntegralType(IntegralType, SignedType):
    pass

class UnsignedIntegralType(IntegralType, UnsignedType):
    pass

class FloatingType(NumericType):
    pass

class ComplexType(NumericType):
    pass

class AnyType(NumericType):
    pass

class ObjectType(NumericType):
    pass

# C-API Type Any

Any = AnyType("Any", None, None, _tAny)

Object = ObjectType("Object", None, None, _tObject)

# Numeric Types:

Bool  = BooleanType("Bool", 1, 0, _tBool)
Int8  = SignedIntegralType( "Int8", 1, 0, _tInt8)
Int16 = SignedIntegralType("Int16", 2, 0, _tInt16)
Int32 = SignedIntegralType("Int32", 4, 0, _tInt32)
Int64 = SignedIntegralType("Int64", 8, 0, _tInt64)

Float32  = FloatingType("Float32", 4, 0.0, _tFloat32)
Float64  = FloatingType("Float64", 8, 0.0, _tFloat64)

UInt8  = UnsignedIntegralType( "UInt8", 1, 0, _tUInt8)
UInt16 = UnsignedIntegralType("UInt16", 2, 0, _tUInt16)
UInt32 = UnsignedIntegralType("UInt32", 4, 0, _tUInt32)
UInt64 = UnsignedIntegralType("UInt64", 8, 0, _tUInt64)

Complex32  = ComplexType("Complex32", 8,  complex(0.0), _tComplex32)
Complex64  = ComplexType("Complex64", 16, complex(0.0), _tComplex64)

Object.dtype = 'O'
Bool.dtype = '?'
Int8.dtype = 'i1'
Int16.dtype = 'i2'
Int32.dtype = 'i4'
Int64.dtype = 'i8'

UInt8.dtype = 'u1'
UInt16.dtype = 'u2'
UInt32.dtype = 'u4'
UInt64.dtype = 'u8'

Float32.dtype = 'f4'
Float64.dtype = 'f8'

Complex32.dtype = 'c8'
Complex64.dtype = 'c16'

# Aliases

Byte = _register("Byte",   Int8)
Short = _register("Short",  Int16)
Int = _register("Int",    Int32)
if LP64:
    Long = _register("Long", Int64)
    if HasUInt64:
        _register("ULong",  UInt64)
        MaybeLong = _register("MaybeLong", Int64)
        __all__.append('MaybeLong')
else:
    Long = _register("Long", Int32)
    _register("ULong", UInt32)
    MaybeLong = _register("MaybeLong", Int32)
    __all__.append('MaybeLong')


_register("UByte",  UInt8)
_register("UShort", UInt16)
_register("UInt",   UInt32)
Float = _register("Float",  Float64)
Complex = _register("Complex",  Complex64)

# short forms

_register("b1", Bool)
_register("u1", UInt8)
_register("u2", UInt16)
_register("u4", UInt32)
_register("i1", Int8)
_register("i2", Int16)
_register("i4", Int32)

_register("i8", Int64)
if HasUInt64:
    _register("u8", UInt64)

_register("f4", Float32)
_register("f8", Float64)
_register("c8", Complex32)
_register("c16", Complex64)

# NumPy forms

_register("1", Int8)
_register("B", Bool)
_register("c", Int8)
_register("b", UInt8)
_register("s", Int16)
_register("w", UInt16)
_register("i", Int32)
_register("N", Int64)
_register("u", UInt32)
_register("U", UInt64)

if LP64:
    _register("l", Int64)
else:
    _register("l", Int32)

_register("d", Float64)
_register("f", Float32)
_register("D", Complex64)
_register("F", Complex32)

# scipy.base forms

def _scipy_alias(scipy_type, numarray_type):
    _register(scipy_type, eval(numarray_type))
    globals()[scipy_type] = globals()[numarray_type]

_scipy_alias("bool_", "Bool")
_scipy_alias("bool8", "Bool")
_scipy_alias("int8", "Int8")
_scipy_alias("uint8", "UInt8")
_scipy_alias("int16", "Int16")
_scipy_alias("uint16", "UInt16")
_scipy_alias("int32", "Int32")
_scipy_alias("uint32", "UInt32")
_scipy_alias("int64", "Int64")
_scipy_alias("uint64", "UInt64")

_scipy_alias("float64", "Float64")
_scipy_alias("float32", "Float32")
_scipy_alias("complex128", "Complex64")
_scipy_alias("complex64", "Complex32")

# The rest is used by numeric modules to determine conversions

# Ranking of types from lowest to highest (sorta)
if not HasUInt64:
    genericTypeRank = ['Bool','Int8','UInt8','Int16','UInt16',
                       'Int32', 'UInt32', 'Int64',
                       'Float32','Float64', 'Complex32', 'Complex64',  'Object']
else:
    genericTypeRank = ['Bool','Int8','UInt8','Int16','UInt16',
                       'Int32', 'UInt32', 'Int64', 'UInt64',
                       'Float32','Float64', 'Complex32', 'Complex64', 'Object']

pythonTypeRank = [ bool, int, long, float, complex ]

# The next line is not platform independent XXX Needs to be generalized
if not LP64:
    pythonTypeMap  = {
        int:("Int32","int"),
        long:("Int64","int"),
        float:("Float64","float"),
        complex:("Complex64","complex")}

    scalarTypeMap = {
        int:"Int32",
        long:"Int64",
        float:"Float64",
        complex:"Complex64"}
else:
    pythonTypeMap  = {
        int:("Int64","int"),
        long:("Int64","int"),
        float:("Float64","float"),
        complex:("Complex64","complex")}

    scalarTypeMap = {
        int:"Int64",
        long:"Int64",
        float:"Float64",
        complex:"Complex64"}

pythonTypeMap.update({bool:("Bool","bool") })
scalarTypeMap.update({bool:"Bool"})

# Generate coercion matrix

def _initGenericCoercions():
    global genericCoercions
    genericCoercions = {}

    # vector with ...
    for ntype1 in genericTypeRank:
        nt1 = typeDict[ntype1]
        rank1 = genericTypeRank.index(ntype1)
        ntypesize1, inttype1, signedtype1 = nt1.bytes, \
                    isinstance(nt1, IntegralType), isinstance(nt1, SignedIntegralType)
        for ntype2 in genericTypeRank:
            # vector
            nt2 = typeDict[ntype2]
            ntypesize2, inttype2, signedtype2 = nt2.bytes, \
                    isinstance(nt2, IntegralType), isinstance(nt2, SignedIntegralType)
            rank2 = genericTypeRank.index(ntype2)
            if (signedtype1 != signedtype2) and inttype1 and inttype2:
                # mixing of signed and unsigned ints is a special case
                # If unsigned same size or larger, final size needs to be bigger
                #   if possible
                if signedtype1:
                    if ntypesize2 >= ntypesize1:
                        size = min(2*ntypesize2, MAX_INT_SIZE)
                    else:
                        size = ntypesize1
                else:
                    if ntypesize1 >= ntypesize2:
                        size = min(2*ntypesize1, MAX_INT_SIZE)
                    else:
                        size = ntypesize2
                outtype = "Int"+str(8*size)
            else:
                if rank1 >= rank2:
                    outtype = ntype1
                else:
                    outtype = ntype2
            genericCoercions[(ntype1, ntype2)] = outtype

        for ntype2 in pythonTypeRank:
            # scalar
            mapto, kind = pythonTypeMap[ntype2]
            if ((inttype1 and kind=="int") or (not inttype1 and kind=="float")):
                # both are of the same "kind" thus vector type dominates
                outtype = ntype1
            else:
                rank2 = genericTypeRank.index(mapto)
                if rank1 >= rank2:
                    outtype = ntype1
                else:
                    outtype = mapto
            genericCoercions[(ntype1, ntype2)] = outtype
            genericCoercions[(ntype2, ntype1)] = outtype

    # scalar-scalar
    for ntype1 in pythonTypeRank:
        maptype1 = scalarTypeMap[ntype1]
        genericCoercions[(ntype1,)] = maptype1
        for ntype2 in pythonTypeRank:
            maptype2 = scalarTypeMap[ntype2]
            genericCoercions[(ntype1, ntype2)] = genericCoercions[(maptype1, maptype2)]

    # Special cases more easily dealt with outside of the loop
    genericCoercions[("Complex32", "Float64")] = "Complex64"
    genericCoercions[("Float64", "Complex32")] = "Complex64"
    genericCoercions[("Complex32", "Int64")] = "Complex64"
    genericCoercions[("Int64", "Complex32")] = "Complex64"
    genericCoercions[("Complex32", "UInt64")] = "Complex64"
    genericCoercions[("UInt64", "Complex32")] = "Complex64"

    genericCoercions[("Int64","Float32")] = "Float64"
    genericCoercions[("Float32", "Int64")] = "Float64"
    genericCoercions[("UInt64","Float32")] = "Float64"
    genericCoercions[("Float32", "UInt64")] = "Float64"

    genericCoercions[(float, "Bool")] = "Float64"
    genericCoercions[("Bool", float)] = "Float64"

    genericCoercions[(float,float,float)] = "Float64" # for scipy.special
    genericCoercions[(int,int,float)] = "Float64" # for scipy.special

_initGenericCoercions()

# If complex is subclassed, the following may not be necessary
genericPromotionExclusions = {
    'Bool': (),
    'Int8': (),
    'Int16': (),
    'Int32': ('Float32','Complex32'),
    'UInt8': (),
    'UInt16': (),
    'UInt32': ('Float32','Complex32'),
    'Int64' : ('Float32','Complex32'),
    'UInt64' : ('Float32','Complex32'),
    'Float32': (),
    'Float64': ('Complex32',),
    'Complex32':(),
    'Complex64':()
} # e.g., don't allow promotion from Float64 to Complex32 or Int64 to Float32

# Numeric typecodes
typecodes = {'Integer': '1silN',
             'UnsignedInteger': 'bBwuU',
             'Float': 'fd',
             'Character': 'c',
             'Complex': 'FD' }

if HasUInt64:
    _MaximumType = {
        Bool :  UInt64,

        Int8  : Int64,
        Int16 : Int64,
        Int32 : Int64,
        Int64 : Int64,

        UInt8  : UInt64,
        UInt16 : UInt64,
        UInt32 : UInt64,
        UInt8  : UInt64,

        Float32 : Float64,
        Float64 : Float64,

        Complex32 : Complex64,
        Complex64 : Complex64
        }
else:
    _MaximumType = {
        Bool :  Int64,

        Int8  : Int64,
        Int16 : Int64,
        Int32 : Int64,
        Int64 : Int64,

        UInt8  : Int64,
        UInt16 : Int64,
        UInt32 : Int64,
        UInt8  : Int64,

        Float32 : Float64,
        Float64 : Float64,

        Complex32 : Complex64,
        Complex64 : Complex64
        }

def MaximumType(t):
    """returns the type of highest precision of the same general kind as 't'"""
    return _MaximumType[t]


def getType(type):
    """Return the numeric type object for type

    type may be the name of a type object or the actual object
    """
    if isinstance(type, NumericType):
        return type
    try:
        return typeDict[type]
    except KeyError:
        raise TypeError("Not a numeric type")

scalarTypes = (bool,int,long,float,complex)

_scipy_dtypechar = {
    Int8 : 'b',
    UInt8 : 'B',
    Int16 : 'h',
    UInt16 : 'H',
    Int32 : 'i',
    UInt32 : 'I',
    Int64 : 'q',
    UInt64 : 'Q',
    Float32 : 'f',
    Float64 : 'd',
    Complex32 : 'F',  # Note the switchup here:
    Complex64 : 'D'   #   numarray.Complex32 == scipy.complex64, etc.
    }

_scipy_dtypechar_inverse = {}
for key,value in _scipy_dtypechar.items():
    _scipy_dtypechar_inverse[value] = key

_val = numpy.int_(0).itemsize
if _val == 8:
    _scipy_dtypechar_inverse['l'] = Int64
    _scipy_dtypechar_inverse['L'] = UInt64
elif _val == 4:
    _scipy_dtypechar_inverse['l'] = Int32
    _scipy_dtypechar_inverse['L'] = UInt32

del _val

if LP64:
    _scipy_dtypechar_inverse['p'] = Int64
    _scipy_dtypechar_inverse['P'] = UInt64
else:
    _scipy_dtypechar_inverse['p'] = Int32
    _scipy_dtypechar_inverse['P'] = UInt32

def typefrom(obj):
    return _scipy_dtypechar_inverse[obj.dtype.char]
