# Lifted from Precision.py.  This is for compatibility only.
#
#  The character strings are still for "new" NumPy
#   which is the only Incompatibility with Numeric

__all__ = ['Character', 'Complex', 'Float',
           'PrecisionError', 'PyObject', 'Int', 'UInt',
           'UnsignedInt', 'UnsignedInteger', 'string', 'typecodes', 'zeros']

from functions import zeros
import string   # for backwards compatibility

typecodes = {'Character':'c', 'Integer':'bhil', 'UnsignedInteger':'BHIL', 'Float':'fd', 'Complex':'FD'}

def _get_precisions(typecodes):
    lst = []
    for t in typecodes:
        lst.append( (zeros( (1,), t ).itemsize*8, t) )
    return lst

def _fill_table(typecodes, table={}):
    for key, value in typecodes.items():
        table[key] = _get_precisions(value)
    return table

_code_table = _fill_table(typecodes)

class PrecisionError(Exception):
    pass

def _lookup(table, key, required_bits):
    lst = table[key]
    for bits, typecode in lst:
        if bits >= required_bits:
            return typecode
    raise PrecisionError(key + " of " + str(required_bits) +
            " bits not available on this system")

Character = 'c'

try:
    UnsignedInt8 = _lookup(_code_table, "UnsignedInteger", 8)
    UInt8 = UnsignedInt8
    __all__.extend(['UnsignedInt8', 'UInt8'])
except(PrecisionError):
    pass
try:
    UnsignedInt16 = _lookup(_code_table, "UnsignedInteger", 16)
    UInt16 = UnsignedInt16
    __all__.extend(['UnsignedInt16', 'UInt16'])
except(PrecisionError):
    pass
try:
    UnsignedInt32 = _lookup(_code_table, "UnsignedInteger", 32)
    UInt32 = UnsignedInt32
    __all__.extend(['UnsignedInt32', 'UInt32'])
except(PrecisionError):
    pass
try:
    UnsignedInt64 = _lookup(_code_table, "UnsignedInteger", 64)
    UInt64 = UnsignedInt64
    __all__.extend(['UnsignedInt64', 'UInt64'])
except(PrecisionError):
    pass
try:
    UnsignedInt128 = _lookup(_code_table, "UnsignedInteger", 128)
    UInt128 = UnsignedInt128
    __all__.extend(['UnsignedInt128', 'UInt128'])
except(PrecisionError):
    pass
UInt = UnsignedInt = UnsignedInteger = 'u'

try:
    Int0 = _lookup(_code_table, 'Integer', 0)
    __all__.append('Int0')
except(PrecisionError):
    pass
try:
    Int8 = _lookup(_code_table, 'Integer', 8)
    __all__.append('Int8')
except(PrecisionError):
    pass
try:
    Int16 = _lookup(_code_table, 'Integer', 16)
    __all__.append('Int16')
except(PrecisionError):
    pass
try:
    Int32 = _lookup(_code_table, 'Integer', 32)
    __all__.append('Int32')
except(PrecisionError):
    pass
try:
    Int64 = _lookup(_code_table, 'Integer', 64)
    __all__.append('Int64')
except(PrecisionError):
    pass
try:
    Int128 = _lookup(_code_table, 'Integer', 128)
    __all__.append('Int128')
except(PrecisionError):
    pass
Int = 'l'

try:
    Float0 = _lookup(_code_table, 'Float', 0)
    __all__.append('Float0')
except(PrecisionError):
    pass
try:
    Float8 = _lookup(_code_table, 'Float', 8)
    __all__.append('Float8')
except(PrecisionError):
    pass
try:
    Float16 = _lookup(_code_table, 'Float', 16)
    __all__.append('Float16')
except(PrecisionError):
    pass
try:
    Float32 = _lookup(_code_table, 'Float', 32)
    __all__.append('Float32')
except(PrecisionError):
    pass
try:
    Float64 = _lookup(_code_table, 'Float', 64)
    __all__.append('Float64')
except(PrecisionError):
    pass
try:
    Float128 = _lookup(_code_table, 'Float', 128)
    __all__.append('Float128')
except(PrecisionError):
    pass
Float = 'd'

try:
    Complex0 = _lookup(_code_table, 'Complex', 0)
    __all__.append('Complex0')
except(PrecisionError):
    pass
try:
    Complex8 = _lookup(_code_table, 'Complex', 16)
    __all__.append('Complex8')
except(PrecisionError):
    pass
try:
    Complex16 = _lookup(_code_table, 'Complex', 32)
    __all__.append('Complex16')
except(PrecisionError):
    pass
try:
    Complex32 = _lookup(_code_table, 'Complex', 64)
    __all__.append('Complex32')
except(PrecisionError):
    pass
try:
    Complex64 = _lookup(_code_table, 'Complex', 128)
    __all__.append('Complex64')
except(PrecisionError):
    pass
try:
    Complex128 = _lookup(_code_table, 'Complex', 256)
    __all__.append('Complex128')
except(PrecisionError):
    pass
Complex = 'D'

PyObject = 'O'
