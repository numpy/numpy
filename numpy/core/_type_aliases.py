"""
This file builds `allTypes`, `sctypeDict` and `sctypes`.
Pre-refactor version can be found in `_expired_type_aliases.py` file.

Types of types & aliases:

1. `words` (e.g. "float", "cdouble", "int_")
2. `words+bits` (e.g. "int16", "complex64")
3. `symbols` (e.g. "p", "L")
4. `symbols+bytes` (e.g. "c8", "b1")
5. `abstracts` (e.g. "inexact", "integer")
6. `numbers` (e.g. 1, 2, 3)
7. `aliases` (e.g. "complex_", "longfloat")

`allTypes` - contains `words`, `words+bits`, `abstracts`

`sctypeDict` - contains `words`, `words+bits`, `numbers`, 
                        `symbols`, `symbols+bytes`, `aliases`

`sctypes` - A map from generic types, "int", "float", 
            to available sizes (e.g. "int32")

"""
from typing import Dict, List

from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype, typeinforanged
from numpy.core._dtype import _kind_name


# separate the actual type info from the abstract base classes
_abstract_types = {}
_concrete_typeinfo = {}
for k, v in typeinfo.items():
    # make all the keys lowercase
    k = english_lower(k)
    if isinstance(v, type):
        _abstract_types[k] = v
    else:
        _concrete_typeinfo[k] = v
_concrete_types = {v.type for k, v in _concrete_typeinfo.items()}


# 1. `words`
word_dict = {}
# 2. `words+bits`
word_bits_dict = {}
# 3. `symbols`
symbol_dict = {}
# 4. `symbols+bytes`
symbol_bytes_dict = {}
# 5. `abstracts`
abstract_types_dict = _abstract_types
# 6. `numbers`
numbers_dict = {}
# 7. `aliases`
extra_aliases_dict = {}


# Build dictionaries following C naming
def _build_dicts() -> None:
    # For int and uint types each bits size is inserted only once. 
    # If two types have the same size we select one with a lower priority.
    _int_ctypes_prio = {
        "long": 0, "longlong": 1, "int": 2, "short": 3, "byte": 4
    }
    _uint_ctypes_prio = {
        "ulong": 0, "ulonglong": 1, "uint": 2, "ushort": 3, "ubyte": 4
    }

    _floating_ctypes = ["half", "float", "double", "longdouble"]
    _complex_ctypes = ["cfloat", "cdouble", "clongdouble"]
    _other_types = [
        "object", "string", "unicode", "void", "datetime", "timedelta", "bool"
    ]

    STATUS: type = bool
    PROCESSED: bool = True
    REJECTED: bool = False

    # this function populates word+bits and symbol+bytes for integer types
    def _process_integer_type(
        name: str, 
        info: typeinforanged, 
        priority_dict: Dict[str, int], 
        seen_bits_dict: Dict[int, str],
        long_name: str, 
        short_name: str
    ) -> STATUS:
        bits: int = info.bits 
        if name in priority_dict and (
            bits not in seen_bits_dict or 
            priority_dict[seen_bits_dict[bits]] > priority_dict[name]
        ):
            word_bits_dict[f"{long_name}{bits}"] = info.type
            symbol_bytes_dict[f"{short_name}{bits//8}"] = info.type
            seen_bits_dict[bits] = name
            return PROCESSED
        else:
            return REJECTED

    # We will track which sizes we have already seen for ints and uints
    _seen_int_bits = {}
    _seen_uint_bits = {}

    # MAIN LOOP: Traversing all contrete types that come from 
    # the compiled multiarray module
    for name, info in _concrete_typeinfo.items():
        
        bits: int = info.bits
        symbol: str = info.char
        
        # Adding type to non-bit dictionaries
        word_dict[name] = info.type
        symbol_dict[symbol] = info.type
        numbers_dict[info.num] = info.type

        # Adding type to bit-aware dictionaries
        # 1. Checking if the type is a signed integer
        result = _process_integer_type(
            name, info, _int_ctypes_prio, _seen_int_bits, "int", "i"
        )
        if result is PROCESSED:
            continue
        # 2. If not, then we check if it's an unsigned integer
        result = _process_integer_type(
            name, info, _uint_ctypes_prio, _seen_uint_bits, "uint", "u"
        )
        if result is PROCESSED:
            continue
        if name in _complex_ctypes + _floating_ctypes + _other_types:
            # 3. Otherwise it can be float/complex or other type
            dt = dtype(info.type)
            word_bits_name = f"{_kind_name(dt)}{bits}"
            symbol_bytes_name = f"{dt.kind}{bits//8}"

            # ensure that (c)longdouble does not overwrite the aliases 
            # assigned to (c)double, when both have same number of bits
            if (
                name in ('longdouble', 'clongdouble') and
                word_bits_name in word_bits_dict
            ):
                continue

            word_bits_dict[word_bits_name] = info.type
            symbol_bytes_dict[symbol_bytes_name] = info.type


_build_dicts()


# Rename types to Python conventions and introduce aliases
def _renaming_and_aliases() -> None:
    renaming_dict = [
        # In Python `float` is `double`
        ("single", "float"),
        ("float", "double"),

        # In Python `cfloat` is `cdouble`
        ("csingle", "cfloat"),
        ("cfloat", "cdouble"),

        ("intc", "int"),
        ("uintc", "uint"),

        # In Python `int` is `long`
        ("int", "long"),
        ("int_", "long"),
        ("uint", "ulong"),

        ("bool_", "bool"),
        ("bytes_", "string"),
        ("str_", "unicode"),
        ("object_", "object"),
        ("object0", "object"),

        # aliases to be removed
        ("float_", "double"),
        ("complex_", "cdouble"),
        ("longfloat", "longdouble"),
        ("clongfloat", "clongdouble"),
        ("longcomplex", "clongdouble"),
        ("singlecomplex", "csingle"),
        ("string_", "string"),
        ("unicode_", "unicode")
    ]

    for alias, t in renaming_dict:
        word_dict[alias] = word_dict[t]

    extra_aliases = [
        ("complex", "cdouble"),
        ("float", "double"),
        ("str", "unicode"),
        ("bytes", "string"),
        ("a", "string"),
        ("int0", "intp"),
        ("uint0", "uintp")
    ]

    for k, v in extra_aliases:
        extra_aliases_dict[k] = word_dict[v]


_renaming_and_aliases()


# Let's build final dictionaries according to our recipes!
# First we build `allTypes` and `sctypeDict`
allTypes = word_dict | word_bits_dict | abstract_types_dict
# delete C names in `allTypes` and exceptions
for s in ["ulong", "long", "unicode", "object", "bool", "datetime",
          "string", "timedelta", "float", "int", "bool8", "bytes0", 
          "object32", "object64", "str0", "void0", "object0"]:
    try:
        del allTypes[s]
    except KeyError:
        pass

sctypeDict = numbers_dict | symbol_dict | symbol_bytes_dict | \
    word_dict | word_bits_dict | extra_aliases_dict
# delete exceptions
for s in ["O8", "O4", "S0", "U0", "V0", "datetime", "object32",
          "object64", "string", "timedelta"]:
    try:
        del sctypeDict[s]
    except KeyError:
        pass


# Finally, we build `sctypes`` mapping
def _build_sctypes() -> Dict[str, List[type]]:
    def _add_array_type(typename, bits, sctypes):
        try:
            t = sctypeDict[f"{typename}{bits}"]
        except KeyError:
            pass
        else:
            sctypes[typename].append(t)

    sctypes = {
        'int': [],
        'uint': [],
        'float': [],
        'complex': [],
        'others': [bool, object, bytes, str, allTypes['void']]
    }
    
    ibytes = [1, 2, 4, 8, 16, 32, 64]
    fbytes = [2, 4, 8, 10, 12, 16, 32, 64]
    for b in ibytes:
        bits = 8*b
        _add_array_type('int', bits, sctypes)
        _add_array_type('uint', bits, sctypes)
    for b in fbytes:
        bits = 8*b
        _add_array_type('float', bits, sctypes)
        _add_array_type('complex', 2*bits, sctypes)
    
    # include c pointer sized integer
    _gi = dtype('p')
    if _gi.type not in sctypes['int']:
        indx = 0
        sz = _gi.itemsize
        _lst = sctypes['int']
        while (indx < len(_lst) and sz >= _lst[indx](0).itemsize):
            indx += 1
        sctypes['int'].insert(indx, _gi.type)
        sctypes['uint'].insert(indx, dtype('P').type)
    return sctypes


sctypes = _build_sctypes()
