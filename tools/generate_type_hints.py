#!/usr/bin/env python
"""A script for generating stub files with platform-specific types."""

import os
import sys
import argparse
import ctypes as ct
from collections import defaultdict
from typing import Union, Dict, IO

__all__ = ["generate_alias"]

_AnyPath = Union[str, bytes, os.PathLike]

ANNOTATED_CHARCODES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
}

ALLOWED_SIZES = {8, 16, 32, 64, 80, 96, 128, 256}

#: A mapping with `number` aliases as keys and the name of their
#: character-code-containing `Literal` as values.
TO_BE_ANNOTATED = {
    # `ctypes.c_void_p` is not quite a direct `np.intp` counterpart,
    # but its size is the same
    "byte": (ct.c_byte, "_ByteCodes", "int"),
    "short": (ct.c_short, "_ShortCodes", "int"),
    "intc": (ct.c_int, "_IntCCodes", "int"),
    "intp": (ct.c_void_p, "_IntPCodes", "int"),
    "int_": (ct.c_long, "_IntCodes", "int"),
    "longlong": (ct.c_longlong, "_LongLongCodes", "int"),

    "ubyte": (ct.c_ubyte, "_UByteCodes", "uint"),
    "ushort": (ct.c_ushort, "_UShortCodes", "uint"),
    "uintc": (ct.c_uint, "_UIntCCodes", "uint"),
    "uintp": (ct.c_void_p, "_UIntPCodes", "uint"),
    "uint": (ct.c_ulong, "_UIntCodes", "uint"),
    "ulonglong": (ct.c_ulonglong, "_ULongLongCodes", "uint"),

    "double": (ct.c_double, "_DoubleCodes", "float"),
    "longdouble": (ct.c_longdouble, "_LongDoubleCodes", "float"),

    # `ctypes` lacks `complex`-based types, so instead grab their float-based
    # counterpart and multiply its size by 2
    "cdouble": (ct.c_double, "_CDoubleCodes", "complex"),
    "clongdouble": (ct.c_longdouble, "_CLongDoubleCodes", "complex"),
}

TEMPLATE = r'''"""THIS FILE WAS AUTOMATICALLY GENERATED."""

from typing import Union, Any

from numpy import (
    signedinteger,
    unsignedinteger,
    floating,
    complexfloating,
)

from . import (
    _256Bit,
    _128Bit,
    _96Bit,
    _80Bit,
    _64Bit,
    _32Bit,
    _16Bit,
    _8Bit,
)

from ._char_codes import (
    _Int8CodesBase,
    _Int16CodesBase,
    _Int32CodesBase,
    _Int64CodesBase,
    _UInt8CodesBase,
    _UInt16CodesBase,
    _UInt32CodesBase,
    _UInt64CodesBase,
    _Float16CodesBase,
    _Float32CodesBase,
    _Float64CodesBase,
    _Complex64CodesBase,
    _Complex128CodesBase,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
)

_Int8Codes = Union[_Int8CodesBase{int8}]
_Int16Codes = Union[_Int16CodesBase{int16}]
_Int32Codes = Union[_Int32CodesBase{int32}]
_Int64Codes = Union[_Int64CodesBase{int64}]

_UInt8Codes = Union[_UInt8CodesBase{uint8}]
_UInt16Codes = Union[_UInt16CodesBase{uint16}]
_UInt32Codes = Union[_UInt32CodesBase{uint32}]
_UInt64Codes = Union[_UInt64CodesBase{uint64}]

_Float16Codes = Union[_Float16CodesBase, _HalfCodes]
_Float32Codes = Union[_Float32CodesBase, _SingleCodes]
_Float64Codes = Union[_Float64CodesBase{float64}]

_Complex64Codes = Union[_Complex64CodesBase{complex64}]
_Complex128Codes = Union[_Complex128CodesBase{complex128}]

# Note that these variables are private despite the lack of underscore;
# don't expose them to the `numpy.typing` __init__ file.
#
# NOTE: mypy has issues parsing double assignments, don't use them
# (e.g. `cdouble = csingle = cfloat`)
byte = signedinteger[{byte}]
short = signedinteger[{short}]
intc = signedinteger[{intc}]
intp = signedinteger[{intp}]
int0 = intp
int_ = signedinteger[{int_}]
longlong = signedinteger[{longlong}]

ubyte = unsignedinteger[{ubyte}]
ushort = unsignedinteger[{ushort}]
uintc = unsignedinteger[{uintc}]
uintp = unsignedinteger[{uintp}]
uint0 = uintp
uint = unsignedinteger[{uint}]
ulonglong = unsignedinteger[{ulonglong}]

half = floating[_16Bit]
single = floating[_32Bit]
double = floating[{double}]
float_ = double
longdouble = floating[{longdouble}]
longfloat = longdouble

csingle = complexfloating[_32Bit, _32Bit]
singlecomplex = csingle
cdouble = complexfloating[{cdouble}, {cdouble}]
complex_ = cdouble
cfloat = cdouble
clongdouble = complexfloating[{clongdouble}, {clongdouble}]
longcomplex = clongdouble
clongfloat = clongdouble

# The precision of `np.int_`
_NBitInt = {int_}
'''


def generate_alias(file: IO[str]) -> None:
    """Generate a stub file with platform-specific types.

    Parameters
    ----------
    file : file-like
        A writeable file-like object opened in text mode.

    """
    # Generate type aliases and update the character codes for all known types
    type_alias = {}
    char_codes: Dict[str, str] = defaultdict(str)

    for name, (ctype, code_names, prefix) in TO_BE_ANNOTATED.items():
        size = 8 * ct.sizeof(ctype)
        if size not in ALLOWED_SIZES:
            type_alias[name] = "Any"
            continue
        type_alias[name] = f"_{size}Bit"

        numbered_name = f'{prefix}{size}'
        if numbered_name in ANNOTATED_CHARCODES:
            char_codes[numbered_name] += f", {code_names}"

    string = TEMPLATE.format(**type_alias, **char_codes)
    file.write(string)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        nargs="?",
        default=None,
        help="The file used for writing the output",
    )

    args = parser.parse_args()
    file = args.file

    if file is None:
        generate_alias(sys.stdout)
    else:
        with open(file, "w", encoding="utf8") as f:
            generate_alias(f)


if __name__ == "__main__":
    main()
