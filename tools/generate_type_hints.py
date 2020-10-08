#!/usr/bin/env python
"""A script for generating stub files with platform-specific types."""

import os
import sys
import argparse
import datetime as dt
from collections import defaultdict
from typing import Union, IO, Dict, ContextManager, Any

# Import directly from `numerictypes` in order to get this script
# to work with the main `setup.py`
import numpy.core.numerictypes as nt

if sys.version_info >= (3, 7):
    from contextlib import nullcontext
else:
    from numpy.compat import contextlib_nullcontext as nullcontext

__all__ = ["generate_alias"]

_AnyPath = Union[str, bytes, os.PathLike]

#: A set with all currently annotated `number` subclasses.
ANNOTATED = {
    nt.int8,
    nt.int16,
    nt.int32,
    nt.int64,
    nt.uint8,
    nt.uint16,
    nt.uint32,
    nt.uint64,
    nt.float16,
    nt.float32,
    nt.float64,
    nt.complex64,
    nt.complex128,
}

#: A mapping with `number` aliases as keys and the name of their
#: character-code-containing `Literal` as values.
TO_BE_ANNOTATED = {
    "byte": "_ByteCodes",
    "short": "_ShortCodes",
    "intc": "_IntCCodes",
    "intp": "_IntPCodes",
    "int_": "_IntCodes",
    "longlong": "_LongLongCodes",
    "ubyte": "_UByteCodes",
    "ushort": "_UShortCodes",
    "uintc": "_UIntCCodes",
    "uintp": "_UIntPCodes",
    "uint": "_UIntCodes",
    "ulonglong": "_ULongLongCodes",
    "half": "_HalfCodes",
    "single": "_SingleCodes",
    "float_": "_DoubleCodes",
    "longfloat": "_LongFloatCodes",
    "csingle": "_CSingleCodes",
    "complex_": "_CDoubleCodes",
    "clongfloat": "_CLongFloatCodes",
}

TEMPLATE = r"""{docstring}

from typing import Union, Any

from numpy import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
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
    _LongFloatCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongFloatCodes,
)

_Int8Codes = Union[_Int8CodesBase{int8}]
_Int16Codes = Union[_Int16CodesBase{int16}]
_Int32Codes = Union[_Int32CodesBase{int32}]
_Int64Codes = Union[_Int64CodesBase{int64}]

_UInt8Codes = Union[_UInt8CodesBase{uint8}]
_UInt16Codes = Union[_UInt16CodesBase{uint16}]
_UInt32Codes = Union[_UInt32CodesBase{uint32}]
_UInt64Codes = Union[_UInt64CodesBase{uint64}]

_Float16Codes = Union[_Float16CodesBase{float16}]
_Float32Codes = Union[_Float32CodesBase{float32}]
_Float64Codes = Union[_Float64CodesBase{float64}]

_Complex64Codes = Union[_Complex64CodesBase{complex64}]
_Complex128Codes = Union[_Complex128CodesBase{complex128}]

# Note that these variables are private despite the lack of underscore;
# don't expose them to the `numpy.typing` __init__ file.
byte = {byte}
short = {short}
intc = {intc}
intp = int0 = {intp}
int_ = {int_}
longlong = {longlong}

ubyte = {ubyte}
ushort = {ushort}
uintc = {uintc}
uintp = uint0 = {uintp}
uint = {uint}
ulonglong = {ulonglong}

half = {half}
single = {single}
float_ = double = {float_}
longfloat = longdouble = {longfloat}

csingle = singlecomplex = {csingle}
complex_ = cdouble = cfloat = {complex_}
clongfloat = longcomplex = {clongfloat}

"""


def _file_to_context(
    file: Union[IO[str], _AnyPath], *args: Any, **kwargs: Any
) -> ContextManager[IO[str]]:
    """Create a context manager from a path- or file-like object."""
    try:
        return open(file, *args, **kwargs)
    except TypeError:
        return nullcontext(file)


def generate_alias(file: Union[IO[str], _AnyPath] = sys.stdout) -> None:
    """Generate a stub file with platform-specific types.

    Parameters
    ----------
    file : file- or path-like
        A file- or path-like object used for writing the results.

    """
    # Generate type aliases and update the character codes for all known types
    type_alias = {}
    char_codes: Dict[str, str] = defaultdict(str)

    for name, code_names in TO_BE_ANNOTATED.items():
        typ = getattr(nt, name)
        if typ in ANNOTATED:
            type_alias[name] = typ.__name__
            char_codes[typ.__name__] += f", {code_names}"
        else:
            # Plan B in case an untyped class is encountered
            type_alias[name] = "Any"

    # Generate the docstring
    now = dt.datetime.now()
    docstring = f'"""THIS FILE WAS AUTOMATICALLY GENERATED ON {now}."""'

    # Create the new stub file
    with _file_to_context(file, "w", encoding="utf8") as f:
        string = TEMPLATE.format(docstring=docstring, **type_alias, **char_codes)
        f.write(string)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        nargs="?",
        default=sys.stdout,
        help="The file used for writing the output",
    )

    args = parser.parse_args()
    generate_alias(args.file)


if __name__ == "__main__":
    main()
