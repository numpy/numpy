from typing import Literal

type _BoolCodes = Literal["bool", "bool_", "?", "b1", "|b1", "=b1", "<b1", ">b1"]

type _Int8Codes = Literal["int8", "byte", "b", "i1", "|i1", "=i1", "<i1", ">i1"]
type _Int16Codes = Literal["int16", "short", "h", "i2", "|i2", "=i2", "<i2", ">i2"]
type _Int32Codes = Literal["int32", "i4", "|i4", "=i4", "<i4", ">i4"]
type _Int64Codes = Literal["int64", "i8", "|i8", "=i8", "<i8", ">i8"]

type _UInt8Codes = Literal["uint8", "ubyte", "B", "u1", "|u1", "=u1", "<u1", ">u1"]
type _UInt16Codes = Literal["uint16", "ushort", "H", "u2", "|u2", "=u2", "<u2", ">u2"]
type _UInt32Codes = Literal["uint32", "u4", "|u4", "=u4", "<u4", ">u4"]
type _UInt64Codes = Literal["uint64", "u8", "|u8", "=u8", "<u8", ">u8"]

type _IntCCodes = Literal["intc", "i", "|i", "=i", "<i", ">i"]
type _LongCodes = Literal["long", "l", "|l", "=l", "<l", ">l"]
type _LongLongCodes = Literal["longlong", "q", "|q", "=q", "<q", ">q"]
type _IntPCodes = Literal["intp", "int", "int_", "n", "|n", "=n", "<n", ">n"]

type _UIntCCodes = Literal["uintc", "I", "|I", "=I", "<I", ">I"]
type _ULongCodes = Literal["ulong", "L", "|L", "=L", "<L", ">L"]
type _ULongLongCodes = Literal["ulonglong", "Q", "|Q", "=Q", "<Q", ">Q"]
type _UIntPCodes = Literal["uintp", "uint", "N", "|N", "=N", "<N", ">N"]

type _Float16Codes = Literal["float16", "half", "e", "f2", "|f2", "=f2", "<f2", ">f2"]
type _Float32Codes = Literal["float32", "single", "f", "f4", "|f4", "=f4", "<f4", ">f4"]
type _Float64Codes = Literal[
    "float64", "float", "double", "d", "f8", "|f8", "=f8", "<f8", ">f8"
]

type _LongDoubleCodes = Literal["longdouble", "g", "|g", "=g", "<g", ">g"]

type _Complex64Codes = Literal[
    "complex64", "csingle", "F", "c8", "|c8", "=c8", "<c8", ">c8"
]

type _Complex128Codes = Literal[
    "complex128", "complex", "cdouble", "D", "c16", "|c16", "=c16", "<c16", ">c16"
]

type _CLongDoubleCodes = Literal["clongdouble", "G", "|G", "=G", "<G", ">G"]

type _StrCodes = Literal["str", "str_", "unicode", "U", "|U", "=U", "<U", ">U"]
type _BytesCodes = Literal["bytes", "bytes_", "S", "|S", "=S", "<S", ">S"]
type _VoidCodes = Literal["void", "V", "|V", "=V", "<V", ">V"]
type _ObjectCodes = Literal["object", "object_", "O", "|O", "=O", "<O", ">O"]

# datetime64
type _DT64Codes_any = Literal["datetime64", "M", "M8", "|M8", "=M8", "<M8", ">M8"]
type _DT64Codes_date = Literal[
    "datetime64[Y]", "M8[Y]", "|M8[Y]", "=M8[Y]", "<M8[Y]", ">M8[Y]",
    "datetime64[M]", "M8[M]", "|M8[M]", "=M8[M]", "<M8[M]", ">M8[M]",
    "datetime64[W]", "M8[W]", "|M8[W]", "=M8[W]", "<M8[W]", ">M8[W]",
    "datetime64[D]", "M8[D]", "|M8[D]", "=M8[D]", "<M8[D]", ">M8[D]",
]  # fmt: skip
type _DT64Codes_datetime = Literal[
    "datetime64[h]", "M8[h]", "|M8[h]", "=M8[h]", "<M8[h]", ">M8[h]",
    "datetime64[m]", "M8[m]", "|M8[m]", "=M8[m]", "<M8[m]", ">M8[m]",
    "datetime64[s]", "M8[s]", "|M8[s]", "=M8[s]", "<M8[s]", ">M8[s]",
    "datetime64[ms]", "M8[ms]", "|M8[ms]", "=M8[ms]", "<M8[ms]", ">M8[ms]",
    "datetime64[us]", "M8[us]", "|M8[us]", "=M8[us]", "<M8[us]", ">M8[us]",
    "datetime64[μs]", "M8[μs]", "|M8[μs]", "=M8[μs]", "<M8[μs]", ">M8[μs]",
]  # fmt: skip
type _DT64Codes_int = Literal[
    "datetime64[ns]", "M8[ns]", "|M8[ns]", "=M8[ns]", "<M8[ns]", ">M8[ns]",
    "datetime64[ps]", "M8[ps]", "|M8[ps]", "=M8[ps]", "<M8[ps]", ">M8[ps]",
    "datetime64[fs]", "M8[fs]", "|M8[fs]", "=M8[fs]", "<M8[fs]", ">M8[fs]",
    "datetime64[as]", "M8[as]", "|M8[as]", "=M8[as]", "<M8[as]", ">M8[as]",
]  # fmt: skip
type _DT64Codes = Literal[
    _DT64Codes_any,
    _DT64Codes_date,
    _DT64Codes_datetime,
    _DT64Codes_int,
]

# timedelta64
type _TD64Codes_any = Literal["timedelta64", "m", "m8", "|m8", "=m8", "<m8", ">m8"]
type _TD64Codes_int = Literal[
    "timedelta64[Y]", "m8[Y]", "|m8[Y]", "=m8[Y]", "<m8[Y]", ">m8[Y]",
    "timedelta64[M]", "m8[M]", "|m8[M]", "=m8[M]", "<m8[M]", ">m8[M]",
    "timedelta64[ns]", "m8[ns]", "|m8[ns]", "=m8[ns]", "<m8[ns]", ">m8[ns]",
    "timedelta64[ps]", "m8[ps]", "|m8[ps]", "=m8[ps]", "<m8[ps]", ">m8[ps]",
    "timedelta64[fs]", "m8[fs]", "|m8[fs]", "=m8[fs]", "<m8[fs]", ">m8[fs]",
    "timedelta64[as]", "m8[as]", "|m8[as]", "=m8[as]", "<m8[as]", ">m8[as]",
]  # fmt: skip
type _TD64Codes_timedelta = Literal[
    "timedelta64[W]", "m8[W]", "|m8[W]", "=m8[W]", "<m8[W]", ">m8[W]",
    "timedelta64[D]", "m8[D]", "|m8[D]", "=m8[D]", "<m8[D]", ">m8[D]",
    "timedelta64[h]", "m8[h]", "|m8[h]", "=m8[h]", "<m8[h]", ">m8[h]",
    "timedelta64[m]", "m8[m]", "|m8[m]", "=m8[m]", "<m8[m]", ">m8[m]",
    "timedelta64[s]", "m8[s]", "|m8[s]", "=m8[s]", "<m8[s]", ">m8[s]",
    "timedelta64[ms]", "m8[ms]", "|m8[ms]", "=m8[ms]", "<m8[ms]", ">m8[ms]",
    "timedelta64[us]", "m8[us]", "|m8[us]", "=m8[us]", "<m8[us]", ">m8[us]",
    "timedelta64[μs]", "m8[μs]", "|m8[μs]", "=m8[μs]", "<m8[μs]", ">m8[μs]",
]  # fmt: skip
type _TD64Codes = Literal[_TD64Codes_any, _TD64Codes_int, _TD64Codes_timedelta]

# NOTE: `StringDType' has no scalar type, and therefore has no name that can
# be passed to the `dtype` constructor
type _StringCodes = Literal["T", "|T", "=T", "<T", ">T"]

# NOTE: Nested literals get flattened and de-duplicated at runtime, which isn't
# the case for a `Union` of `Literal`s.
# So even though they're equivalent when type-checking, they differ at runtime.
# Another advantage of nesting, is that they always have a "flat"
# `Literal.__args__`, which is a tuple of *literally* all its literal values.

type _SignedIntegerCodes = Literal[
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntCCodes,
    _LongCodes,
    _LongLongCodes,
    _IntPCodes,
]
type _UnsignedIntegerCodes = Literal[
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UIntCCodes,
    _ULongCodes,
    _ULongLongCodes,
    _UIntPCodes,
]
type _FloatingCodes = Literal[
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _LongDoubleCodes,
]
type _ComplexFloatingCodes = Literal[
    _Complex64Codes,
    _Complex128Codes,
    _CLongDoubleCodes,
]
type _IntegerCodes = Literal[_UnsignedIntegerCodes, _SignedIntegerCodes]
type _InexactCodes = Literal[_FloatingCodes, _ComplexFloatingCodes]
type _NumberCodes = Literal[_IntegerCodes, _InexactCodes]

type _CharacterCodes = Literal[_BytesCodes, _StrCodes]
type _FlexibleCodes = Literal[_CharacterCodes, _VoidCodes]

type _GenericCodes = Literal[
    _BoolCodes,
    _NumberCodes,
    _FlexibleCodes,
    _DT64Codes,
    _TD64Codes,
    _ObjectCodes,
    # TODO: add `_StringCodes` once it has a scalar type
    # _StringCodes,
]
