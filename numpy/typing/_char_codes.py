import sys
from typing import Any, TYPE_CHECKING

if sys.version_info >= (3, 8):
    from typing import Literal
    HAVE_LITERAL = True
else:
    try:
        from typing_extensions import Literal
    except ImportError:
        HAVE_LITERAL = False
    else:
        HAVE_LITERAL = True

if TYPE_CHECKING or HAVE_LITERAL:
    _BoolCodes = Literal["?", "=?", "<?", ">?", "bool", "bool_", "bool8"]

    _UInt8CodesBase = Literal["uint8", "u1", "=u1", "<u1", ">u1"]
    _UInt16CodesBase = Literal["uint16", "u2", "=u2", "<u2", ">u2"]
    _UInt32CodesBase = Literal["uint32", "u4", "=u4", "<u4", ">u4"]
    _UInt64CodesBase = Literal["uint64", "u8", "=u8", "<u8", ">u8"]

    _Int8CodesBase = Literal["int8", "i1", "=i1", "<i1", ">i1"]
    _Int16CodesBase = Literal["int16", "i2", "=i2", "<i2", ">i2"]
    _Int32CodesBase = Literal["int32", "i4", "=i4", "<i4", ">i4"]
    _Int64CodesBase = Literal["int64", "i8", "=i8", "<i8", ">i8"]

    _Float16CodesBase = Literal["float16", "f2", "=f2", "<f2", ">f2"]
    _Float32CodesBase = Literal["float32", "f4", "=f4", "<f4", ">f4"]
    _Float64CodesBase = Literal["float64", "f8", "=f8", "<f8", ">f8"]

    _Complex64CodesBase = Literal["complex64", "c8", "=c8", "<c8", ">c8"]
    _Complex128CodesBase = Literal["complex128", "c16", "=c16", "<c16", ">c16"]

    _ByteCodes = Literal["byte", "b", "=b", "<b", ">b"]
    _ShortCodes = Literal["short", "h", "=h", "<h", ">h"]
    _IntCCodes = Literal["intc", "i", "=i", "<i", ">i"]
    _IntPCodes = Literal["intp", "int0", "p", "=p", "<p", ">p"]
    _IntCodes = Literal["long", "int", "int_", "l", "=l", "<l", ">l"]
    _LongLongCodes = Literal["longlong", "q", "=q", "<q", ">q"]

    _UByteCodes = Literal["ubyte", "B", "=B", "<B", ">B"]
    _UShortCodes = Literal["ushort", "H", "=H", "<H", ">H"]
    _UIntCCodes = Literal["uintc", "I", "=I", "<I", ">I"]
    _UIntPCodes = Literal["uintp", "uint0", "P", "=P", "<P", ">P"]
    _UIntCodes = Literal["uint", "L", "=L", "<L", ">L"]
    _ULongLongCodes = Literal["ulonglong", "Q", "=Q", "<Q", ">Q"]

    _HalfCodes = Literal["half", "e", "=e", "<e", ">e"]
    _SingleCodes = Literal["single", "f", "=f", "<f", ">f"]
    _DoubleCodes = Literal["double" "float", "float_", "d", "=d", "<d", ">d"]
    _LongFloatCodes = Literal["longfloat", "longdouble", "g", "=g", "<g", ">g"]

    _CSingleCodes = Literal["csingle", "singlecomplex", "F", "=F", "<F", ">F"]
    _CDoubleCodes = Literal["cdouble" "complex", "complex_", "cfloat", "D", "=D", "<D", ">D"]
    _CLongFloatCodes = Literal["clongfloat", "longcomplex", "clongdouble", "G", "=G", "<G", ">G"]

    _Datetime64Codes = Literal["datetime64", "M", "=M", "<M", ">M"]
    _Timedelta64Codes = Literal["timedelta64", "m", "=m", "<m", ">m"]

    _StrCodes = Literal["str", "str_", "str0", "unicode", "unicode_", "U", "=U", "<U", ">U"]
    _BytesCodes = Literal["bytes", "bytes_", "bytes0", "S", "=S", "<S", ">S"]
    _VoidCodes = Literal["void", "void0", "V", "=V", "<V", ">V"]
    _ObjectCodes = Literal["object", "object_", "O", "=O", "<O", ">O"]

else:
    _BoolCodes = Any

    _UInt8CodesBase = Any
    _UInt16CodesBase = Any
    _UInt32CodesBase = Any
    _UInt64CodesBase = Any

    _Int8CodesBase = Any
    _Int16CodesBase = Any
    _Int32CodesBase = Any
    _Int64CodesBase = Any

    _Float16CodesBase = Any
    _Float32CodesBase = Any
    _Float64CodesBase = Any

    _Complex64CodesBase = Any
    _Complex128CodesBase = Any

    _ByteCodes = Any
    _ShortCodes = Any
    _IntCCodes = Any
    _IntPCodes = Any
    _IntCodes = Any
    _LongLongCodes = Any

    _UByteCodes = Any
    _UShortCodes = Any
    _UIntCCodes = Any
    _UIntPCodes = Any
    _UIntCodes = Any
    _ULongLongCodes = Any

    _HalfCodes = Any
    _SingleCodes = Any
    _DoubleCodes = Any
    _LongFloatCodes = Any

    _CSingleCodes = Any
    _CDoubleCodes = Any
    _CLongFloatCodes = Any

    _Datetime64Codes = Any
    _Timedelta64Codes = Any

    _StrCodes = Any
    _BytesCodes = Any
    _VoidCodes = Any
