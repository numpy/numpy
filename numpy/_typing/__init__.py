"""Private counterpart of ``numpy.typing``."""

from __future__ import annotations

from numpy import ufunc
from numpy.core.overrides import set_module
from typing import TYPE_CHECKING, final


@final  # Disallow the creation of arbitrary `NBitBase` subclasses
@set_module("numpy.typing")
class NBitBase:
    """
    A type representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    .. versionadded:: 1.20

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating
    a function that takes a float and integer of arbitrary precision
    as arguments and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import TypeVar, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> T1 = TypeVar("T1", bound=npt.NBitBase)
        >>> T2 = TypeVar("T2", bound=npt.NBitBase)

        >>> def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
            "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()


# Silence errors about subclassing a `@final`-decorated class
class _256Bit(NBitBase):  # type: ignore[misc]
    pass

class _128Bit(_256Bit):  # type: ignore[misc]
    pass

class _96Bit(_128Bit):  # type: ignore[misc]
    pass

class _80Bit(_96Bit):  # type: ignore[misc]
    pass

class _64Bit(_80Bit):  # type: ignore[misc]
    pass

class _32Bit(_64Bit):  # type: ignore[misc]
    pass

class _16Bit(_32Bit):  # type: ignore[misc]
    pass

class _8Bit(_16Bit):  # type: ignore[misc]
    pass


from ._nested_sequence import _NestedSequence
from ._nbit import (
    _NBitByte,
    _NBitShort,
    _NBitIntC,
    _NBitIntP,
    _NBitInt,
    _NBitLongLong,
    _NBitHalf,
    _NBitSingle,
    _NBitDouble,
    _NBitLongDouble,
)
from ._char_codes import (
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
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
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)
from ._scalars import (
    _CharLike_co,
    _BoolLike_co,
    _UIntLike_co,
    _IntLike_co,
    _FloatLike_co,
    _ComplexLike_co,
    _TD64Like_co,
    _NumberLike_co,
    _ScalarLike_co,
    _VoidLike_co,
)
from ._shape import _Shape, _ShapeLike
from ._dtype_like import (
    DTypeLike as DTypeLike,
    _DTypeLike,
    _SupportsDType,
    _VoidDTypeLike,
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
    _DTypeLikeTD64,
    _DTypeLikeDT64,
    _DTypeLikeObject,
    _DTypeLikeVoid,
    _DTypeLikeStr,
    _DTypeLikeBytes,
    _DTypeLikeComplex_co,
)
from ._array_like import (
    ArrayLike as ArrayLike,
    _ArrayLike,
    _FiniteNestedSequence,
    _SupportsArray,
    _SupportsArrayFunc,
    _ArrayLikeInt,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeVoid_co,
    _ArrayLikeStr_co,
    _ArrayLikeBytes_co,
)
from ._generic_alias import (
    NDArray as NDArray,
    _DType,
    _GenericAlias,
)

if TYPE_CHECKING:
    from ._ufunc import (
        _UFunc_Nin1_Nout1,
        _UFunc_Nin2_Nout1,
        _UFunc_Nin1_Nout2,
        _UFunc_Nin2_Nout2,
        _GUFunc_Nin2_Nout1,
    )
else:
    # Declare the (type-check-only) ufunc subclasses as ufunc aliases during
    # runtime; this helps autocompletion tools such as Jedi (numpy/numpy#19834)
    _UFunc_Nin1_Nout1 = ufunc
    _UFunc_Nin2_Nout1 = ufunc
    _UFunc_Nin1_Nout2 = ufunc
    _UFunc_Nin2_Nout2 = ufunc
    _GUFunc_Nin2_Nout1 = ufunc
