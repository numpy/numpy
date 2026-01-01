from _typeshed import Incomplete
from collections.abc import Buffer, Sequence
from typing import Any, Generic, Self, SupportsIndex, overload, override
from typing_extensions import TypeVar

import numpy as np
from numpy import _ByteOrder, _ToIndices
from numpy._typing import (
    DTypeLike,
    _AnyShape,
    _ArrayLikeBool_co,
    _DTypeLike,
    _HasDType,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
    _VoidDTypeLike,
)

from .core import MaskedArray

__all__ = ["MaskedRecords", "mrecarray", "fromarrays", "fromrecords", "fromtextfile", "addfield"]

###

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)

type _Ignored = object

###
# mypy: disable-error-code=no-untyped-def

class MaskedRecords(MaskedArray[_ShapeT_co, _DTypeT_co], Generic[_ShapeT_co, _DTypeT_co]):
    _mask: Any
    _fill_value: Any

    def __new__(
        cls,
        shape: _ShapeLike,
        dtype: DTypeLike | None = None,
        buf: Buffer | None = None,
        offset: SupportsIndex = 0,
        strides: _ShapeLike | None = None,
        formats: DTypeLike | None = None,
        names: str | Sequence[str] | None = None,
        titles: str | Sequence[str] | None = None,
        byteorder: _ByteOrder | None = None,
        aligned: bool = False,
        mask: _ArrayLikeBool_co = ...,
        hard_mask: bool = False,
        fill_value: _ScalarLike_co | None = None,
        keep_mask: bool = True,
        copy: bool = False,
        **options: _Ignored,
    ) -> Self: ...

    #
    @property
    @override
    def _data(self, /) -> np.recarray[_ShapeT_co, _DTypeT_co]: ...
    @property
    def _fieldmask(self, /) -> np.ndarray[_ShapeT_co, np.dtype[np.bool]] | np.bool: ...

    #
    @override
    def __array_finalize__(self, obj: np.ndarray) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __getitem__(self, indx: str | _ToIndices, /) -> Incomplete: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __setitem__(self, indx: str | _ToIndices, value: Incomplete, /) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    # unlike `MaskedArray`, these two methods don't return `Self`
    @override
    def harden_mask(self) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def soften_mask(self) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    # keep in sync with `MaskedArray.view`, but without the `fill_value`
    @override  # type: ignore[override]
    @overload  # ()
    def view(self, /, dtype: None = None, type: None = None) -> Self: ...
    @overload  # (dtype: DTypeT)
    def view[DTypeT: np.dtype](
        self, /, dtype: DTypeT | _HasDType[DTypeT], type: None = None
    ) -> MaskedRecords[_ShapeT_co, DTypeT]: ...
    @overload  # (dtype: dtype[ScalarT])
    def view[ScalarT: np.generic](
        self, /, dtype: _DTypeLike[ScalarT], type: None = None
    ) -> MaskedRecords[_ShapeT_co, np.dtype[ScalarT]]: ...
    @overload  # ([dtype: _, ]*, type: ArrayT)
    def view[ArrayT: np.ndarray](self, /, dtype: DTypeLike | None = None, *, type: type[ArrayT]) -> ArrayT: ...
    @overload  # (dtype: _, type: ArrayT)
    def view[ArrayT: np.ndarray](self, /, dtype: DTypeLike | None, type: type[ArrayT]) -> ArrayT: ...
    @overload  # (dtype: ArrayT, /)
    def view[ArrayT: np.ndarray](self, /, dtype: type[ArrayT], type: None = None) -> ArrayT: ...
    @overload  # (dtype: <like `DTypeLike` but without `_DTypeLike[Any]`>)
    def view(self, /, dtype: _VoidDTypeLike | str | None, type: None = None) -> MaskedRecords[_ShapeT_co, np.dtype]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    # unlike `MaskedArray` and `ndarray`, this `copy` method has no `order` parameter
    @override
    def copy(self, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

mrecarray = MaskedRecords

def fromarrays(
    arraylist,
    dtype=None,
    shape=None,
    formats=None,
    names=None,
    titles=None,
    aligned=False,
    byteorder=None,
    fill_value=None,
): ...

def fromrecords(
    reclist,
    dtype=None,
    shape=None,
    formats=None,
    names=None,
    titles=None,
    aligned=False,
    byteorder=None,
    fill_value=None,
    mask=...,
): ...

def fromtextfile(
    fname,
    delimiter=None,
    commentchar="#",
    missingchar="",
    varnames=None,
    vartypes=None,
): ...

def addfield(mrecord, newfield, newfieldname=None): ...
