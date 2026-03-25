"""Test the runtime usage of `numpy.typing`."""

from typing import (
    Any,
    NamedTuple,
    Self,
    TypeAliasType,
    get_args,
    get_origin,
    get_type_hints,
)

import pytest

import numpy as np
import numpy._typing as _npt
import numpy.typing as npt


class TypeTup(NamedTuple):
    typ: type  # type expression
    args: tuple[type, ...]  # generic type parameters or arguments
    origin: type | None  # e.g. `UnionType` or `GenericAlias`

    @classmethod
    def from_type_alias(cls, alias: TypeAliasType, /) -> Self:
        # PEP 695 `type _ = ...` aliases wrap the type expression as a
        # `types.TypeAliasType` instance with a `__value__` attribute.
        tp = alias.__value__
        return cls(typ=tp, args=get_args(tp), origin=get_origin(tp))


TYPES = {
    "ArrayLike": TypeTup.from_type_alias(npt.ArrayLike),
    "DTypeLike": TypeTup.from_type_alias(npt.DTypeLike),
    "NBitBase": TypeTup(npt.NBitBase, (), None),  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]
    "NDArray": TypeTup.from_type_alias(npt.NDArray),
}


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_args(name: type, tup: TypeTup) -> None:
    """Test `typing.get_args`."""
    typ, ref = tup.typ, tup.args
    out = get_args(typ)
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_origin(name: type, tup: TypeTup) -> None:
    """Test `typing.get_origin`."""
    typ, ref = tup.typ, tup.origin
    out = get_origin(typ)
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints`."""
    typ = tup.typ

    def func(a: typ) -> None: pass

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints_str(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints` with string-representation of types."""
    typ_str, typ = f"npt.{name}", tup.typ

    def func(a: typ_str) -> None: pass

    out = get_type_hints(func)
    ref = {"a": getattr(npt, str(name)), "return": type(None)}
    assert out == ref


def test_keys() -> None:
    """Test that ``TYPES.keys()`` and ``numpy.typing.__all__`` are synced."""
    keys = TYPES.keys()
    ref = set(npt.__all__)
    assert keys == ref


PROTOCOLS: dict[str, tuple[type[Any], object]] = {
    "_SupportsArray": (_npt._SupportsArray, np.arange(10)),
    "_SupportsArrayFunc": (_npt._SupportsArrayFunc, np.arange(10)),
    "_NestedSequence": (_npt._NestedSequence, [1]),
}


@pytest.mark.parametrize("cls,obj", PROTOCOLS.values(), ids=PROTOCOLS.keys())
class TestRuntimeProtocol:
    def test_isinstance(self, cls: type[Any], obj: object) -> None:
        assert isinstance(obj, cls)
        assert not isinstance(None, cls)

    def test_issubclass(self, cls: type[Any], obj: object) -> None:
        assert issubclass(type(obj), cls)
        assert not issubclass(type(None), cls)
