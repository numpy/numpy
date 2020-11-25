"""A module containing `numpy`-specific plugins for mypy."""

import typing as t

import numpy as np

import mypy.types
from mypy.types import Type
from mypy.plugin import Plugin, AnalyzeTypeContext

__all__: t.List[str] = []

HookFunc = t.Callable[[AnalyzeTypeContext], Type]


def _get_precision_dict() -> t.Dict[str, str]:
    names = [
        ("_NBitByte", np.byte),
        ("_NBitShort", np.short),
        ("_NBitIntC", np.intc),
        ("_NBitIntP", np.intp),
        ("_NBitInt", np.int_),
        ("_NBitLongLong", np.longlong),

        ("_NBitHalf", np.half),
        ("_NBitSingle", np.single),
        ("_NBitDouble", np.double),
        ("_NBitLongDouble", np.longdouble),
    ]
    ret = {}
    for name, typ in names:
        n: int = 8 * typ().dtype.alignment
        ret[f'numpy.typing._nbit.{name}'] = f"numpy._{n}Bit"
    return ret


#: A dictionary mapping type-aliases in `numpy.typing._nbit` to
#: concrete `numpy.typing.NBitBase` subclasses.
_PRECISION_DICT = _get_precision_dict()


def _hook(ctx: AnalyzeTypeContext) -> Type:
    """Replace a type-alias with a concrete ``NBitBase`` subclass."""
    typ, _, api = ctx
    name = typ.name.split(".")[-1]
    name_new = _PRECISION_DICT[f"numpy.typing._nbit.{name}"]
    return api.named_type(name_new)


class _NumpyPlugin(Plugin):
    """A plugin for assigning platform-specific `numpy.number` precisions."""

    def get_type_analyze_hook(self, fullname: str) -> t.Optional[HookFunc]:
        if fullname in _PRECISION_DICT:
            return _hook
        return None


def plugin(version: str) -> t.Type[_NumpyPlugin]:
    """An entry-point for mypy."""
    return _NumpyPlugin
