import typing as t

import mypy.types
from mypy.types import Type, Instance
from mypy.plugin import Plugin, MethodContext, FunctionContext

_NAMES: t.Mapping[str, str] = {
    "numpy.signedinteger": "numpy.int",
    "numpy.unsignedinteger": "numpy.uint",
    "numpy.floating": "numpy.float",
    "numpy.complexfloating": "numpy.complex",
}

_PRECISION: t.Mapping[str, int] = {
    "numpy._64Bit": 64,
    "numpy._32Bit": 32,
    "numpy._16Bit": 16,
    "numpy._8Bit": 8,
}


def _hook(ctx: t.Union[FunctionContext, MethodContext]) -> Type:
    api = ctx.api
    ret_type = ctx.default_return_type
    if not isinstance(ret_type, Instance):
        return ret_type

    # There are 3 dict lookups where a `KeyError` could potentially be raised
    # If this hapens, return the original `ret_type` in unaltered form
    try:
        name = _NAMES[ret_type.type.fullname]  # dict lookup #1

        # Parse the precision
        _precision = ret_type.args[0]
        if not isinstance(_precision, Instance):
            return ret_type
        precision = _PRECISION[_precision.type.fullname]  # dict lookup #2

        if name == "numpy.complex":
            precision *= 2

        return api.named_type(f'{name}{precision}')  # Dict lookup #3
    except KeyError:
        return ret_type


class NumberPlugin(Plugin):
    def get_method_hook(self, fullname: str
                        ) -> t.Optional[t.Callable[[MethodContext], Type]]:
        return _hook

    def get_function_hook(self, fullname: str
                          ) -> t.Optional[t.Callable[[FunctionContext], Type]]:
        return _hook


def plugin(version: str) -> t.Type[NumberPlugin]:
    return NumberPlugin
