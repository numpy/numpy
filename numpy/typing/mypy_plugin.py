"""A module containing `numpy`-specific plugins for mypy."""

from __future__ import annotations

import typing as t

import numpy as np

try:
    import mypy.types
    from mypy.types import Type
    from mypy.plugin import Plugin, AnalyzeTypeContext
    from mypy.nodes import MypyFile, ImportFrom, Statement
    from mypy.build import PRI_MED

    _HookFunc = t.Callable[[AnalyzeTypeContext], Type]
    MYPY_EX: t.Optional[ModuleNotFoundError] = None
except ModuleNotFoundError as ex:
    MYPY_EX = ex

__all__: t.List[str] = []


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
        n: int = 8 * typ().dtype.itemsize
        ret[f'numpy.typing._nbit.{name}'] = f"numpy._{n}Bit"
    return ret


def _get_extended_precision_list() -> t.List[str]:
    extended_types = [np.ulonglong, np.longlong, np.longdouble, np.clongdouble]
    extended_names = {
        "uint128",
        "uint256",
        "int128",
        "int256",
        "float80",
        "float96",
        "float128",
        "float256",
        "complex160",
        "complex192",
        "complex256",
        "complex512",
    }
    return [i.__name__ for i in extended_types if i.__name__ in extended_names]


def _get_c_intp_name() -> str:
    if np.ctypeslib.c_intp is np.intp:
        return "c_int64"  # Plan B, in case `ctypes` fails to import
    else:
        return np.ctypeslib.c_intp.__qualname__


#: A dictionary mapping type-aliases in `numpy.typing._nbit` to
#: concrete `numpy.typing.NBitBase` subclasses.
_PRECISION_DICT: t.Final = _get_precision_dict()

#: A list with the names of all extended precision `np.number` subclasses.
_EXTENDED_PRECISION_LIST: t.Final = _get_extended_precision_list()

#: The name of the ctypes quivalent of `np.intp`
_C_INTP: t.Final = _get_c_intp_name()


def _hook(ctx: AnalyzeTypeContext) -> Type:
    """Replace a type-alias with a concrete ``NBitBase`` subclass."""
    typ, _, api = ctx
    name = typ.name.split(".")[-1]
    name_new = _PRECISION_DICT[f"numpy.typing._nbit.{name}"]
    return api.named_type(name_new)


if t.TYPE_CHECKING or MYPY_EX is None:
    def _index(iterable: t.Iterable[Statement], id: str) -> int:
        """Identify the first ``ImportFrom`` instance the specified `id`."""
        for i, value in enumerate(iterable):
            if getattr(value, "id", None) == id:
                return i
        else:
            raise ValueError("Failed to identify a `ImportFrom` instance "
                             f"with the following id: {id!r}")

    def _override_imports(
        file: MypyFile,
        module: str,
        imports: t.List[t.Tuple[str, t.Optional[str]]],
    ) -> None:
        """Override the first `module`-based import with new `imports`."""
        # Construct a new `from module import y` statement
        import_obj = ImportFrom(module, 0, names=imports)
        import_obj.is_top_level = True

        # Replace the first `module`-based import statement with `import_obj`
        for lst in [file.defs, file.imports]:  # type: t.List[Statement]
            i = _index(lst, module)
            lst[i] = import_obj

    class _NumpyPlugin(Plugin):
        """A mypy plugin for handling versus numpy-specific typing tasks."""

        def get_type_analyze_hook(self, fullname: str) -> t.Optional[_HookFunc]:
            """Set the precision of platform-specific `numpy.number` subclasses.

            For example: `numpy.int_`, `numpy.longlong` and `numpy.longdouble`.
            """
            if fullname in _PRECISION_DICT:
                return _hook
            return None

        def get_additional_deps(self, file: MypyFile) -> t.List[t.Tuple[int, str, int]]:
            """Handle all import-based overrides.

            * Import platform-specific extended-precision `numpy.number`
              subclasses (*e.g.* `numpy.float96`, `numpy.float128` and
              `numpy.complex256`).
            * Import the appropriate `ctypes` equivalent to `numpy.intp`.

            """
            ret = [(PRI_MED, file.fullname, -1)]

            if file.fullname == "numpy":
                _override_imports(
                    file, "numpy.typing._extended_precision",
                    imports=[(v, v) for v in _EXTENDED_PRECISION_LIST],
                )
            elif file.fullname == "numpy.ctypeslib":
                _override_imports(
                    file, "ctypes",
                    imports=[(_C_INTP, "_c_intp")],
                )
            return ret

    def plugin(version: str) -> t.Type[_NumpyPlugin]:
        """An entry-point for mypy."""
        return _NumpyPlugin

else:
    def plugin(version: str) -> t.Type[_NumpyPlugin]:
        """An entry-point for mypy."""
        raise MYPY_EX
