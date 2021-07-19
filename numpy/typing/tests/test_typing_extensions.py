"""Tests for the optional typing-extensions dependency."""

import sys
import types
import inspect
import importlib

import typing_extensions
import numpy.typing as npt


def _is_sub_module(obj: object) -> bool:
    """Check if `obj` is a `numpy.typing` submodule."""
    return inspect.ismodule(obj) and obj.__name__.startswith("numpy.typing")


def _is_dunder(name: str) -> bool:
    """Check whether `name` is a dunder."""
    return name.startswith("__") and name.endswith("__")


def _clear_attr(module: types.ModuleType) -> None:
    """Clear all (non-dunder) module-level attributes."""
    del_names = [name for name in vars(module) if not _is_dunder(name)]
    for name in del_names:
        delattr(module, name)


MODULES = {"numpy.typing": npt}
MODULES.update({
    f"numpy.typing.{k}": v for k, v in vars(npt).items() if _is_sub_module(v)
})


def test_no_typing_extensions() -> None:
    """Import `numpy.typing` in the absence of typing-extensions.

    Notes
    -----
    Ideally, we'd just run the normal typing tests in an environment where
    typing-extensions is not installed, but unfortunatelly this is currently
    impossible as it is an indirect hard dependency of pytest.

    """
    assert "typing_extensions" in sys.modules

    try:
        sys.modules["typing_extensions"] = None
        for name, module in MODULES.items():
            _clear_attr(module)
            assert importlib.reload(module), name
    finally:
        sys.modules["typing_extensions"] = typing_extensions
        for module in MODULES.values():
            _clear_attr(module)
            importlib.reload(module)
