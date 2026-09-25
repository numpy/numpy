"""Typing tests for `_core._ufunc_config`."""

from _typeshed import SupportsWrite
from collections.abc import Callable
from typing import Any, assert_type

import numpy as np
from numpy._core._ufunc_config import _ErrDict

def func(a: str, b: int) -> None: ...

class Write:
    def write(self, value: str) -> None: ...

assert_type(np.seterr(all=None), _ErrDict)
assert_type(np.seterr(divide="ignore"), _ErrDict)
assert_type(np.seterr(over="warn"), _ErrDict)
assert_type(np.seterr(under="call"), _ErrDict)
assert_type(np.seterr(invalid="raise"), _ErrDict)
assert_type(np.geterr(), _ErrDict)

assert_type(np.setbufsize(4096), int)
assert_type(np.getbufsize(), int)

assert_type(np.seterrcall(func), Callable[[str, int], Any] | SupportsWrite[str] | None)
assert_type(np.seterrcall(Write()), Callable[[str, int], Any] | SupportsWrite[str] | None)
assert_type(np.geterrcall(), Callable[[str, int], Any] | SupportsWrite[str] | None)

assert_type(np.errstate(call=func, all="call"), np.errstate)
assert_type(np.errstate(call=Write(), divide="log", over="log"), np.errstate)
