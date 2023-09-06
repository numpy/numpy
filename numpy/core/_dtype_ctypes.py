from numpy._core import _dtype_ctypes
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in _dtype_ctypes.__dir__():
    _globals[item] = getattr(_dtype_ctypes, item)
