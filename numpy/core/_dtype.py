from numpy._core import _dtype
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in _dtype.__dir__():
    _globals[item] = getattr(_dtype, item)
