from numpy._core import multiarray
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in multiarray.__dir__():
    _globals[item] = getattr(multiarray, item)
