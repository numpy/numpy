from numpy._core import numeric
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in numeric.__dir__():
    _globals[item] = getattr(numeric, item)
