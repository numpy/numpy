from numpy._core import _internal
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in _internal.__dir__():
    _globals[item] = getattr(_internal, item)
