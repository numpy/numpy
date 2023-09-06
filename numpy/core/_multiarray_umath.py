from numpy._core import _multiarray_umath
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in _multiarray_umath.__dir__():
    _globals[item] = getattr(_multiarray_umath, item)
