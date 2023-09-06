from numpy._core import umath
from ._utils import _raise_warning

_raise_warning("*")

_globals = globals()

for item in umath.__dir__():
    _globals[item] = getattr(umath, item)
