from numpy._core import _multiarray_umath
from numpy import ufunc

for item in _multiarray_umath.__dir__():
    # ufuncs appear in pickles with a path in numpy.core._multiarray_umath
    # and so must import from this namespace without warning or error
    attr = getattr(_multiarray_umath, item)
    if isinstance(attr, ufunc):
        globals()[item] = attr

def __getattr__(attr_name):
    from numpy._core import _multiarray_umath
    from ._utils import _raise_warning
    ret = getattr(_multiarray_umath, attr_name, None)
    if ret is None:
        raise AttributeError(
            "module 'numpy.core._multiarray_umath' has no attribute "
            f"{attr_name}")
    _raise_warning(attr_name, "_multiarray_umath")
    return ret


del _multiarray_umath, ufunc
