"""
This module provides the Python-level wrappers for many NumPy functions
such as ``sum``, ``mean``, and ``argmax``.

Most functions here forward calls to lower-level implementations in NumPy
core. Accessing these functions via ``np.<function>`` may trigger warnings
that guide users toward the recommended APIs.
"""
def __getattr__(attr_name):
    from numpy._core import fromnumeric

    from ._utils import _raise_warning
    ret = getattr(fromnumeric, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.fromnumeric' has no attribute {attr_name}")
    _raise_warning(attr_name, "fromnumeric")
    return ret
