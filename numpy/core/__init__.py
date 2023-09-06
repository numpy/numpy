from numpy import _core
from ._utils import _raise_warning


def __getattr__(attr_name):

    attr = getattr(_core, attr_name)
    _raise_warning(attr_name)
    return attr
