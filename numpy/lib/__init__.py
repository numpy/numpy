"""
**Note:** almost all functions in the ``numpy.lib.submodule`` namespaces
are also present in the main ``numpy`` namespace.  Please use the
functions as ``np.<funcname>`` where possible.

``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

Most contains basic functions that are used by several submodules and are
useful to have in the main name-space.

"""
# NOTE: This file is the _public_ interface for additional submodules
# housed in `numpy.lib`. The private import mechanism lives in the
# `_lib_importer.py` file, which defines `__all__` to include all names to
# be added to the main namespace!

from . import scimath as emath  # import explicitly to rename

from .arrayterator import Arrayterator
from numpy.core._multiarray_umath import tracemalloc_domain
from ._version import NumpyVersion

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester

from . import _lib_importer as _hidden_lib_namespace


# Ensure that __all__ only contains those things that we want it to contain.
# Because this is an __init__, we would otherwise include all submodules.
#
# NOTE: The symbols not included in `__all__` will be hidden, but do not go
#       through `__getattr__` (see `__dir__`).
__all__ = {'emath', 'Arrayterator', 'tracemalloc_domain', 'NumpyVersion'}

__lib_submodules_ = {
    'type_check',
    'emath',
    'arraysetops',
    'utils',
    'format',
    'npyio',
    'arrayterator',
    'mixins',
    'stride_tricks',
    'twodim_base',
    'histograms',
    'index_tricks',
    'nanfunctions',
    'shape_base',
}
# Add submodules to all (and finalize all)
__all__.update(__lib_submodules_)
__all__ = sorted(__all__)

# Add hidden submodules:
__lib_submodules_.update({
    'arraypad',
    'function_base',
    'polynomial',
    'scimath',
    "ufunclike",
    })


def __getattr__(name):
    if name in __lib_submodules_:
        import importlib
        # Allow lazy import (and avoid requiring explicit import), at the
        # time of writing `mixins` may be the only namespace using this.
        return importlib.import_module("." + name, __name__)

    try:
        attribute = getattr(_hidden_lib_namespace, name)
    except AttributeError:
        pass
    else:
        # Adding a warning in this branch would allow us to properly deprecate
        # all hidden attributes. Raise an error here to test NumPy itself.
        return attribute

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Overriding __dir__ hides symbols from tab completion for example.
    # Note that it does _not_ hide symbols from `np.lib.__dict__()`, to do
    # that, we would have to delete them from the locals here and hide them
    # into `__getattr__`.  But this seems sufficient, e.g. for tab completion.
    return __all__
