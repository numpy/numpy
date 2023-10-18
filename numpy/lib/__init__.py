"""
**Note:** almost all functions in the ``numpy.lib`` namespace
are also present in the main ``numpy`` namespace.  Please use the
functions as ``np.<funcname>`` where possible.

``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

Most contains basic functions that are used by several submodules and are
useful to have in the main name-space.

"""

# Public submodules
# Note: recfunctions and (maybe) format are public too, but not imported
from . import mixins
from . import scimath as emath

# Private submodules
# load module names. See https://github.com/networkx/networkx/issues/5838
from . import _type_check_impl
from . import _index_tricks_impl
from . import _nanfunctions_impl
from . import _function_base_impl
from . import _stride_tricks_impl
from . import _shape_base_impl
from . import stride_tricks
from . import _twodim_base_impl
from . import _ufunclike_impl
from . import _histograms_impl
from . import _utils_impl
from . import _arraysetops_impl
from . import _polynomial_impl
from . import _npyio_impl
from . import npyio
from . import arrayterator
from . import _arraypad_impl
from . import array_utils
from . import _version

from .arrayterator import Arrayterator
from ._version import *
from .introspect import *
from numpy._core._multiarray_umath import add_docstring, tracemalloc_domain
from numpy._core.function_base import add_newdoc

__all__ = ['emath']

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester

def __getattr__(attr):
    # Warn for reprecated attributes
    import math
    import warnings

    if attr == 'math':
        warnings.warn(
            "`np.lib.math` is a deprecated alias for the standard library "
            "`math` module (Deprecated Numpy 1.25). Replace usages of "
            "`numpy.lib.math` with `math`", DeprecationWarning, stacklevel=2)
        return math
    elif attr in (
        "histograms", "type_check", "nanfunctions", "function_base",
        "arraypad", "arraysetops", "ufunclike", "utils", "twodim_base",
        "shape_base", "polynomial", "index_tricks",
    ):
        raise AttributeError(
            f"`np.lib.{attr}` is now private. If you are using a public "
            "function, it should be available in the main numpy namespace, "
            "otherwise check the NumPy 2.0 migration guide."
        )
    else:
        raise AttributeError("module {!r} has no attribute "
                             "{!r}".format(__name__, attr))
