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
import math

from numpy.core._multiarray_umath import tracemalloc_domain
from numpy.version import version as __version__

# Private submodules
# load module names. See https://github.com/networkx/networkx/issues/5838
# Public submodules
# Note: recfunctions and (maybe) format are public too, but not imported
from . import (_version, arraypad, arraysetops, arrayterator, function_base,
               histograms, index_tricks, mixins, nanfunctions, npyio,
               polynomial)
from . import scimath as emath
from . import (shape_base, stride_tricks, twodim_base, type_check, ufunclike,
               utils)
from ._version import *
from .arraypad import *
from .arraysetops import *
from .arrayterator import Arrayterator
from .function_base import *
from .histograms import *
from .index_tricks import *
from .nanfunctions import *
from .npyio import *
from .polynomial import *
from .shape_base import *
from .stride_tricks import *
from .twodim_base import *
from .type_check import *
from .ufunclike import *
from .utils import *

__all__ = ['emath', 'math', 'tracemalloc_domain', 'Arrayterator']
__all__ += type_check.__all__
__all__ += index_tricks.__all__
__all__ += function_base.__all__
__all__ += shape_base.__all__
__all__ += stride_tricks.__all__
__all__ += twodim_base.__all__
__all__ += ufunclike.__all__
__all__ += arraypad.__all__
__all__ += polynomial.__all__
__all__ += utils.__all__
__all__ += arraysetops.__all__
__all__ += npyio.__all__
__all__ += nanfunctions.__all__
__all__ += histograms.__all__

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester
