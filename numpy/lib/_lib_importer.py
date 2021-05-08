# This file defines which functions end up in the main namespace of NumPy.
# Note that the use of * import with an __all__ based on the locals(), means
# that the __all__ of the subpackages also defines what
# There are at least two approaches how this could be modified in the future
# conveniently:
# * We could create an `__export_to_numpy_namespace__` list in each package
#   and use this to define the `__all__` here (which is still copied into
#   the main namespace.
# * At least all functions copied to the main namespace probably already
#   use the `@set_module` decorator. So we could just use that (or something
#   similar) to manually add it to the main namespace.
#   (in which case we still need to make sure the file is imported)


import math

# Public submodules
# Note: recfunctions and (maybe) format are public too, but not imported
from . import scimath as emath

# Private submodules
from .type_check import *
from .index_tricks import *
from .function_base import *
from .nanfunctions import *
from .shape_base import *
from .stride_tricks import *
from .twodim_base import *
from .ufunclike import *
from .histograms import *

from .polynomial import *
from .utils import *
from .arraysetops import *
from .npyio import *
from .arraypad import *


__all__ = [k for k in locals().keys() if not k.startswith("_")]
