
from info_scipy_base import __doc__
from scipy_base_version import scipy_base_version as __version__

import numerix
from numerix import *

import limits
from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from matrix_base import *

from polynomial import *
from scimath import *
from machar import *
from pexec import *

from scipy_test.testing import ScipyTest
test = ScipyTest('scipy_base').test

import sys as _sys
if _sys.modules.has_key('scipy_base.Matrix') \
   and _sys.modules['scipy_base.Matrix'] is None:
    del _sys.modules['scipy_base.Matrix']

