
from info_scipy_base import __doc__
from scipy_base_version import scipy_base_version as __version__

from ppimport import ppimport, ppimport_attr

# The following statement is equivalent to
#
#   from Matrix import Matrix as mat
#
# but avoids expensive LinearAlgebra import when
# Matrix is not used.
mat = ppimport_attr(ppimport('Matrix'), 'Matrix')

# Force Numeric to use scipy_base.fastumath instead of Numeric.umath.
import fastumath  # no need to use scipy_base.fastumath
import sys as _sys
_sys.modules['umath'] = fastumath

import Numeric
from Numeric import *

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

Inf = inf = fastumath.PINF
try:
    NAN = NaN = nan = fastumath.NAN
except AttributeError:
    NaN = NAN = nan = fastumath.PINF/fastumath.PINF

#---- testing ----#

def test(level=10):
    import unittest
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
    return runner

def test_suite(level=1):
    import scipy_test.testing
    import scipy_base
    this_mod = scipy_base
    # testing is the module that actually does all the testing...
    ignore = ['testing']
    return scipy_test.testing.harvest_test_suites(this_mod,ignore = ignore,
                                                  level=level)
