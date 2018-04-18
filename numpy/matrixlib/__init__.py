"""Sub-package containing the matrix class and related functions.

"""
from __future__ import division as _, absolute_import as _, print_function as _

from .defmatrix import *

__all__ = defmatrix.__all__

from numpy.testing._private.pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
