"""Sub-package containing the matrix class and related functions."""
from defmatrix import *

__all__ = defmatrix.__all__

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
