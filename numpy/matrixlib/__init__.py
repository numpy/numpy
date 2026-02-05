"""Sub-package containing the matrix class and related functions.

"""
from . import defmatrix
from .defmatrix import *
from .modular import matrix_inv_mod

__all__ = defmatrix.__all__

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester
