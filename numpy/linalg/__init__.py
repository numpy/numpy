"""
Core Linear Algebra Tools
=========================

=============== ==========================================================
Linear algebra basics
==========================================================================
norm            Vector or matrix norm
inv             Inverse of a square matrix
solve           Solve a linear system of equations
det             Determinant of a square matrix
slogdet         Logarithm of the determinant of a square matrix
lstsq           Solve linear least-squares problem
pinv            Pseudo-inverse (Moore-Penrose) calculated using a singular
                value decomposition
matrix_power    Integer power of a square matrix
matrix_rank     Calculate matrix rank using an SVD-based method
=============== ==========================================================

=============== ==========================================================
Eigenvalues and decompositions
==========================================================================
eig             Eigenvalues and vectors of a square matrix
eigh            Eigenvalues and eigenvectors of a Hermitian matrix
eigvals         Eigenvalues of a square matrix
eigvalsh        Eigenvalues of a Hermitian matrix
qr              QR decomposition of a matrix
svd             Singular value decomposition of a matrix
cholesky        Cholesky decomposition of a matrix
=============== ==========================================================

=============== ==========================================================
Tensor operations
==========================================================================
tensorsolve     Solve a linear tensor equation
tensorinv       Calculate an inverse of a tensor
=============== ==========================================================

=============== ==========================================================
Exceptions
==========================================================================
LinAlgError     Indicates a failed linear algebra operation
=============== ==========================================================

"""
from __future__ import division, absolute_import, print_function

# To get sub-modules
from .info import __doc__

from .linalg import *

import numpy.distutils.system_info as sinfo
if len(sinfo.get_info('openblas')) > 2:
    # openblas >= 0.3.4 provides version information;
    # numpy/distutils/system_info has already initialized
    # at this stage, so just update the dictionary for
    # openblas info
    from numpy.linalg import openblas_config
    openblas_config_str = openblas_config._openblas_info()
    openblas_config_list = openblas_config_str.split()
    # slightly obscure API for system_info objects;
    # they weren't really intended for public modification
    # as the docs clearly indicate
    info = sinfo.system_info.saved_results['openblas_info']
    new_info = {}
    if openblas_config_list[0] == b"OpenBLAS":
        # version string will be present
        new_info['version'] = openblas_config_list[1]
    else:
        # older OpenBLAS config API
        new_info['version'] = None
    info.update(new_info)

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
