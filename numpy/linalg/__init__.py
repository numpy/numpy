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

from .linalg import (Inf,
                     LinAlgError,
                     abs,
                     absolute_import,
                     add,
                     all,
                     amax,
                     amin,
                     array,
                     asanyarray,
                     asarray,
                     asfarray,
                     atleast_2d,
                     broadcast,
                     cdouble,
                     cholesky,
                     complexfloating,
                     cond,
                     count_nonzero,
                     csingle,
                     det,
                     divide,
                     division,
                     dot,
                     double,
                     eig,
                     eigh,
                     eigvals,
                     eigvalsh,
                     empty,
                     empty_like,
                     errstate,
                     fastCopyAndTranspose,
                     finfo,
                     fortran_int,
                     get_linalg_error_extobj,
                     geterrobj,
                     inexact,
                     intc,
                     intp,
                     inv,
                     isComplexType,
                     isfinite,
                     lapack_lite,
                     longdouble,
                     lstsq,
                     matmul,
                     matrix_power,
                     matrix_rank,
                     maximum,
                     moveaxis,
                     multi_dot,
                     multiply,
                     newaxis,
                     norm,
                     normalize_axis_index,
                     object_,
                     ones,
                     pinv,
                     print_function,
                     product,
                     qr,
                     ravel,
                     single,
                     size,
                     slogdet,
                     solve,
                     sqrt,
                     sum,
                     svd,
                     swapaxes,
                     tensorinv,
                     tensorsolve,
                     transpose,
                     triu,
                     warnings,
                     zeros)

from numpy.testing import _numpy_tester
test = _numpy_tester().test
bench = _numpy_tester().bench
