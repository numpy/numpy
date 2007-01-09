"""Backward compatible with LinearAlgebra from Numeric
"""
# This module is a lite version of the linalg.py module in SciPy which contains
# high-level Python interface to the LAPACK library.  The lite version
# only accesses the following LAPACK functions: dgesv, zgesv, dgeev,
# zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf, dpotrf.


__all__ = ['LinAlgError', 'solve_linear_equations',
           'inverse', 'cholesky_decomposition', 'eigenvalues',
           'Heigenvalues', 'generalized_inverse',
           'determinant', 'singular_value_decomposition',
           'eigenvectors',  'Heigenvectors',
           'linear_least_squares'
           ]

from numpy.core import transpose
import numpy.linalg as linalg

# Linear equations

LinAlgError = linalg.LinAlgError

def solve_linear_equations(a, b):
    return linalg.solve(a,b)

# Matrix inversion

def inverse(a):
    return linalg.inv(a)

# Cholesky decomposition

def cholesky_decomposition(a):
    return linalg.cholesky(a)

# Eigenvalues

def eigenvalues(a):
    return linalg.eigvals(a)

def Heigenvalues(a, UPLO='L'):
    return linalg.eigvalsh(a,UPLO)

# Eigenvectors

def eigenvectors(A):
    w, v = linalg.eig(A)
    return w, transpose(v)

def Heigenvectors(A):
    w, v = linalg.eigh(A)
    return w, transpose(v)

# Generalized inverse

def generalized_inverse(a, rcond = 1.e-10):
    return linalg.pinv(a, rcond)

# Determinant

def determinant(a):
    return linalg.det(a)

# Linear Least Squares

def linear_least_squares(a, b, rcond=1.e-10):
    """returns x,resids,rank,s
where x minimizes 2-norm(|b - Ax|)
      resids is the sum square residuals
      rank is the rank of A
      s is the rank of the singular values of A in descending order

If b is a matrix then x is also a matrix with corresponding columns.
If the rank of A is less than the number of columns of A or greater than
the number of rows, then residuals will be returned as an empty array
otherwise resids = sum((b-dot(A,x)**2).
Singular values less than s[0]*rcond are treated as zero.
"""
    return linalg.lstsq(a,b,rcond)

def singular_value_decomposition(A, full_matrices=0):
    return linalg.svd(A, full_matrices)
