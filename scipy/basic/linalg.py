"""Lite version of scipy.linalg.
"""

from scipy import transpose
from basic_lite import *

def singular_value_decomposition(A, full_matrices=0):
    return svd(A, 0)

def eigenvectors(A):
    w, v = eig(A)
    return w, transpose(v)

def Heigenvectors(A):
    w, v = eigh(A)
    return w, transpose(v)

inv = inverse
solve = solve_linear_equations
cholesky = cholesky_decomposition
eigvals = eigenvalues
eigvalsh = Heigenvalues
pinv = generalized_inverse
det = determinant
lstsq = linear_least_squares

