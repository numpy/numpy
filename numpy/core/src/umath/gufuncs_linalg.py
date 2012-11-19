# python wrappers for linalg gufuncs

import numpy.core.umath_linalg as _impl

# usable "as is"
inner1d = _impl.inner1d
matrix_multiply = _impl.matrix_dot
det = _impl.det
slogdet = _impl.slogdet
inv = _impl.inv
cholesky = _impl.cholesky
quadratic_form = _impl.quadratic_form
add3 = _impl.add3
multiply3 = _impl.multiply3
multiply3_add = _impl.multiply3_add
multiply_add = _impl.multiply_add
multiply_add2 = _impl.multiply_add2
multiply4 = _impl.multiply4
multiply4_add = _impl.multiply4_add
eig = _impl.eig
eigvals = _impl.eigvals

# wrappers
def eigh(A, UPLO='L', **kw_args):
    """
    Computes the eigen values and eigenvectors for the square matrices in the inner dimensions of A, being those matrices symmetric/hermitian
    """
    if 'L' == UPLO:
        gufunc = _impl.eigh_lo
    else:
        gufunc = _impl.eigh_up

    return gufunc(A, **kw_args)


def eigvalsh(A, UPLO='L', **kw_args):
    """
    Computes the eigen values and eigenvectors for the square matrices in the inner dimensions of A, being those matrices symmetric/hermitian
    """
    if ('L' == UPLO):
        gufunc = _impl.eigvalsh_lo
    else:
        gufunc = _impl.eigvalsh_up

    return gufunc(A,**kw_args)


def solve(A,B,**kw_args):
    """
    Solves the systems of equations AX=B or Ax=b in the inner dimensions of A/B
    """
    if len(B.shape) == (len(A.shape) - 1):
        gufunc = _impl.solve1
    else:
        gufunc = _impl.solve

    return gufunc(A,B,**kw_args)


def svd(a, full_matrices=1, compute_uv=1 ,**kw_args):
    m = a.shape[-2]
    n = a.shape[-1]
    if 1 == compute_uv:
        if 1 == full_matrices:
            if m < n:
                gufunc = _impl.svd_m_f
            else:
                gufunc = _impl.svd_n_f
        else:
            if m < n:
                gufunc = _impl.svd_m_s
            else:
                gufunc = _impl.svd_n_s
    else:
        if m < n:
            gufunc = _impl.svd_m
        else:
            gufunc = _impl.svd_n
    return gufunc(a, **kw_args)


def chosolve(A, B, UPLO='L', **kw_args):
    if len(B.shape) == (len(A.shape) - 1):
        if 'L' == UPLO:
            gufunc = _impl.chosolve1_lo
        else:
            gufunc = _impl.chosolve1_up
    else:
        if 'L' == UPLO:
            gufunc = _impl.chosolve_lo
        else:
            gufunc = _impl.chosolve_up

    return gufunc(A, B, **kw_args)


def poinv(A, UPLO='L', **kw_args):
    if 'L' == UPLO:
        gufunc = _impl.poinv_lo
    else:
        gufunc = _impl.poinv_up

    return gufunc(A, **kw_args);
