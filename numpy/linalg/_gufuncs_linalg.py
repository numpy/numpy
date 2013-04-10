"""Linear Algebra functions implemented as gufuncs, so they broadcast.

.. warning:: This module is only for testing, the functionality will be
   integrated into numpy.linalg proper.

=======================
 gufuncs_linalg module
=======================

gufuncs_linalg implements a series of linear algebra functions as gufuncs.
Most of these functions are already present in numpy.linalg, but as they
are implemented using gufunc kernels they can be broadcasting. Some parts
that are python in numpy.linalg are implemented inside C functions, as well
as the iteration used when used on vectors. This can result in faster
execution as well.

In addition, there are some ufuncs thrown in that implement fused operations
over numpy vectors that can result in faster execution on large vector 
compared to non-fused versions (for example: multiply_add, multiply3).

In fact, gufuncs_linalg is a very thin wrapper of python code that wraps
the actual kernels (gufuncs). This wrapper was needed in order to provide
a sane interface for some functions. Mostly working around limitations on
what can be described in a gufunc signature. Things like having one dimension
of a result depending on the minimum of two dimensions of the sources (like
in svd) or passing an uniform keyword parameter to the whole operation
(like UPLO on functions over symmetric/hermitian matrices).

The gufunc kernels are in a c module named _umath_linalg, that is imported
privately in gufuncs_linalg.

==========
 Contents
==========

Here is an enumeration of the functions. These are the functions exported by
the module and should appear in its __all__ attribute. All the functions
contain a docstring explaining them in detail.

General
=======
- inner1d
- innerwt
- matrix_multiply
- quadratic_form

Lineal Algebra
==============
- det
- slogdet
- cholesky
- eig
- eigvals
- eigh
- eigvalsh
- solve
- svd
- chosolve
- inv
- poinv

Fused Operations
================
- add3
- multiply3
- multiply3_add
- multiply_add
- multiply_add2
- multiply4
- multiply4_add

================
 Error Handling
================
Unlike the numpy.linalg module, this module does not use exceptions to notify
errors in the execution of the kernels. As these functions are thougth to be 
used in a vector way it didn't seem appropriate to raise exceptions on failure
of an element. So instead, when an error computing an element occurs its 
associated result will be set to an invalid value (all NaNs).

Exceptions can occur if the arguments fail to map properly to the underlying
gufunc (due to signature mismatch, for example).

================================
 Notes about the implementation
================================
Where possible, the wrapper functions map directly into a gufunc implementing
the computation.

That's not always the case, as due to limitations of the gufunc interface some
functions cannot be mapped straight into a kernel.

Two cases come to mind:
- An uniform parameter is needed to configure the way the computation is 
performed (like UPLO in the functions working on symmetric/hermitian matrices)
- svd, where it was impossible to map the function to a gufunc signature.

In the case of uniform parameters like UPLO, there are two separate entry points
in the C module that imply either 'U' or 'L'. The wrapper just selects the
kernel to use by checking the appropriate keyword parameter. This way a
function interface similar to numpy.linalg can be kept.

In the case of SVD not only there were problems with the support of keyword
arguments. There was the added problem of the signature system not being able
to cope with the needs of this functions. Just for the singular values a
a signature like (m,n)->(min(m,n)) was needed. This has been worked around by
implementing different kernels for the cases where min(m,n) == m and where
min(m,n) == n. The wrapper code automatically selects the appropriate one.


"""

from __future__ import division, absolute_import, print_function


__all__ = ['inner1d', 'dotc1d', 'innerwt', 'matrix_multiply', 'det', 'slogdet',
           'inv', 'cholesky', 'quadratic_form', 'add3', 'multiply3', 
           'multiply3_add', 'multiply_add', 'multiply_add2', 'multiply4', 
           'multiply4_add', 'eig', 'eigvals', 'eigh', 'eigvalsh', 'solve', 
           'svd', 'chosolve', 'poinv']

import numpy as np

from . import _umath_linalg as _impl


def inner1d(a, b, **kwargs):
    """
    Compute the dot product of vectors over the inner dimension, with
    broadcasting.

    Parameters
    ----------
    a : (..., N) array
        Input array
    b : (..., N) array
        Input array

    Returns
    -------
    inner : (...) array
        dot product over the inner dimension.

    Notes
    -----
    Numpy broadcasting rules apply when matching dimensions.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    For single and double types this is equivalent to dotc1d.

    Maps to Blas functions sdot, ddot, cdotu and zdotu.

    See Also
    --------
    dotc1d : dot product conjugating first vector.
    innerwt : weighted (i.e. triple) inner product.

    Examples
    --------
    >>> a = np.arange(1,5).reshape(2,2)
    >>> b = np.arange(1,8,2).reshape(2,2)
    >>> res = inner1d(a,b)
    >>> res.shape
    (2,)
    >>> print res
    [  7.  43.]

    """
    return _impl.inner1d(a, b, **kwargs)


def dotc1d(a, b, **kwargs):
    """
    Compute the dot product of vectors over the inner dimension, conjugating
    the first vector, with broadcasting

    Parameters
    ----------
    a : (..., N) array
        Input array
    b : (..., N) array
        Input array

    Returns
    -------
    dotc : (...) array
        dot product conjugating the first vector over the inner
        dimension.

    Notes
    -----
    Numpy broadcasting rules apply when matching dimensions.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    For single and double types this is equivalent to inner1d.

    Maps to Blas functions sdot, ddot, cdotc and zdotc.

    See Also
    --------
    inner1d : dot product
    innerwt : weighted (i.e. triple) inner product.

    Examples
    --------
    >>> a = np.arange(1,5).reshape(2,2)
    >>> b = np.arange(1,8,2).reshape(2,2)
    >>> res = inner1d(a,b)
    >>> res.shape
    (2,)
    >>> print res
    [  7.  43.]

    """
    return _impl.dotc1d(a, b, **kwargs)


def innerwt(a, b, c, **kwargs):
    """
    Compute the weighted (i.e. triple) inner product, with
    broadcasting.

    Parameters
    ----------
    a, b, c : (..., N) array
        Input arrays

    Returns
    -------
    inner : (...) array
        The weighted (i.e. triple) inner product.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    inner1d : inner product.
    dotc1d : dot product conjugating first vector.

    Examples
    --------
    >>> a = np.arange(1,5).reshape(2,2)
    >>> b = np.arange(1,8,2).reshape(2,2)
    >>> c = np.arange(0.25,1.20,0.25).reshape(2,2)
    >>> res = innerwt(a,b,c)
    >>> res.shape
    (2,)
    >>> res
    array([  3.25,  39.25])

    """
    return _impl.innerwt(a, b, c, **kwargs)


def matrix_multiply(a,b,**kwargs):
    """
    Compute matrix multiplication, with broadcasting

    Parameters
    ----------
    a : (..., M, N) array
        Input array.
    b : (..., N, P) array
        Input array.

    Returns
    -------
    r : (..., M, P) array matrix multiplication of a and b over any number of
        outer dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    Matrix multiplication is computed using BLAS _gemm functions.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.arange(1,17).reshape(2,2,4)
    >>> b = np.arange(1,25).reshape(2,4,3)
    >>> res = matrix_multiply(a,b)
    >>> res.shape
    (2, 2, 3)
    >>> res
    array([[[   70.,    80.,    90.],
            [  158.,   184.,   210.]],
    <BLANKLINE>
           [[  750.,   792.,   834.],
            [ 1030.,  1088.,  1146.]]])

    """
    return _impl.matrix_multiply(a,b,**kwargs)


def det(a, **kwargs):
    """
    Compute the determinant of arrays, with broadcasting.

    Parameters
    ----------
    a : (NDIMS, M, M) array
        Input array. Its inner dimensions must be those of a square 2-D array.

    Returns
    -------
    det : (NDIMS) array
        Determinants of `a`

    See Also
    --------
    slogdet : Another representation for the determinant, more suitable
        for large matrices where underflow/overflow may occur

    Notes
    -----
    Numpy broadcasting rules apply.

    The determinants are computed via LU factorization using the LAPACK
    routine _getrf.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.allclose(-2.0, det(a))
    True

    >>> a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]] ])
    >>> np.allclose(-2.0, det(a))
    True

    """
    return _impl.det(a, **kwargs)


def slogdet(a, **kwargs):
    """
    Compute the sign and (natural) logarithm of the determinant of an array,
    with broadcasting.

    If an array has a very small or very large determinant, then a call to 
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself

    Parameters
    ----------
    a : (..., M, M) array
        Input array. Its inner dimensions must be those of a square 2-D array.

    Returns
    -------
    sign : (...) array
        An array of numbers representing the sign of the determinants. For real
        matrices, this is 1, 0, or -1. For complex matrices, this is a complex 
        number with absolute value 1 (i.e., it is on the unit circle), or else
        0.
    logdet : (...) array
        The natural log of the absolute value of the determinant. This is always
        a real type.

    If the determinant is zero, then `sign` will be 0 and `logdet` will be -Inf.
    In all cases, the determinant is equal to ``sign * np.exp(logdet)``.

    See Also
    --------
    det

    Notes
    -----
    Numpy broadcasting rules apply.

    The determinants are computed via LU factorization using the LAPACK
    routine _getrf.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply. For complex versions `logdet` will be of the associated real
    type (single for csingle, double for cdouble).

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logdet) = slogdet(a)
    >>> sign.shape
    ()
    >>> logdet.shape
    ()
    >>> np.allclose(-2.0, sign * np.exp(logdet))
    True

    >>> a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]] ])
    >>> (sign, logdet) = slogdet(a)
    >>> sign.shape
    (2,)
    >>> logdet.shape
    (2,)
    >>> np.allclose(-2.0, sign * np.exp(logdet))
    True

    """
    return _impl.slogdet(a, **kwargs)


def inv(a, **kwargs):
    """
    Compute the (multiplicative) inverse of matrices, with broadcasting.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``matrix_multiply(a, ainv) = matrix_multiply(ainv, a) = Identity matrix``

    Parameters
    ----------
    a : (..., M, M) array
        Matrices to be inverted

    Returns
    -------
    ainv : (..., M, M) array
        (Multiplicative) inverse of the `a` matrices.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Singular matrices and thus, not invertible, result in an array of NaNs.

    See Also
    --------
    poinv : compute the multiplicative inverse of hermitian/symmetric matrices,
            using cholesky decomposition.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> ainv = inv(a)
    >>> np.allclose(matrix_multiply(a, ainv), np.eye(2))
    True
    >>> np.allclose(matrix_multiply(ainv, a), np.eye(2))
    True

    """
    return _impl.inv(a, **kwargs)


def cholesky(a, UPLO='L', **kwargs):
    """
    Compute the cholesky decomposition of `a`, with broadcasting

    The Cholesky decomposition (or Cholesky triangle) is a decomposition of a
    Hermitian, positive-definite matrix into the product of a lower triangular
    matrix and its conjugate transpose.

    A = LL*

    where L* is the positive-definite matrix.

    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which compute the cholesky decomposition

    Returns
    -------
    l : (..., M, M) array
        Matrices for each element where each entry is the lower triangular
        matrix with strictly positive diagonal entries such that a = ll* for
        all outer dimensions

    See Also
    --------
    chosolve : solve a system using cholesky decomposition
    poinv : compute the inverse of a matrix using cholesky decomposition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Decomposition is performed using LAPACK routine _potrf.

    For elements where the LAPACK routine fails, the result will be set to NaNs.

    If an element of the source array is not a positive-definite matrix the
    result for that element is undefined.

    Examples
    --------
    >>> A = np.array([[1,-2j],[2j,5]])
    >>> A
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = cholesky(A)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])

    """
    if 'L' == UPLO:
        gufunc = _impl.cholesky_lo
    else:
        gufunc = _impl.cholesky_up

    return gufunc(a, **kwargs)


def eig(a, **kwargs):
    """
    Compute the eigenvalues and right eigenvectors of square arrays,
    with broadcasting

    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed

    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be always be of complex type. When `a` is real
        the resulting eigenvalues will be real (0 imaginary part) or
        occur in conjugate pairs

    v : (..., M, M) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.

    See Also
    --------
    eigvals : eigenvalues of general arrays.
    eigh : eigenvalues and right eigenvectors of symmetric/hermitian
        arrays.
    eigvalsh : eigenvalues of symmetric/hermitian arrays.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    Eigenvalues and eigenvectors for single and double versions will
    always be typed csingle and cdouble, even if all the results are
    real (imaginary part will be 0).

    This is implemented using the _geev LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Examples
    --------
    First, a utility function to check if eigvals/eigvectors are correct.
    This checks the definition of eigenvectors. For each eigenvector v
    with associated eigenvalue w of a matrix M the following equality must
    hold: Mv == wv

    >>> def check_eigen(M, w, v):
    ...     '''vectorial check of Mv==wv'''
    ...     lhs = matrix_multiply(M, v)
    ...     rhs = w*v
    ...     return np.allclose(lhs, rhs)

    (Almost) Trivial example with real e-values and e-vectors. Note
    the complex types of the results

    >>> M = np.diag((1,2,3)).astype(float)
    >>> w, v = eig(M)
    >>> check_eigen(M, w, v)
    True

    Real matrix possessing complex e-values and e-vectors; note that the
    e-values are complex conjugates of each other.

    >>> M = np.array([[1, -1], [1, 1]])
    >>> w, v = eig(M)
    >>> check_eigen(M, w, v)
    True

    Complex-valued matrix with real e-values (but complex-valued e-vectors);
    note that a.conj().T = a, i.e., a is Hermitian.

    >>> M = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = eig(M)
    >>> check_eigen(M, w, v)
    True

    """
    return _impl.eig(a, **kwargs)


def eigvals(a, **kwargs):
    """
    Compute the eigenvalues of general matrices, with broadcasting.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : (..., M, M) array
        Matrices whose eigenvalues will be computed

    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be always be of complex type. When `a` is real
        the resulting eigenvalues will be real (0 imaginary part) or
        occur in conjugate pairs

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays.
    eigh : eigenvalues and right eigenvectors of symmetric/hermitian
        arrays.
    eigvalsh : eigenvalues of symmetric/hermitian arrays.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    Eigenvalues for single and double versions will always be typed
    csingle and cdouble, even if all the results are real (imaginary
    part will be 0).

    This is implemented using the _geev LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Examples
    --------

    Eigenvalues for a diagonal matrix are its diagonal elements

    >>> D = np.diag((-1,1))
    >>> eigvals(D)
    array([-1.+0.j,  1.+0.j])

    Multiplying on the left by an orthogonal matrix, `Q`, and on the
    right by `Q.T` (the transpose of `Q` preserves the eigenvalues of
    the original matrix

    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> A = matrix_multiply(Q, D)
    >>> A = matrix_multiply(A, Q.T)
    >>> eigvals(A)
    array([-1.+0.j,  1.+0.j])

    """
    return _impl.eigvals(a, **kwargs)


def quadratic_form(u,Q,v, **kwargs):
    """
    Compute the quadratic form uQv, with broadcasting

    Parameters
    ----------
    u : (..., M) array
        The u vectors of the quadratic form uQv

    Q : (..., M, N) array
        The Q matrices of the quadratic form uQv

    v : (..., N) array
        The v vectors of the quadratic form uQv

    Returns
    -------
    qf : (...) array
        The result of the quadratic forms

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    This is similar to PDL inner2

    Examples
    --------

    The result in absence of broadcasting is just as np.dot(np.dot(u,Q),v)
    or np.dot(u, np.dot(Q,v))

    >>> u = np.array([2., 3.])
    >>> Q = np.array([[1.,1.], [0.,1.]])
    >>> v = np.array([1.,2.])
    >>> quadratic_form(u,Q,v)
    12.0

    >>> np.dot(np.dot(u,Q),v)
    12.0

    >>> np.dot(u, np.dot(Q,v))
    12.0

    """
    return _impl.quadratic_form(u, Q, v, **kwargs)


def add3(a, b, c, **kwargs):
    """
    Element-wise addition of 3 arrays: a + b + c.

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the addends

    Returns
    -------
    add3 : (...) array
        resulting element-wise addition.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 30.0, 30)
    >>> add3(a[0::3], a[1::3], a[2::3])
    array([  6.,  15.,  24.,  33.,  42.,  51.,  60.,  69.,  78.,  87.])

    """
    return _impl.add3(a, b, c, **kwargs)


def multiply3(a, b, c, **kwargs):
    """
    Element-wise multiplication of 3 arrays: a*b*c.

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the factors

    Returns
    -------
    m3 : (...) array
        resulting element-wise product

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-was addition
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply3(a, 1.01, a)
    array([   1.01,    4.04,    9.09,   16.16,   25.25,   36.36,   49.49,
             64.64,   81.81,  101.  ])

    """
    return _impl.multiply3(a, b, c, **kwargs)


def multiply3_add(a, b, c, d, **kwargs):
    """
    Element-wise multiplication of 3 arrays adding an element
    of the a 4th array to the result: a*b*c + d

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the factors

    d : (...) array
        array with the addend

    Returns
    -------
    m3a : (...) array
        resulting element-wise addition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-was addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply3_add(a, 1.01, a, 42e-4)
    array([   1.0142,    4.0442,    9.0942,   16.1642,   25.2542,   36.3642,
             49.4942,   64.6442,   81.8142,  101.0042])

    """
    return _impl.multiply3_add(a, b, c, d, **kwargs)


def multiply_add(a, b, c, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the addends

    Returns
    -------
    add3 : (...) array
        resulting element-wise addition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-was addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply_add(a, a, 42e-4)
    array([   1.0042,    4.0042,    9.0042,   16.0042,   25.0042,   36.0042,
             49.0042,   64.0042,   81.0042,  100.0042])

    """
    return _impl.multiply_add(a, b, c, **kwargs)


def multiply_add2(a, b, c, d, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the addends

    Returns
    -------
    add3 : (...) array
        resulting element-wise addition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-was addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply_add2(a, a, a, 42e-4)
    array([   2.0042,    6.0042,   12.0042,   20.0042,   30.0042,   42.0042,
             56.0042,   72.0042,   90.0042,  110.0042])

    """
    return _impl.multiply_add2(a, b, c, d, **kwargs)


def multiply4(a, b, c, d, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the addends

    Returns
    -------
    add3 : (...) array
        resulting element-wise addition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-was addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply4(a, a, a[::-1], 1.0001)
    array([  10.001 ,   36.0036,   72.0072,  112.0112,  150.015 ,  180.018 ,
            196.0196,  192.0192,  162.0162,  100.01  ])

    """
    return _impl.multiply4(a, b, c, d, **kwargs)


def multiply4_add(a, b, c, d, e, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the addends

    Returns
    -------
    add3 : (...) array
        resulting element-wise addition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-was addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply4_add(a, a, a[::-1], 1.01, 42e-4)
    array([  10.1042,   36.3642,   72.7242,  113.1242,  151.5042,  181.8042,
            197.9642,  193.9242,  163.6242,  101.0042])

    """
    return _impl.multiply4_add(a, b, c, d, e,**kwargs)


def eigh(A, UPLO='L', **kw_args):
    """
    Computes the eigenvalues and eigenvectors for the square matrices
    in the inner dimensions of A, being those matrices
    symmetric/hermitian.

    Parameters
    ----------
    A : (..., M, M) array
         Hermitian/Symmetric matrices whose eigenvalues and
         eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    w : (..., M) array
        The eigenvalues, not necessarily ordered.
    v : (..., M, M) array
        The inner dimensions contain matrices with the normalized
        eigenvectors as columns. The column-numbers are coherent with
        the associated eigenvalue.

    Notes
    -----
    Numpy broadcasting rules apply.

    The eigenvalues/eigenvectors are computed using LAPACK routines _ssyevd,
    _heevd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Unlike eig, the results for single and double will always be single
    and doubles. It is not possible for symmetrical real matrices to result
    in complex eigenvalues/eigenvectors

    See Also
    --------
    eigvalsh : eigenvalues of symmetric/hermitian arrays.
    eig : eigenvalues and right eigenvectors for general matrices.
    eigvals : eigenvalues for general matrices.

    Examples
    --------
    First, a utility function to check if eigvals/eigvectors are correct.
    This checks the definition of eigenvectors. For each eigenvector v
    with associated eigenvalue w of a matrix M the following equality must
    hold: Mv == wv

    >>> def check_eigen(M, w, v):
    ...     '''vectorial check of Mv==wv'''
    ...     lhs = matrix_multiply(M, v)
    ...     rhs = w*v
    ...     return np.allclose(lhs, rhs)

    A simple example that computes eigenvectors and eigenvalues of
    a hermitian matrix and checks that A*v = v*w for both pairs of
    eignvalues(w) and eigenvectors(v)

    >>> M = np.array([[1, -2j], [2j, 1]])
    >>> w, v = eigh(M)
    >>> check_eigen(M, w, v)
    True

    """
    if 'L' == UPLO:
        gufunc = _impl.eigh_lo
    else:
        gufunc = _impl.eigh_up

    return gufunc(A, **kw_args)


def eigvalsh(A, UPLO='L', **kw_args):
    """
    Computes the eigenvalues for the square matrices in the inner
    dimensions of A, being those matrices symmetric/hermitian.

    Parameters
    ----------
    A : (..., M, M) array
         Hermitian/Symmetric matrices whose eigenvalues and
         eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    w : (..., M) array
        The eigenvalues, not necessarily ordered.

    Notes
    -----
    Numpy broadcasting rules apply.

    The eigenvalues are computed using LAPACK routines _ssyevd, _heevd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Unlike eigvals, the results for single and double will always be single
    and doubles. It is not possible for symmetrical real matrices to result
    in complex eigenvalues.

    See Also
    --------
    eigh : eigenvalues of symmetric/hermitian arrays.
    eig : eigenvalues and right eigenvectors for general matrices.
    eigvals : eigenvalues for general matrices.

    Examples
    --------
    eigvalsh results should be the same as the eigenvalues returned by eigh

    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> w, v = eigh(a)
    >>> np.allclose(w, eigvalsh(a))
    True

    eigvalsh on an identity matrix is all ones
    >>> eigvalsh(np.eye(6))
    array([ 1.,  1.,  1.,  1.,  1.,  1.])

    """
    if ('L' == UPLO):
        gufunc = _impl.eigvalsh_lo
    else:
        gufunc = _impl.eigvalsh_up

    return gufunc(A,**kw_args)


def solve(A,B,**kw_args):
    """
    Solve the linear matrix equations on the inner dimensions.

    Computes the "exact" solution, `x`. of the well-determined,
    i.e., full rank, linear matrix equations `ax = b`.

    Parameters
    ----------
    A : (..., M, M) array
        Coefficient matrices.
    B : (..., M, N) array
        Ordinate or "dependent variable" values.

    Returns
    -------
    X : (..., M, N) array
        Solutions to the system A X = B for all the outer dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    The solutions are computed using LAPACK routine _gesv

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    chosolve : solve a system using cholesky decomposition (for equations
               having symmetric/hermitian coefficient matrices)

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True

    """
    if len(B.shape) == (len(A.shape) - 1):
        gufunc = _impl.solve1
    else:
        gufunc = _impl.solve

    return gufunc(A,B,**kw_args)


def svd(a, full_matrices=1, compute_uv=1 ,**kw_args):
    """
    Singular Value Decomposition on the inner dimensions.

    Factors the matrices in `a` as ``u * np.diag(s) * v``, where `u`
    and `v` are unitary and `s` is a 1-d array of `a`'s singular
    values.

    Parameters
    ----------
    a : (..., M, N) array
        The array of matrices to decompose.
    full_matrices : bool, optional
        If True (default), `u` and `v` have the shapes (`M`, `M`) and
        (`N`, `N`), respectively. Otherwise, the shapes are (`M`, `K`)
        and (`K`, `N`), respectively, where `K` = min(`M`, `N`).
    compute_uv : bool, optional
        Whether or not to compute `u` and `v` in addition to `s`. True
        by default.

    Returns
    -------
    u : { (..., M, M), (..., M, K) } array
        Unitary matrices. The actual shape depends on the value of
        ``full_matrices``. Only returned when ``compute_uv`` is True.
    s : (..., K) array
        The singular values for every matrix, sorted in descending order.
    v : { (..., N, N), (..., K, N) } array
        Unitary matrices. The actual shape depends on the value of
        ``full_matrices``. Only returned when ``compute_uv`` is True.

    Notes
    -----
    Numpy broadcasting rules apply.

    Singular Value Decomposition is performed using LAPACK routine _gesdd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)

    Reconstruction based on full SVD:

    >>> U, s, V = svd(a, full_matrices=True)
    >>> U.shape, V.shape, s.shape
    ((9, 9), (6, 6), (6,))
    >>> S = np.zeros((9, 6), dtype=complex)
    >>> S[:6, :6] = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    True

    Reconstruction based on reduced SVD:

    >>> U, s, V = svd(a, full_matrices=False)
    >>> U.shape, V.shape, s.shape
    ((9, 6), (6, 6), (6,))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    True

    """
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
    """
    Solve the linear matrix equations on the inner dimensions, using
    cholesky decomposition.

    Computes the "exact" solution, `x`. of the well-determined,
    i.e., full rank, linear matrix equations `ax = b`, where a is
    a symmetric/hermitian positive-definite matrix.

    Parameters
    ----------
    A : (..., M, M) array
        Coefficient symmetric/hermitian positive-definite matrices.
    B : (..., M, N) array
        Ordinate or "dependent variable" values.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    X : (..., M, N) array
        Solutions to the system A X = B for all elements in the outer
        dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    The solutions are computed using LAPACK routines _potrf, _potrs

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    solve : solve a system using cholesky decomposition (for equations
            having symmetric/hermitian coefficient matrices)

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:
    (note the matrix is symmetric in this system)

    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True

    """
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
    """
    Compute the (multiplicative) inverse of symmetric/hermitian positive 
    definite matrices, with broadcasting.

    Given a square symmetic/hermitian positive-definite matrix `a`, return 
    the matrix `ainv` satisfying ``matrix_multiply(a, ainv) = 
    matrix_multiply(ainv, a) = Identity matrix``.

    Parameters
    ----------
    a : (..., M, M) array
        Symmetric/hermitian postive definite matrices to be inverted.

    Returns
    -------
    ainv : (..., M, M) array
        (Multiplicative) inverse of the `a` matrices.

    Notes
    -----
    Numpy broadcasting rules apply.

    The inverse is computed using LAPACK routines _potrf, _potri

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    inv : compute the multiplicative inverse of general matrices.

    Examples
    --------
    >>> a = np.array([[5, 3], [3, 5]])
    >>> ainv = poinv(a)
    >>> np.allclose(matrix_multiply(a, ainv), np.eye(2))
    True
    >>> np.allclose(matrix_multiply(ainv, a), np.eye(2))
    True

    """
    if 'L' == UPLO:
        gufunc = _impl.poinv_lo
    else:
        gufunc = _impl.poinv_up

    return gufunc(A, **kw_args);


if __name__ == "__main__":
    import doctest
    doctest.testmod()
