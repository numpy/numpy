"""Linear Algebra functions implemented as gufuncs, so they can be broadcast.

Notes
-----
This module contains functionality that could be found in the linalg module,
but in a gufunc-like way. This allows the use of vectorization and broadcasting
on the operands.

This module itself provides wrappers to kernels written as numpy
generalized-ufuncs that perform the heavy-lifting of computation. The wrappers
exist in order to provide a sane interface, like providing keyword arguments in
line with the ones used by linalg, or just to automatically select the
appropriate kernel depending on the parameters. All wrappers forward the keyword
parameters to the underlying generalized ufunc (the kernel).

The functions are intended to be used on arrays of functions. For those
functions where a result may not be possible to obtain (like the inverse of
a matrix that is not invertible) no exception is raised, but the results for
the elements involved are set to NaN.

Additional functions some fused arithmetic, useful for efficient operation over
"""

__all__ = ['inner1d', 'innerwt', 'matrix_multiply', 'det', 'slogdet', 'inv',
           'cholesky', 'quadratic_form', 'add3', 'multiply3', 'multiply3_add',
           'multiply_add', 'multiply_add2', 'multiply4', 'multiply4_add', 'eig',
           'eigvals', 'eigh', 'eigvalsh', 'solve', 'svd', 'chosolve', 'poinv']


import numpy.core._umath_linalg as _impl
import numpy as np

def inner1d(a, b, **kwargs):
    """
    Compute the inner product, with broadcasting

    Parameters
    ----------
    a : (<NDIMS>, N) array
        Input array
    b : (<NDIMS>, N) array
        Input array

    Returns
    -------
    inner : (<NDIM>) array
        Inner product over the inner dimension.

    Notes
    -----
    Numpy broadcasting rules apply when matching dimensions.

    Implemented for types single and double. Numpy conversion rules apply.

    See Also
    --------
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


def innerwt(a, b, c, **kwargs):
    """
    Compute the weighted (i.e. triple) inner product, with broadcasting.

    Parameters
    ----------
    a, b, c : (<NDIMS>, N) array
        Input arrays

    Returns
    -------
    inner : (<NDIMS>) array
        The weighted (i.e. triple) inner product.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single and double. Numpy conversion rules apply.

    See Also
    --------
    inner1d : inner product.

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
    a : (<NDIMS>, M, N) array
        Input array.
    b : (<NDIMS>, N, P) array
        Input array.

    Returns
    -------
    r : (<NDIMS>, M, P) array
        matrix multiplication of a and b over <NDIMS>.
        <NDIMS> can be any number of dimensions.

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
    a : (<NDIMS>, M, M) array
        Input array. Its inner dimensions must be those of a square 2-D array.

    Returns
    -------
    sign : (<NDIMS>) array
        An array of numbers representing the sign of the determinants. For real
        matrices, this is 1, 0, or -1. For complex matrices, this is a complex 
        number with absolute value 1 (i.e., it is on the unit circle), or else
        0.
    logdet : (<NDIMS>) array
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
    a : (<NDIMS>, M, M) array
        Matrices to be inverted

    Returns
    -------
    ainv : (<NDIMS>, M, M) array
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
    a : (<NDIMS>, M, M) array
        Matrices for which compute the cholesky decomposition

    Returns
    -------
    l : (<NDIMS>, M, M) array
        <NDIMS> matrices where each entry is the lower triangular matrix with
        strictly positive diagonal entries such that a = ll* for all <NDIMS>.

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
    a : (<NDIMS>, M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed

    Returns
    -------
    w : (<NDIMS>, M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be always be of complex type. When `a` is real
        the resulting eigenvalues will be real (0 imaginary part) or
        occur in conjugate pairs

    v : (<NDIMS>, M, M) array
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

    (Almost) Trivial example with real e-values and e-vectors. Note
    the complex types of the results

    >>> w, v = eig(np.diag((1,2,3)))
    >>> w; v
    array([ 1.+0.j,  2.+0.j,  3.+0.j])
    array([[ 1.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j]])

    Real matrix possessing complex e-values and e-vectors; note that the
    e-values are complex conjugates of each other.

    >>> w, v = eig(np.array([[1, -1], [1, 1]]))
    >>> w; v
    array([ 1. + 1.j,  1. - 1.j])
    array([[ 0.70710678+0.j        ,  0.70710678+0.j        ],
           [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]])

    Complex-valued matrix with real e-values (but complex-valued e-vectors);
    note that a.conj().T = a, i.e., a is Hermitian.

    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = eig(a)
    >>> w; v
    array([  2.00000000e+00+0.j,   5.98651912e-36+0.j]) # i.e., {2, 0}
    array([[ 0.00000000+0.70710678j,  0.70710678+0.j        ],
           [ 0.70710678+0.j        ,  0.00000000+0.70710678j]])

    """
    return _impl.eig(a, **kwargs)

def eigvals(a, **kwargs):
    """
    Compute the eigenvalues of general matrices, with broadcasting.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : (<NDIMS>, M, M) array
        Matrices whose eigenvalues will be computed

    Returns
    -------
    w : (<NDIMS>, M) array
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
    u : (<NDIMS>, M) array
        The u vectors of the quadratic form uQv

    Q : (<NDIMS>, M, N) array
        The Q matrices of the quadratic form uQv

    v : (<NDIMS>, N) array
        The v vectors of the quadratic form uQv

    Returns
    -------
    qf : (<NDIMS>) array
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
    a, b, c : (<NDIMS>) array
        arrays with the addends

    Returns
    -------
    add3 : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.add3(a, b, c, **kwargs)

def multiply3(a, b, c, **kwargs):
    """
    Element-wise multiplication of 3 arrays: a*b*c.

    Parameters
    ----------
    a, b, c : (<NDIMS>) array
        arrays with the factors

    Returns
    -------
    m3 : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.multiply3(a, b, c, **kwargs)

def multiply3_add(a, b, c, d, **kwargs):
    """
    Element-wise multiplication of 3 arrays adding an element
    of the a 4th array to the result: a*b*c + d

    Parameters
    ----------
    a, b, c : (<NDIMS>) array
        arrays with the factors

    d : (<NDIMS>) array
        array with the addend

    Returns
    -------
    m3a : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.multiply3_add(a, b, c, d, **kwargs)

def multiply_add(a, b, c, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (<NDIMS>) array
        arrays with the addends

    Returns
    -------
    add3 : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.multiply_add(a, b, c, **kwargs)

def multiply_add2(a, b, c, d, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (<NDIMS>) array
        arrays with the addends

    Returns
    -------
    add3 : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.multiply_add2(a, b, c, d, **kwargs)

def multiply4(a, b, c, d, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (<NDIMS>) array
        arrays with the addends

    Returns
    -------
    add3 : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.multiply4(a, b, c, d, **kwargs)

def multiply4_add(a, b, c, d, e, **kwargs):
    """
    Element-wise addition of 3 arrays

    Parameters
    ----------
    a, b, c : (<NDIMS>) array
        arrays with the addends

    Returns
    -------
    add3 : (<NDIMS>) array
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
    <Some example in doctest format>
    """
    return _impl.multiply4_add(a, b, c, d, e,**kwargs)

def eigh(A, UPLO='L', **kw_args):
    """
    Computes the eigenvalues and eigenvectors for the square matrices
    in the inner dimensions of A, being those matrices
    symmetric/hermitian.

    Parameters
    ----------
    A : (<NDIMS>, M, M) array
         Hermitian/Symmetric matrices whose eigenvalues and
         eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    w : (<NDIMS>, M) array
        The eigenvalues, not necessarily ordered.
    v : (<NDIMS>, M, M) array
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

    See Also
    --------
    eigvalsh : eigenvalues of symmetric/hermitian arrays.
    eig : eigenvalues and right eigenvectors for general matrices.
    eigvals : eigenvalues for general matrices.

    Examples
    --------
    <Some example in doctest format>
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
    A : (<NDIMS>, M, M) array
         Hermitian/Symmetric matrices whose eigenvalues and
         eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    w : (<NDIMS>, M) array
        The eigenvalues, not necessarily ordered.

    Notes
    -----
    Numpy broadcasting rules apply.

    The eigenvalues are computed using LAPACK routines _ssyevd, _heevd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    eigh : eigenvalues of symmetric/hermitian arrays.
    eig : eigenvalues and right eigenvectors for general matrices.
    eigvals : eigenvalues for general matrices.

    Examples
    --------
    <Some example in doctest format>
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
    A : (<NDIMS>, M, M) array
        Coefficient matrices.
    B : (<NDIMS>, M, N) array
        Ordinate or "dependent variable" values.

    Returns
    -------
    X : (<NDIMS>, M, N) array
        Solutions to the system A X = B for all elements in <NDIMS>

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
    <Some example in doctest format>
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
    a : (<NDIMS>, M, N) array
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
    u : { (<NDIMS>, M, M), (<NDIMS>, M, K) } array
        Unitary matrices. The actual shape depends on the value of
        ``full_matrices``. Only returned when ``compute_uv`` is True.
    s : (<NDIMS>, K) array
        The singular values for every matrix, sorted in descending order.
    v : { (<NDIMS>, N, N), (<NDIMS>, K, N) } array
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
    <Some example in doctest format>
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
    a symmetric/hermitian matrix.

    Parameters
    ----------
    A : (<NDIMS>, M, M) array
        Coefficient symmetric/hermitian matrices.
    B : (<NDIMS>, M, N) array
        Ordinate or "dependent variable" values.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    X : (<NDIMS>, M, N) array
        Solutions to the system A X = B for all elements in <NDIMS>

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
    <Some example in doctest format>
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
    Compute the (multiplicative) inverse of symmetric/hermitian matrices,
    with broadcasting.

    Given a square symmetic/hermitian matrix `a`, return the matrix `ainv`
    satisfying ``matrix_multiply(a, ainv) = matrix_multiply(ainv, a) =
    Identity matrix``-

    Parameters
    ----------
    a : (<NDIMS>, M, M) array
        Symmetric/hermitian matrices to be inverted

    Returns
    -------
    ainv : (<NDIMS>, M, M) array
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
    <Some example in doctest format>
    """
    if 'L' == UPLO:
        gufunc = _impl.poinv_lo
    else:
        gufunc = _impl.poinv_up

    return gufunc(A, **kw_args);


""" doc template (23 lines)"""
"""
    <Description>

    Parameters
    ----------
    <insert parameters + explanations>

    Returns
    -------
    <insert return values>

    Notes
    -----
    <insert any notes that may be interesting, optional>

    See Also
    --------
    <reference related functions>

    Examples
    --------
    <Some example in doctest format>
"""
