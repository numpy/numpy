"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
contains high-level Python interface to the LAPACK library.  The lite
version only accesses the following LAPACK functions: dgesv, zgesv,
dgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,
zgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.
"""

__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv',
           'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det',
           'svd', 'eig', 'eigh','lstsq', 'norm', 'qr', 'cond', 'matrix_rank',
           'LinAlgError']

from numpy.core import array, asarray, zeros, empty, transpose, \
        intc, single, double, csingle, cdouble, inexact, complexfloating, \
        newaxis, ravel, all, Inf, dot, add, multiply, identity, sqrt, \
        maximum, flatnonzero, diagonal, arange, fastCopyAndTranspose, sum, \
        isfinite, size, finfo, absolute, log, exp
from numpy.lib import triu
from numpy.linalg import lapack_lite
from numpy.matrixlib.defmatrix import matrix_power
from numpy.compat import asbytes

# For Python2/3 compatibility
_N = asbytes('N')
_V = asbytes('V')
_A = asbytes('A')
_S = asbytes('S')
_L = asbytes('L')

fortran_int = intc

# Error object
class LinAlgError(Exception):
    """
    Generic Python-exception-derived object raised by linalg functions.

    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> LA.inv(np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "...linalg.py", line 350,
        in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
      File "...linalg.py", line 249,
        in solve
        raise LinAlgError, 'Singular matrix'
    numpy.linalg.linalg.LinAlgError: Singular matrix

    """
    pass

def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def isComplexType(t):
    return issubclass(t, complexfloating)

_real_types_map = {single : single,
                   double : double,
                   csingle : single,
                   cdouble : double}

_complex_types_map = {single : csingle,
                      double : cdouble,
                      csingle : csingle,
                      cdouble : cdouble}

def _realType(t, default=double):
    return _real_types_map.get(t, default)

def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)

def _linalgRealType(t):
    """Cast the type t to either double or cdouble."""
    return double

_complex_types_map = {single : csingle,
                      double : cdouble,
                      csingle : csingle,
                      cdouble : cdouble}

def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, inexact):
            if isComplexType(a.dtype.type):
                is_complex = True
            rt = _realType(a.dtype.type, default=None)
            if rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            rt = double
        if rt is double:
            result_type = double
    if is_complex:
        t = cdouble
        result_type = _complex_types_map[result_type]
    else:
        t = double
    return t, result_type

# _fastCopyAndTranpose assumes the input is 2D (as all the calls in here are).

_fastCT = fastCopyAndTranspose

def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ('=', '|'):
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret

def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.type is type:
            cast_arrays = cast_arrays + (_fastCT(a),)
        else:
            cast_arrays = cast_arrays + (_fastCT(a.astype(type)),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays

def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError, '%d-dimensional array given. Array must be \
            two-dimensional' % len(a.shape)

def _assertSquareness(*arrays):
    for a in arrays:
        if max(a.shape) != min(a.shape):
            raise LinAlgError, 'Array must be square'

def _assertFinite(*arrays):
    for a in arrays:
        if not (isfinite(a).all()):
            raise LinAlgError, "Array must not contain infs or NaNs"

def _assertNonEmpty(*arrays):
    for a in arrays:
        if size(a) == 0:
            raise LinAlgError("Arrays cannot be empty")


# Linear equations

def tensorsolve(a, b, axes=None):
    """
    Solve the tensor equation ``a x = b`` for x.

    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=len(b.shape))``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
       ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
       'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : ndarray, shape Q

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    tensordot, tensorinv, einsum

    Examples
    --------
    >>> a = np.eye(2*3*4)
    >>> a.shape = (2*3, 4, 2, 3, 4)
    >>> b = np.random.randn(2*3, 4)
    >>> x = np.linalg.tensorsolve(a, b)
    >>> x.shape
    (2, 3, 4)
    >>> np.allclose(np.tensordot(a, x, axes=3), b)
    True

    """
    a,wrap = _makearray(a)
    b = asarray(b)
    an = a.ndim

    if axes is not None:
        allaxes = range(0, an)
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)

    oldshape = a.shape[-(an-b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k

    a = a.reshape(-1, prod)
    b = b.ravel()
    res = wrap(solve(a, b))
    res.shape = oldshape
    return res

def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Coefficient matrix.
    b : array_like, shape (M,) or (M, N)
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : ndarray, shape (M,) or (M, N) depending on b
        Solution to the system a x = b

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

    Notes
    -----
    `solve` is a wrapper for the LAPACK routines `dgesv`_ and
    `zgesv`_, the former being used if `a` is real-valued, the latter if
    it is complex-valued.  The solution to the system of linear equations
    is computed using an LU decomposition [1]_ with partial pivoting and
    row interchanges.

    .. _dgesv: http://www.netlib.org/lapack/double/dgesv.f

    .. _zgesv: http://www.netlib.org/lapack/complex16/zgesv.f

    `a` must be square and of full-rank, i.e., all rows (or, equivalently,
    columns) must be linearly independent; if either is not true, use
    `lstsq` for the least-squares best "solution" of the
    system/equation.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 22.

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = np.linalg.solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> (np.dot(a, x) == b).all()
    True

    """
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    one_eq = len(b.shape) == 1
    if one_eq:
        b = b[:, newaxis]
    _assertRank2(a, b)
    _assertSquareness(a)
    n_eq = a.shape[0]
    n_rhs = b.shape[1]
    if n_eq != b.shape[0]:
        raise LinAlgError, 'Incompatible dimensions'
    t, result_t = _commonType(a, b)
#    lapack_routine = _findLapackRoutine('gesv', t)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgesv
    else:
        lapack_routine = lapack_lite.dgesv
    a, b = _fastCopyAndTranspose(t, a, b)
    a, b = _to_native_byte_order(a, b)
    pivots = zeros(n_eq, fortran_int)
    results = lapack_routine(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Singular matrix'
    if one_eq:
        return wrap(b.ravel().astype(result_t))
    else:
        return wrap(b.transpose().astype(result_t))


def tensorinv(a, ind=2):
    """
    Compute the 'inverse' of an N-dimensional array.

    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[:ind] + a.shape[ind:]``.

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    tensordot, tensorsolve

    Examples
    --------
    >>> a = np.eye(4*6)
    >>> a.shape = (4, 6, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)
    >>> b = np.random.randn(4, 6)
    >>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))
    True

    >>> a = np.eye(4*6)
    >>> a.shape = (24, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=1)
    >>> ainv.shape
    (8, 3, 24)
    >>> b = np.random.randn(24)
    >>> np.allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))
    True

    """
    a = asarray(a)
    oldshape = a.shape
    prod = 1
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        for k in oldshape[ind:]:
            prod *= k
    else:
        raise ValueError, "Invalid ind argument."
    a = a.reshape(prod, -1)
    ia = inv(a)
    return ia.reshape(*invshape)


# Matrix inversion

def inv(a):
    """
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Matrix to be inverted.

    Returns
    -------
    ainv : ndarray or matrix, shape (M, M)
        (Multiplicative) inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = LA.inv(a)
    >>> np.allclose(np.dot(a, ainv), np.eye(2))
    True
    >>> np.allclose(np.dot(ainv, a), np.eye(2))
    True

    If a is a matrix object, then the return value is a matrix as well:

    >>> ainv = LA.inv(np.matrix(a))
    >>> ainv
    matrix([[-2. ,  1. ],
            [ 1.5, -0.5]])

    """
    a, wrap = _makearray(a)
    return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))


# Cholesky decomposition

def cholesky(a):
    """
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.H`, of the square matrix `a`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `a` is real-valued).  `a` must be
    Hermitian (symmetric if real-valued) and positive-definite.  Only `L` is
    actually returned.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.

    Returns
    -------
    L : ndarray, or matrix object if `a` is, shape (M, M)
        Lower-triangular Cholesky factor of a.

    Raises
    ------
    LinAlgError
       If the decomposition fails, for example, if `a` is not
       positive-definite.

    Notes
    -----
    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \\mathbf{x} = \\mathbf{b}

    (when `A` is both Hermitian/symmetric and positive-definite).

    First, we solve for :math:`\\mathbf{y}` in

    .. math:: L \\mathbf{y} = \\mathbf{b},

    and then for :math:`\\mathbf{x}` in

    .. math:: L.H \\mathbf{x} = \\mathbf{y}.

    Examples
    --------
    >>> A = np.array([[1,-2j],[2j,5]])
    >>> A
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> np.dot(L, L.T.conj()) # verify that L * L.H = A
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?
    >>> np.linalg.cholesky(A) # an ndarray object is returned
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> # But a matrix object is returned if A is a matrix object
    >>> LA.cholesky(np.matrix(A))
    matrix([[ 1.+0.j,  0.+0.j],
            [ 0.+2.j,  1.+0.j]])

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    m = a.shape[0]
    n = a.shape[1]
    if isComplexType(t):
        lapack_routine = lapack_lite.zpotrf
    else:
        lapack_routine = lapack_lite.dpotrf
    results = lapack_routine(_L, n, a, m, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Matrix is not positive definite - \
        Cholesky decomposition cannot be computed'
    s = triu(a, k=0).transpose()
    if (s.dtype != result_t):
        s = s.astype(result_t)
    return wrap(s)

# QR decompostion

def qr(a, mode='full'):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    Parameters
    ----------
    a : array_like
        Matrix to be factored, of shape (M, N).
    mode : {'full', 'r', 'economic'}, optional
        Specifies the values to be returned. 'full' is the default.
        Economic mode is slightly faster then 'r' mode if only `r` is needed.

    Returns
    -------
    q : ndarray of float or complex, optional
        The orthonormal matrix, of shape (M, K). Only returned if
        ``mode='full'``.
    r : ndarray of float or complex, optional
        The upper-triangular matrix, of shape (K, N) with K = min(M, N).
        Only returned when ``mode='full'`` or ``mode='r'``.
    a2 : ndarray of float or complex, optional
        Array of shape (M, N), only returned when ``mode='economic``'.
        The  diagonal and the upper triangle of `a2` contains `r`, while
        the rest of the matrix is undefined.

    Raises
    ------
    LinAlgError
        If factoring fails.

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, and zungqr.

    For more information on the qr factorization, see for example:
    http://en.wikipedia.org/wiki/QR_factorization

    Subclasses of `ndarray` are preserved, so if `a` is of type `matrix`,
    all the return values will be matrices too.

    Examples
    --------
    >>> a = np.random.randn(9, 6)
    >>> q, r = np.linalg.qr(a)
    >>> np.allclose(a, np.dot(q, r))  # a does equal qr
    True
    >>> r2 = np.linalg.qr(a, mode='r')
    >>> r3 = np.linalg.qr(a, mode='economic')
    >>> np.allclose(r, r2)  # mode='r' returns the same r as mode='full'
    True
    >>> # But only triu parts are guaranteed equal when mode='economic'
    >>> np.allclose(r, np.triu(r3[:6,:6], k=0))
    True

    Example illustrating a common use of `qr`: solving of least squares
    problems

    What are the least-squares-best `m` and `y0` in ``y = y0 + mx`` for
    the following data: {(0,1), (1,0), (1,2), (2,1)}. (Graph the points
    and you'll see that it should be y0 = 0, m = 1.)  The answer is provided
    by solving the over-determined matrix equation ``Ax = b``, where::

      A = array([[0, 1], [1, 1], [1, 1], [2, 1]])
      x = array([[y0], [m]])
      b = array([[1], [0], [2], [1]])

    If A = qr such that q is orthonormal (which is always possible via
    Gram-Schmidt), then ``x = inv(r) * (q.T) * b``.  (In numpy practice,
    however, we simply use `lstsq`.)

    >>> A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])
    >>> A
    array([[0, 1],
           [1, 1],
           [1, 1],
           [2, 1]])
    >>> b = np.array([1, 0, 2, 1])
    >>> q, r = LA.qr(A)
    >>> p = np.dot(q.T, b)
    >>> np.dot(LA.inv(r), p)
    array([  1.1e-16,   1.0e+00])

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    mn = min(m, n)
    tau = zeros((mn,), t)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgeqrf
        routine_name = 'zgeqrf'
    else:
        lapack_routine = lapack_lite.dgeqrf
        routine_name = 'dgeqrf'

    # calculate optimal size of work data 'work'
    lwork = 1
    work = zeros((lwork,), t)
    results = lapack_routine(m, n, a, m, tau, work, -1, 0)
    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    # do qr decomposition
    lwork = int(abs(work[0]))
    work = zeros((lwork,), t)
    results = lapack_routine(m, n, a, m, tau, work, lwork, 0)

    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    #  economic mode. Isn't actually economic.
    if mode[0] == 'e':
        if t != result_t :
            a = a.astype(result_t)
        return a.T

    #  generate r
    r = _fastCopyAndTranspose(result_t, a[:,:mn])
    for i in range(mn):
        r[i,:i].fill(0.0)

    #  'r'-mode, that is, calculate only r
    if mode[0] == 'r':
        return r

    #  from here on: build orthonormal matrix q from a

    if isComplexType(t):
        lapack_routine = lapack_lite.zungqr
        routine_name = 'zungqr'
    else:
        lapack_routine = lapack_lite.dorgqr
        routine_name = 'dorgqr'

    # determine optimal lwork
    lwork = 1
    work = zeros((lwork,), t)
    results = lapack_routine(m, mn, mn, a, m, tau, work, -1, 0)
    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    # compute q
    lwork = int(abs(work[0]))
    work = zeros((lwork,), t)
    results = lapack_routine(m, mn, mn, a, m, tau, work, lwork, 0)
    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    q = _fastCopyAndTranspose(result_t, a[:mn,:])

    return wrap(q), wrap(r)


# Eigenvalues


def eigvals(a):
    """
    Compute the eigenvalues of a general matrix.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex- or real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered, nor are they necessarily
        real for real matrices.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays
    eigvalsh : eigenvalues of symmetric or Hermitian arrays.
    eigh : eigenvalues and eigenvectors of symmetric/Hermitian arrays.

    Notes
    -----
    This is a simple interface to the LAPACK routines dgeev and zgeev
    that sets those routines' flags to return only the eigenvalues of
    general real and complex arrays, respectively.

    Examples
    --------
    Illustration, using the fact that the eigenvalues of a diagonal matrix
    are its diagonal elements, that multiplying a matrix on the left
    by an orthogonal matrix, `Q`, and on the right by `Q.T` (the transpose
    of `Q`), preserves the eigenvalues of the "middle" matrix.  In other words,
    if `Q` is orthogonal, then ``Q * A * Q.T`` has the same eigenvalues as
    ``A``:

    >>> from numpy import linalg as LA
    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])
    (1.0, 1.0, 0.0)

    Now multiply a diagonal matrix by Q on one side and by Q.T on the other:

    >>> D = np.diag((-1,1))
    >>> LA.eigvals(D)
    array([-1.,  1.])
    >>> A = np.dot(Q, D)
    >>> A = np.dot(A, Q.T)
    >>> LA.eigvals(A)
    array([ 1., -1.])

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    _assertFinite(a)
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    n = a.shape[0]
    dummy = zeros((1,), t)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgeev
        w = zeros((n,), t)
        rwork = zeros((n,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _N, n, a, n, w,
                                 dummy, 1, dummy, 1, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _N, n, a, n, w,
                                 dummy, 1, dummy, 1, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = zeros((n,), t)
        wi = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _N, n, a, n, wr, wi,
                                 dummy, 1, dummy, 1, work, -1, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _N, n, a, n, wr, wi,
                                 dummy, 1, dummy, 1, work, lwork, 0)
        if all(wi == 0.):
            w = wr
            result_t = _realType(result_t)
        else:
            w = wr+1j*wi
            result_t = _complexType(result_t)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w.astype(result_t)


def eigvalsh(a, UPLO='L'):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix.

    Main difference from eigh: the eigenvectors are not computed.

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex- or real-valued matrix whose eigenvalues are to be
        computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, not necessarily ordered, each repeated according to
        its multiplicity.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigh : eigenvalues and eigenvectors of symmetric/Hermitian arrays.
    eigvals : eigenvalues of general real or complex arrays.
    eig : eigenvalues and right eigenvectors of general real or complex
          arrays.

    Notes
    -----
    This is a simple interface to the LAPACK routines dsyevd and zheevd
    that sets those routines' flags to return only the eigenvalues of
    real symmetric and complex Hermitian arrays, respectively.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> LA.eigvalsh(a)
    array([ 0.17157288+0.j,  5.82842712+0.j])

    """
    UPLO = asbytes(UPLO)
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    n = a.shape[0]
    liwork = 5*n+3
    iwork = zeros((liwork,), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zheevd
        w = zeros((n,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        lrwork = 1
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(_N, UPLO, n, a, n, w, work, -1,
                                 rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(_N, UPLO, n, a, n, w, work, lwork,
                                rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(_N, UPLO, n, a, n, w, work, -1,
                                 iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(_N, UPLO, n, a, n, w, work, lwork,
                                 iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w.astype(result_t)

def _convertarray(a):
    t, result_t = _commonType(a)
    a = _fastCT(a.astype(t))
    return a, t, result_t


# Eigenvectors


def eig(a):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : array_like, shape (M, M)
        A square array of real or complex elements.

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered, nor are they
        necessarily real for real arrays (though for real arrays
        complex-valued eigenvalues should occur in conjugate pairs).

    v : ndarray, shape (M, M)
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of a symmetric or Hermitian (conjugate symmetric)
       array.

    eigvals : eigenvalues of a non-symmetric array.

    Notes
    -----
    This is a simple interface to the LAPACK routines dgeev and zgeev
    which compute the eigenvalues and eigenvectors of, respectively,
    general real- and complex-valued square arrays.

    The number `w` is an eigenvalue of `a` if there exists a vector
    `v` such that ``dot(a,v) = w * v``. Thus, the arrays `a`, `w`, and
    `v` satisfy the equations ``dot(a[i,:], v[i]) = w[i] * v[:,i]``
    for :math:`i \\in \\{0,...,M-1\\}`.

    The array `v` of eigenvectors may not be of maximum rank, that is, some
    of the columns may be linearly dependent, although round-off error may
    obscure that fact. If the eigenvalues are all different, then theoretically
    the eigenvectors are linearly independent. Likewise, the (complex-valued)
    matrix of eigenvectors `v` is unitary if the matrix `a` is normal, i.e.,
    if ``dot(a, a.H) = dot(a.H, a)``, where `a.H` denotes the conjugate
    transpose of `a`.

    Finally, it is emphasized that `v` consists of the *right* (as in
    right-hand side) eigenvectors of `a`.  A vector `y` satisfying
    ``dot(y.T, a) = z * y.T`` for some number `z` is called a *left*
    eigenvector of `a`, and, in general, the left and right eigenvectors
    of a matrix are not necessarily the (perhaps conjugate) transposes
    of each other.

    References
    ----------
    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,
    Academic Press, Inc., 1980, Various pp.

    Examples
    --------
    >>> from numpy import linalg as LA

    (Almost) trivial example with real e-values and e-vectors.

    >>> w, v = LA.eig(np.diag((1, 2, 3)))
    >>> w; v
    array([ 1.,  2.,  3.])
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    Real matrix possessing complex e-values and e-vectors; note that the
    e-values are complex conjugates of each other.

    >>> w, v = LA.eig(np.array([[1, -1], [1, 1]]))
    >>> w; v
    array([ 1. + 1.j,  1. - 1.j])
    array([[ 0.70710678+0.j        ,  0.70710678+0.j        ],
           [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]])

    Complex-valued matrix with real e-values (but complex-valued e-vectors);
    note that a.conj().T = a, i.e., a is Hermitian.

    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = LA.eig(a)
    >>> w; v
    array([  2.00000000e+00+0.j,   5.98651912e-36+0.j]) # i.e., {2, 0}
    array([[ 0.00000000+0.70710678j,  0.70710678+0.j        ],
           [ 0.70710678+0.j        ,  0.00000000+0.70710678j]])

    Be careful about round-off error!

    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. e-values are 1 +/- 1e-9
    >>> w, v = LA.eig(a)
    >>> w; v
    array([ 1.,  1.])
    array([[ 1.,  0.],
           [ 0.,  1.]])

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    _assertFinite(a)
    a, t, result_t = _convertarray(a) # convert to double or cdouble type
    a = _to_native_byte_order(a)
    real_t = _linalgRealType(t)
    n = a.shape[0]
    dummy = zeros((1,), t)
    if isComplexType(t):
        # Complex routines take different arguments
        lapack_routine = lapack_lite.zgeev
        w = zeros((n,), t)
        v = zeros((n, n), t)
        lwork = 1
        work = zeros((lwork,), t)
        rwork = zeros((2*n,), real_t)
        results = lapack_routine(_N, _V, n, a, n, w,
                                 dummy, 1, v, n, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _V, n, a, n, w,
                                 dummy, 1, v, n, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = zeros((n,), t)
        wi = zeros((n,), t)
        vr = zeros((n, n), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _V, n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, -1, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(_N, _V, n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, lwork, 0)
        if all(wi == 0.0):
            w = wr
            v = vr
            result_t = _realType(result_t)
        else:
            w = wr+1j*wi
            v = array(vr, w.dtype)
            ind = flatnonzero(wi != 0.0)      # indices of complex e-vals
            for i in range(len(ind)//2):
                v[ind[2*i]] = vr[ind[2*i]] + 1j*vr[ind[2*i+1]]
                v[ind[2*i+1]] = vr[ind[2*i]] - 1j*vr[ind[2*i+1]]
            result_t = _complexType(result_t)

    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    vt = v.transpose().astype(result_t)
    return w.astype(result_t), wrap(vt)


def eigh(a, UPLO='L'):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex Hermitian or real symmetric matrix.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, not necessarily ordered.
    v : ndarray, or matrix object if `a` is, shape (M, M)
        The column ``v[:, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[i]``.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of symmetric or Hermitian arrays.
    eig : eigenvalues and right eigenvectors for non-symmetric arrays.
    eigvals : eigenvalues of non-symmetric arrays.

    Notes
    -----
    This is a simple interface to the LAPACK routines dsyevd and zheevd,
    which compute the eigenvalues and eigenvectors of real symmetric and
    complex Hermitian arrays, respectively.

    The eigenvalues of real symmetric or complex Hermitian matrices are
    always real. [1]_ The array `v` of (column) eigenvectors is unitary
    and `a`, `w`, and `v` satisfy the equations
    ``dot(a, v[:, i]) = w[i] * v[:, i]``.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 222.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> a
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> w, v = LA.eigh(a)
    >>> w; v
    array([ 0.17157288,  5.82842712])
    array([[-0.92387953+0.j        , -0.38268343+0.j        ],
           [ 0.00000000+0.38268343j,  0.00000000-0.92387953j]])

    >>> np.dot(a, v[:, 0]) - w[0] * v[:, 0] # verify 1st e-val/vec pair
    array([2.77555756e-17 + 0.j, 0. + 1.38777878e-16j])
    >>> np.dot(a, v[:, 1]) - w[1] * v[:, 1] # verify 2nd e-val/vec pair
    array([ 0.+0.j,  0.+0.j])

    >>> A = np.matrix(a) # what happens if input is a matrix object
    >>> A
    matrix([[ 1.+0.j,  0.-2.j],
            [ 0.+2.j,  5.+0.j]])
    >>> w, v = LA.eigh(A)
    >>> w; v
    array([ 0.17157288,  5.82842712])
    matrix([[-0.92387953+0.j        , -0.38268343+0.j        ],
            [ 0.00000000+0.38268343j,  0.00000000-0.92387953j]])

    """
    UPLO = asbytes(UPLO)
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    n = a.shape[0]
    liwork = 5*n+3
    iwork = zeros((liwork,), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zheevd
        w = zeros((n,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        lrwork = 1
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(_V, UPLO, n, a, n, w, work, -1,
                                 rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(_V, UPLO, n, a, n, w, work, lwork,
                                 rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(_V, UPLO, n, a, n, w, work, -1,
                iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(_V, UPLO, n, a, n, w, work, lwork,
                iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    at = a.transpose().astype(result_t)
    return w.astype(_realType(result_t)), wrap(at)


# Singular value decomposition

def svd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factors the matrix `a` as ``u * np.diag(s) * v``, where `u` and `v`
    are unitary and `s` is a 1-d array of `a`'s singular values.

    Parameters
    ----------
    a : array_like
        A real or complex matrix of shape (`M`, `N`) .
    full_matrices : bool, optional
        If True (default), `u` and `v` have the shapes (`M`, `M`) and
        (`N`, `N`), respectively.  Otherwise, the shapes are (`M`, `K`)
        and (`K`, `N`), respectively, where `K` = min(`M`, `N`).
    compute_uv : bool, optional
        Whether or not to compute `u` and `v` in addition to `s`.  True
        by default.

    Returns
    -------
    u : ndarray
        Unitary matrix.  The shape of `u` is (`M`, `M`) or (`M`, `K`)
        depending on value of ``full_matrices``.
    s : ndarray
        The singular values, sorted so that ``s[i] >= s[i+1]``.  `s` is
        a 1-d array of length min(`M`, `N`).
    v : ndarray
        Unitary matrix of shape (`N`, `N`) or (`K`, `N`), depending on
        ``full_matrices``.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    The SVD is commonly written as ``a = U S V.H``.  The `v` returned
    by this function is ``V.H`` and ``u = U``.

    If ``U`` is a unitary matrix, it means that it
    satisfies ``U.H = inv(U)``.

    The rows of `v` are the eigenvectors of ``a.H a``. The columns
    of `u` are the eigenvectors of ``a a.H``.  For row ``i`` in
    `v` and column ``i`` in `u`, the corresponding eigenvalue is
    ``s[i]**2``.

    If `a` is a `matrix` object (as opposed to an `ndarray`), then so
    are all the return values.

    Examples
    --------
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)

    Reconstruction based on full SVD:

    >>> U, s, V = np.linalg.svd(a, full_matrices=True)
    >>> U.shape, V.shape, s.shape
    ((9, 6), (6, 6), (6,))
    >>> S = np.zeros((9, 6), dtype=complex)
    >>> S[:6, :6] = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    True

    Reconstruction based on reduced SVD:

    >>> U, s, V = np.linalg.svd(a, full_matrices=False)
    >>> U.shape, V.shape, s.shape
    ((9, 6), (6, 6), (6,))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    True

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertNonEmpty(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    s = zeros((min(n, m),), real_t)
    if compute_uv:
        if full_matrices:
            nu = m
            nvt = n
            option = _A
        else:
            nu = min(n, m)
            nvt = min(n, m)
            option = _S
        u = zeros((nu, m), t)
        vt = zeros((n, nvt), t)
    else:
        option = _N
        nu = 1
        nvt = 1
        u = empty((1, 1), t)
        vt = empty((1, 1), t)

    iwork = zeros((8*min(m, n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgesdd
        lrwork = min(m,n)*max(5*min(m,n)+7, 2*max(m,n)+2*min(m,n)+1)
        rwork = zeros((lrwork,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, lwork, rwork, iwork, 0)
    else:
        lapack_routine = lapack_lite.dgesdd
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, -1, iwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge'
    s = s.astype(_realType(result_t))
    if compute_uv:
        u = u.transpose().astype(result_t)
        vt = vt.transpose().astype(result_t)
        return wrap(u), s, wrap(vt)
    else:
        return s

def cond(x, p=None):
    """
    Compute the condition number of a matrix.

    This function is capable of returning the condition number using
    one of seven different norms, depending on the value of `p` (see
    Parameters below).

    Parameters
    ----------
    x : array_like, shape (M, N)
        The matrix whose condition number is sought.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional
        Order of the norm:

        =====  ============================
        p      norm for matrices
        =====  ============================
        None   2-norm, computed directly using the ``SVD``
        'fro'  Frobenius norm
        inf    max(sum(abs(x), axis=1))
        -inf   min(sum(abs(x), axis=1))
        1      max(sum(abs(x), axis=0))
        -1     min(sum(abs(x), axis=0))
        2      2-norm (largest sing. value)
        -2     smallest singular value
        =====  ============================

        inf means the numpy.inf object, and the Frobenius norm is
        the root-of-sum-of-squares norm.

    Returns
    -------
    c : {float, inf}
        The condition number of the matrix. May be infinite.

    See Also
    --------
    numpy.linalg.linalg.norm

    Notes
    -----
    The condition number of `x` is defined as the norm of `x` times the
    norm of the inverse of `x` [1]_; the norm can be the usual L2-norm
    (root-of-sum-of-squares) or one of a number of other matrix norms.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, Orlando, FL,
           Academic Press, Inc., 1980, pg. 285.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> a
    array([[ 1,  0, -1],
           [ 0,  1,  0],
           [ 1,  0,  1]])
    >>> LA.cond(a)
    1.4142135623730951
    >>> LA.cond(a, 'fro')
    3.1622776601683795
    >>> LA.cond(a, np.inf)
    2.0
    >>> LA.cond(a, -np.inf)
    1.0
    >>> LA.cond(a, 1)
    2.0
    >>> LA.cond(a, -1)
    1.0
    >>> LA.cond(a, 2)
    1.4142135623730951
    >>> LA.cond(a, -2)
    0.70710678118654746
    >>> min(LA.svd(a, compute_uv=0))*min(LA.svd(LA.inv(a), compute_uv=0))
    0.70710678118654746

    """
    x = asarray(x) # in case we have a matrix
    if p is None:
        s = svd(x,compute_uv=False)
        return s[0]/s[-1]
    else:
        return norm(x,p)*norm(inv(x),p)


def matrix_rank(M, tol=None):
    """
    Return matrix rank of array using SVD method

    Rank of the array is the number of SVD singular values of the
    array that are greater than `tol`.

    Parameters
    ----------
    M : array_like
        array of <=2 dimensions
    tol : {None, float}
       threshold below which SVD values are considered zero. If `tol` is
       None, and ``S`` is an array with singular values for `M`, and
       ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
       set to ``S.max() * eps``.

    Notes
    -----
    Golub and van Loan [1]_ define "numerical rank deficiency" as using
    tol=eps*S[0] (where S[0] is the maximum singular value and thus the
    2-norm of the matrix). This is one definition of rank deficiency,
    and the one we use here.  When floating point roundoff is the main
    concern, then "numerical rank deficiency" is a reasonable choice. In
    some cases you may prefer other definitions. The most useful measure
    of the tolerance depends on the operations you intend to use on your
    matrix. For example, if your data come from uncertain measurements
    with uncertainties greater than floating point epsilon, choosing a
    tolerance near that uncertainty may be preferable.  The tolerance
    may be absolute if the uncertainties are absolute rather than
    relative.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*.
       Baltimore: Johns Hopkins University Press, 1996.

    Examples
    --------
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0

    """
    M = asarray(M)
    if M.ndim > 2:
        raise TypeError('array should have 2 or fewer dimensions')
    if M.ndim < 2:
        return int(not all(M==0))
    S = svd(M, compute_uv=False)
    if tol is None:
        tol = S.max() * finfo(S.dtype).eps
    return sum(S > tol)


# Generalized inverse

def pinv(a, rcond=1e-15 ):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Parameters
    ----------
    a : array_like, shape (M, N)
      Matrix to be pseudo-inverted.
    rcond : float
      Cutoff for small singular values.
      Singular values smaller (in modulus) than
      `rcond` * largest_singular_value (again, in modulus)
      are set to zero.

    Returns
    -------
    B : ndarray, shape (N, M)
      The pseudo-inverse of `a`. If `a` is a `matrix` instance, then so
      is `B`.

    Raises
    ------
    LinAlgError
      If the SVD computation does not converge.

    Notes
    -----
    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
    value decomposition of A, then
    :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
    orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
    of A's so-called singular values, (followed, typically, by
    zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
    consisting of the reciprocals of A's singular values
    (again, followed by zeros). [1]_

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pp. 139-142.

    Examples
    --------
    The following example checks that ``a * a+ * a == a`` and
    ``a+ * a * a+ == a+``:

    >>> a = np.random.randn(9, 6)
    >>> B = np.linalg.pinv(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    """
    a, wrap = _makearray(a)
    _assertNonEmpty(a)
    a = a.conjugate()
    u, s, vt = svd(a, 0)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond*maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.;
    res = dot(transpose(vt), multiply(s[:, newaxis],transpose(u)))
    return wrap(res)

# Determinant

def slogdet(a):
    """
    Compute the sign and (natural) logarithm of the determinant of an array.

    If an array has a very small or very large determinant, than a call to
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself.

    Parameters
    ----------
    a : array_like
        Input array, has to be a square 2-D array.

    Returns
    -------
    sign : float or complex
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1. For a complex matrix, this is a complex number
        with absolute value 1 (i.e., it is on the unit circle), or else 0.
    logdet : float
        The natural log of the absolute value of the determinant.

    If the determinant is zero, then `sign` will be 0 and `logdet` will be
    -Inf. In all cases, the determinant is equal to ``sign * np.exp(logdet)``.

    See Also
    --------
    det

    Notes
    -----
    The determinant is computed via LU factorization using the LAPACK
    routine z/dgetrf.

    .. versionadded:: 1.6.0.

    Examples
    --------
    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logdet) = np.linalg.slogdet(a)
    >>> (sign, logdet)
    (-1, 0.69314718055994529)
    >>> sign * np.exp(logdet)
    -2.0

    This routine succeeds where ordinary `det` does not:

    >>> np.linalg.det(np.eye(500) * 0.1)
    0.0
    >>> np.linalg.slogdet(np.eye(500) * 0.1)
    (1, -1151.2925464970228)

    """
    a = asarray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    n = a.shape[0]
    if isComplexType(t):
        lapack_routine = lapack_lite.zgetrf
    else:
        lapack_routine = lapack_lite.dgetrf
    pivots = zeros((n,), fortran_int)
    results = lapack_routine(n, n, a, n, pivots, 0)
    info = results['info']
    if (info < 0):
        raise TypeError, "Illegal input to Fortran routine"
    elif (info > 0):
        return (t(0.0), _realType(t)(-Inf))
    sign = 1. - 2. * (add.reduce(pivots != arange(1, n + 1)) % 2)
    d = diagonal(a)
    absd = absolute(d)
    sign *= multiply.reduce(d / absd)
    log(absd, absd)
    logdet = add.reduce(absd, axis=-1)
    return sign, logdet

def det(a):
    """
    Compute the determinant of an array.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Input array.

    Returns
    -------
    det : ndarray
        Determinant of `a`.

    Notes
    -----
    The determinant is computed via LU factorization using the LAPACK
    routine z/dgetrf.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.linalg.det(a)
    -2.0

    See Also
    --------
    slogdet : Another way to representing the determinant, more suitable
      for large matrices where underflow/overflow may occur.

    """
    sign, logdet = slogdet(a)
    return sign * exp(logdet)

# Linear Least Squares

def lstsq(a, b, rcond=-1):
    """
    Return the least-squares solution to a linear matrix equation.

    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.

    Parameters
    ----------
    a : array_like, shape (M, N)
        "Coefficient" matrix.
    b : array_like, shape (M,) or (M, K)
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        Singular values are set to zero if they are smaller than `rcond`
        times the largest singular value of `a`.

    Returns
    -------
    x : ndarray, shape (N,) or (N, K)
        Least-squares solution.  The shape of `x` depends on the shape of
        `b`.
    residues : ndarray, shape (), (1,), or (K,)
        Sums of residues; squared Euclidean 2-norm for each column in
        ``b - a*x``.
        If the rank of `a` is < N or > M, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : ndarray, shape (min(M,N),)
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.

    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:

    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])

    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> print m, c
    1.0 -0.95

    Plot the data along with the fitted line:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> plt.legend()
    >>> plt.show()

    """
    import math
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = len(b.shape) == 1
    if is_1d:
        b = b[:, newaxis]
    _assertRank2(a, b)
    m  = a.shape[0]
    n  = a.shape[1]
    n_rhs = b.shape[1]
    ldb = max(n, m)
    if m != b.shape[0]:
        raise LinAlgError, 'Incompatible dimensions'
    t, result_t = _commonType(a, b)
    result_real_t = _realType(result_t)
    real_t = _linalgRealType(t)
    bstar = zeros((ldb, n_rhs), t)
    bstar[:b.shape[0],:n_rhs] = b.copy()
    a, bstar = _fastCopyAndTranspose(t, a, bstar)
    a, bstar = _to_native_byte_order(a, bstar)
    s = zeros((min(m, n),), real_t)
    nlvl = max( 0, int( math.log( float(min(m, n))/2. ) ) + 1 )
    iwork = zeros((3*min(m, n)*nlvl+11*min(m, n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = zeros((lwork,), real_t)
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        rwork = zeros((lwork,), real_t)
        a_real = zeros((m, n), real_t)
        bstar_real = zeros((ldb, n_rhs,), real_t)
        results = lapack_lite.dgelsd(m, n, n_rhs, a_real, m,
                                     bstar_real, ldb, s, rcond,
                                     0, rwork, -1, iwork, 0)
        lrwork = int(rwork[0])
        work = zeros((lwork,), t)
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, rwork, iwork, 0)
    else:
        lapack_routine = lapack_lite.dgelsd
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, iwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge in Linear Least Squares'
    resids = array([], result_real_t)
    if is_1d:
        x = array(ravel(bstar)[:n], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            if isComplexType(t):
                resids = array([sum(abs(ravel(bstar)[n:])**2)],
                               dtype=result_real_t)
            else:
                resids = array([sum((ravel(bstar)[n:])**2)],
                               dtype=result_real_t)
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            if isComplexType(t):
                resids = sum(abs(transpose(bstar)[n:,:])**2, axis=0).astype(
                    result_real_t)
            else:
                resids = sum((transpose(bstar)[n:,:])**2, axis=0).astype(
                    result_real_t)

    st = s[:min(n, m)].copy().astype(result_real_t)
    return wrap(x), wrap(resids), results['rank'], st

def norm(x, ord=None):
    """
    Matrix or vector norm.

    This function is able to return one of seven different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like, shape (M,) or (M, N)
        Input array.
    ord : {non-zero int, inf, -inf, 'fro'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.

    Returns
    -------
    n : float
        Norm of the matrix or vector.

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.norm(a)
    7.745966692414834
    >>> LA.norm(b)
    7.745966692414834
    >>> LA.norm(b, 'fro')
    7.745966692414834
    >>> LA.norm(a, np.inf)
    4
    >>> LA.norm(b, np.inf)
    9
    >>> LA.norm(a, -np.inf)
    0
    >>> LA.norm(b, -np.inf)
    2

    >>> LA.norm(a, 1)
    20
    >>> LA.norm(b, 1)
    7
    >>> LA.norm(a, -1)
    -4.6566128774142013e-010
    >>> LA.norm(b, -1)
    6
    >>> LA.norm(a, 2)
    7.745966692414834
    >>> LA.norm(b, 2)
    7.3484692283495345

    >>> LA.norm(a, -2)
    nan
    >>> LA.norm(b, -2)
    1.8570331885190563e-016
    >>> LA.norm(a, 3)
    5.8480354764257312
    >>> LA.norm(a, -3)
    nan

    """
    x = asarray(x)
    if ord is None: # check the default case first and handle it immediately
        return sqrt(add.reduce((x.conj() * x).ravel().real))

    nd = x.ndim
    if nd == 1:
        if ord == Inf:
            return abs(x).max()
        elif ord == -Inf:
            return abs(x).min()
        elif ord == 0:
            return (x != 0).sum() # Zero norm
        elif ord == 1:
            return abs(x).sum() # special case for speedup
        elif ord == 2:
            return sqrt(((x.conj()*x).real).sum()) # special case for speedup
        else:
            try:
                ord + 1
            except TypeError:
                raise ValueError, "Invalid norm order for vectors."
            return ((abs(x)**ord).sum())**(1.0/ord)
    elif nd == 2:
        if ord == 2:
            return svd(x, compute_uv=0).max()
        elif ord == -2:
            return svd(x, compute_uv=0).min()
        elif ord == 1:
            return abs(x).sum(axis=0).max()
        elif ord == Inf:
            return abs(x).sum(axis=1).max()
        elif ord == -1:
            return abs(x).sum(axis=0).min()
        elif ord == -Inf:
            return abs(x).sum(axis=1).min()
        elif ord in ['fro','f']:
            return sqrt(add.reduce((x.conj() * x).real.ravel()))
        else:
            raise ValueError, "Invalid norm order for matrices."
    else:
        raise ValueError, "Improper number of dimensions to norm."
