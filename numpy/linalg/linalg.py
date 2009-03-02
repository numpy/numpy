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
           'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'det', 'svd',
           'eig', 'eigh','lstsq', 'norm', 'qr', 'cond', 'LinAlgError']

from numpy.core import array, asarray, zeros, empty, transpose, \
        intc, single, double, csingle, cdouble, inexact, complexfloating, \
        newaxis, ravel, all, Inf, dot, add, multiply, identity, sqrt, \
        maximum, flatnonzero, diagonal, arange, fastCopyAndTranspose, sum, \
        isfinite, size
from numpy.lib import triu
from numpy.linalg import lapack_lite
from numpy.core.defmatrix import matrix_power

fortran_int = intc

# Error object
class LinAlgError(Exception):
    pass

def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
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
    Solve the tensor equation a x = b for x

    It is assumed that all indices of x are summed over in the product,
    together with the rightmost indices of a, similarly as in
    tensordot(a, x, axes=len(b.shape)).

    Parameters
    ----------
    a : array_like, shape b.shape+Q
        Coefficient tensor. Shape Q of the rightmost indices of a must
        be such that a is 'square', ie., prod(Q) == prod(b.shape).
    b : array_like, any shape
        Right-hand tensor.
    axes : tuple of integers
        Axes in a to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : array, shape Q

    Examples
    --------
    >>> a = np.eye(2*3*4)
    >>> a.shape = (2*3,4,  2,3,4)
    >>> b = np.random.randn(2*3,4)
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
    Solve the equation ``a x = b`` for ``x``.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Input equation coefficients.
    b : array_like, shape (M,)
        Equation target values.

    Returns
    -------
    x : array, shape (M,)

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

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
    Find the 'inverse' of a N-d array

    The result is an inverse corresponding to the operation
    tensordot(a, b, ind), ie.,

        x == tensordot(tensordot(tensorinv(a), a, ind), x, ind)
          == tensordot(tensordot(a, tensorinv(a), ind), x, ind)

    for all x (up to floating-point accuracy).

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must 'square', ie.,
        prod(a.shape[:ind]) == prod(a.shape[ind:])
    ind : integer > 0
        How many of the first indices are involved in the inverse sum.

    Returns
    -------
    b : array, shape a.shape[:ind]+a.shape[ind:]

    Raises LinAlgError if a is singular or not square

    Examples
    --------
    >>> a = np.eye(4*6)
    >>> a.shape = (4,6,8,3)
    >>> ainv = np.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)
    >>> b = np.random.randn(4,6)
    >>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))
    True

    >>> a = np.eye(4*6)
    >>> a.shape = (24,8,3)
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
    Compute the inverse of a matrix.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Matrix to be inverted

    Returns
    -------
    ainv : ndarray, shape (M, M)
        Inverse of the matrix `a`

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

    Examples
    --------
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> np.linalg.inv(a)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.dot(a, np.linalg.inv(a))
    array([[ 1.,  0.],
           [ 0.,  1.]])

    """
    a, wrap = _makearray(a)
    return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))


# Cholesky decomposition

def cholesky(a):
    """
    Cholesky decomposition.

    Return the Cholesky decomposition, :math:`A = L L^*` of a Hermitian
    positive-definite matrix :math:`A`.

    Parameters
    ----------
    a : array_like, shape (M, M)
        Hermitian (symmetric, if it is real) and positive definite
        input matrix.

    Returns
    -------
    L : array_like, shape (M, M)
        Lower-triangular Cholesky factor of A.

    Raises
    ------
    LinAlgError
       If the decomposition fails.

    Notes
    -----
    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \\mathbf{x} = \\mathbf{b}.

    First, we solve for :math:`\\mathbf{y}` in

    .. math:: L \\mathbf{y} = \\mathbf{b},

    and then for :math:`\\mathbf{x}` in

    .. math:: L^* \\mathbf{x} = \\mathbf{y}.

    Examples
    --------
    >>> A = np.array([[1,-2j],[2j,5]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> np.dot(L, L.T.conj())
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
    m = a.shape[0]
    n = a.shape[1]
    if isComplexType(t):
        lapack_routine = lapack_lite.zpotrf
    else:
        lapack_routine = lapack_lite.dpotrf
    results = lapack_routine('L', n, a, m, 0)
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
    Compute QR decomposition of a matrix.

    Calculate the decomposition :math:`A = Q R` where Q is orthonormal
    and R upper triangular.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be decomposed
    mode : {'full', 'r', 'economic'}
        Determines what information is to be returned. 'full' is the default.
        Economic mode is slightly faster if only R is needed.

    Returns
    -------
    mode = 'full'
    Q : double or complex array, shape (M, K)
    R : double or complex array, shape (K, N)
        Size K = min(M, N)

    mode = 'r'
    R : double or complex array, shape (K, N)

    mode = 'economic'
    A2 : double or complex array, shape (M, N)
        The diagonal and the upper triangle of A2 contains R,
        while the rest of the matrix is undefined.

    If a is a matrix, so are all the return values.

    Raises LinAlgError if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, and zungqr.

    Examples
    --------
    >>> a = np.random.randn(9, 6)
    >>> q, r = np.linalg.qr(a)
    >>> np.allclose(a, np.dot(q, r))
    True
    >>> r2 = np.linalg.qr(a, mode='r')
    >>> r3 = np.linalg.qr(a, mode='economic')
    >>> np.allclose(r, r2)
    True
    >>> np.allclose(r, np.triu(r3[:6,:6], k=0))
    True

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
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

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.

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
    eigvalsh : eigenvalues of symmetric or Hemitiean arrays.
    eigh : eigenvalues and eigenvectors of symmetric/Hermitean arrays.

    Notes
    -----
    This is a simple interface to the LAPACK routines dgeev and zgeev
    that sets the flags to return only the eigenvalues of general real
    and complex arrays respectively.

    The number w is an eigenvalue of a if there exists a vector v
    satisfying the equation dot(a,v) = w*v. Alternately, if w is a root of
    the characteristic equation det(a - w[i]*I) = 0, where det is the
    determinant and I is the identity matrix.

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    _assertFinite(a)
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    n = a.shape[0]
    dummy = zeros((1,), t)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgeev
        w = zeros((n,), t)
        rwork = zeros((n,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, w,
                                 dummy, 1, dummy, 1, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, w,
                                 dummy, 1, dummy, 1, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = zeros((n,), t)
        wi = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, wr, wi,
                                 dummy, 1, dummy, 1, work, -1, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, wr, wi,
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
    Compute the eigenvalues of a Hermitean or real symmetric matrix.

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with data from the
        lower triangular part of `a` ('L', default) or the upper triangular
        part ('U').

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigh : eigenvalues and eigenvectors of symmetric/Hermitean arrays.
    eigvals : eigenvalues of general real or complex arrays.
    eig : eigenvalues and eigenvectors of general real or complex arrays.

    Notes
    -----
    This is a simple interface to the LAPACK routines dsyevd and
    zheevd that sets the flags to return only the eigenvalues of real
    symmetric and complex Hermetian arrays respectively.

    The number w is an eigenvalue of a if there exists a vector v
    satisfying the equation dot(a,v) = w*v. Alternately, if w is a root of
    the characteristic equation det(a - w[i]*I) = 0, where det is the
    determinant and I is the identity matrix.

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
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
        results = lapack_routine('N', UPLO, n, a, n, w, work, -1,
                                 rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine('N', UPLO, n, a, n, w, work, lwork,
                                rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine('N', UPLO, n, a, n, w, work, -1,
                                 iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine('N', UPLO, n, a, n, w, work, lwork,
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
    Compute eigenvalues and right eigenvectors of an array.

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex or real 2-D array.

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered, nor are they
        necessarily real for real matrices.
    v : ndarray, shape (M, M)
        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
        the column ``v[:,i]``.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of symmetric or Hemitiean arrays.
    eig : eigenvalues and right eigenvectors for non-symmetric arrays
    eigvals : eigenvalues of non-symmetric array.

    Notes
    -----
    This is a simple interface to the LAPACK routines dgeev and zgeev
    that compute the eigenvalues and eigenvectors of general real and
    complex arrays respectively.

    The number `w` is an eigenvalue of a if there exists a vector `v`
    satisfying the equation ``dot(a,v) = w*v``. Alternately, if `w` is a root of
    the characteristic equation ``det(a - w[i]*I) = 0``, where `det` is the
    determinant and `I` is the identity matrix. The arrays `a`, `w`, and `v`
    satisfy the equation ``dot(a,v[i]) = w[i]*v[:,i]``.

    The array `v` of eigenvectors may not be of maximum rank, that is, some
    of the columns may be dependent, although roundoff error may obscure
    that fact. If the eigenvalues are all different, then theoretically the
    eigenvectors are independent. Likewise, the matrix of eigenvectors is
    unitary if the matrix `a` is normal, i.e., if ``dot(a, a.H) = dot(a.H, a)``.

    The left and right eigenvectors are not necessarily the (Hermitian)
    transposes of each other.

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    _assertFinite(a)
    a, t, result_t = _convertarray(a) # convert to double or cdouble type
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
        results = lapack_routine('N', 'V', n, a, n, w,
                                 dummy, 1, v, n, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'V', n, a, n, w,
                                 dummy, 1, v, n, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = zeros((n,), t)
        wi = zeros((n,), t)
        vr = zeros((n, n), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'V', n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, -1, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine('N', 'V', n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, lwork, 0)
        if all(wi == 0.0):
            w = wr
            v = vr
            result_t = _realType(result_t)
        else:
            w = wr+1j*wi
            v = array(vr, w.dtype)
            ind = flatnonzero(wi != 0.0)      # indices of complex e-vals
            for i in range(len(ind)/2):
                v[ind[2*i]] = vr[ind[2*i]] + 1j*vr[ind[2*i+1]]
                v[ind[2*i+1]] = vr[ind[2*i]] - 1j*vr[ind[2*i+1]]
            result_t = _complexType(result_t)

    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    vt = v.transpose().astype(result_t)
    return w.astype(result_t), wrap(vt)


def eigh(a, UPLO='L'):
    """
    Eigenvalues and eigenvectors of a Hermitian or real symmetric matrix.

    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex Hermitian or symmetric real matrix.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with data from the
        lower triangular part of `a` ('L', default) or the upper triangular
        part ('U').

    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues. The eigenvalues are not necessarily ordered.
    v : ndarray, shape (M, M)
        The normalized eigenvector corresponding to the eigenvalue w[i] is
        the column v[:,i].

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of symmetric or Hemitiean arrays.
    eig : eigenvalues and right eigenvectors for non-symmetric arrays
    eigvals : eigenvalues of non-symmetric array.

    Notes
    -----
    A simple interface to the LAPACK routines dsyevd and zheevd that compute
    the eigenvalues and eigenvectors of real symmetric and complex Hermitian
    arrays respectively.

    The number w is an eigenvalue of a if there exists a vector v
    satisfying the equation dot(a,v) = w*v. Alternately, if w is a root of
    the characteristic equation det(a - w[i]*I) = 0, where det is the
    determinant and I is the identity matrix. The eigenvalues of real
    symmetric or complex Hermitean matrices are always real. The array v
    of eigenvectors is unitary and a, w, and v satisfy the equation
    dot(a,v[i]) = w[i]*v[:,i].

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
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
        results = lapack_routine('V', UPLO, n, a, n, w, work, -1,
                                 rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine('V', UPLO, n, a, n, w, work, lwork,
                                 rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine('V', UPLO, n, a, n, w, work, -1,
                iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine('V', UPLO, n, a, n, w, work, lwork,
                iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    at = a.transpose().astype(result_t)
    return w.astype(_realType(result_t)), wrap(at)


# Singular value decomposition

def svd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices, ``U`` and ``Vh``,
    and a 1-dimensional array of singular values, ``s`` (real, non-negative),
    such that ``a == U S Vh``, where ``S`` is the diagonal
    matrix ``np.diag(s)``.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to decompose
    full_matrices : boolean, optional
        If True (default), ``U`` and ``Vh`` are shaped
        ``(M,M)`` and ``(N,N)``.  Otherwise, the shapes are
        ``(M,K)`` and ``(K,N)``, where ``K = min(M,N)``.
    compute_uv : boolean
        Whether to compute ``U`` and ``Vh`` in addition to ``s``.
        True by default.

    Returns
    -------
    U : ndarray, shape (M, M) or (M, K) depending on `full_matrices`
        Unitary matrix.
    s :  ndarray, shape (K,) where ``K = min(M, N)``
        The singular values, sorted so that ``s[i] >= s[i+1]``.
    Vh : ndarray, shape (N,N) or (K,N) depending on `full_matrices`
        Unitary matrix.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    If `a` is a matrix (in contrast to an ndarray), then so are all
    the return values.

    Examples
    --------
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> U, s, Vh = np.linalg.svd(a)
    >>> U.shape, Vh.shape, s.shape
    ((9, 9), (6, 6), (6,))

    >>> U, s, Vh = np.linalg.svd(a, full_matrices=False)
    >>> U.shape, Vh.shape, s.shape
    ((9, 6), (6, 6), (6,))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = np.linalg.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertNonEmpty(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    s = zeros((min(n, m),), real_t)
    if compute_uv:
        if full_matrices:
            nu = m
            nvt = n
            option = 'A'
        else:
            nu = min(n, m)
            nvt = min(n, m)
            option = 'S'
        u = zeros((nu, m), t)
        vt = zeros((n, nvt), t)
    else:
        option = 'N'
        nu = 1
        nvt = 1
        u = empty((1, 1), t)
        vt = empty((1, 1), t)

    iwork = zeros((8*min(m, n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgesdd
        rwork = zeros((5*min(m, n)*min(m, n) + 5*min(m, n),), real_t)
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

    The condition number of `x` is the norm of `x` times the norm
    of the inverse of `x`.  The norm can be the usual L2
    (root-of-sum-of-squares) norm or a number of other matrix norms.

    Parameters
    ----------
    x : array_like, shape (M, N)
        The matrix whose condition number is sought.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}
        Order of the norm:

        =====  ============================
        p      norm for matrices
        =====  ============================
        None   2-norm, computed directly using the SVD
        'fro'  Frobenius norm
        inf    max(sum(abs(x), axis=1))
        -inf   min(sum(abs(x), axis=1))
        1      max(sum(abs(x), axis=0))
        -1     min(sum(abs(x), axis=0))
        2      2-norm (largest sing. value)
        -2     smallest singular value
        =====  ============================

    Returns
    -------
    c : float
        The condition number of the matrix. May be infinite.

    """
    x = asarray(x) # in case we have a matrix
    if p is None:
        s = svd(x,compute_uv=False)
        return s[0]/s[-1]
    else:
        return norm(x,p)*norm(inv(x),p)

# Generalized inverse

def pinv(a, rcond=1e-15 ):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    `large` singular values.

    Parameters
    ----------
    a : array_like (M, N)
      Matrix to be pseudo-inverted.
    rcond : float
      Cutoff for `small` singular values.
      Singular values smaller than rcond*largest_singular_value are
      considered zero.

    Returns
    -------
    B : ndarray (N, M)
      The pseudo-inverse of `a`. If `a` is an np.matrix instance, then so
      is `B`.


    Raises
    ------
    LinAlgError
      In case SVD computation does not converge.

    Examples
    --------
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

    """
    a = asarray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
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
        return 0.0
    sign = add.reduce(pivots != arange(1, n+1)) % 2
    return (1.-2.*sign)*multiply.reduce(diagonal(a), axis=-1)


# Linear Least Squares

def lstsq(a, b, rcond=-1):
    """
    Return the least-squares solution to an equation.

    Solves the equation `a x = b` by computing a vector `x` that minimizes
    the norm `|| b - a x ||`.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Input equation coefficients.
    b : array_like, shape (M,) or (M, K)
        Equation target values.  If `b` is two-dimensional, the least
        squares solution is calculated for each of the `K` target sets.
    rcond : float, optional
        Cutoff for ``small`` singular values of `a`.
        Singular values smaller than `rcond` times the largest singular
        value are  considered zero.

    Returns
    -------
    x : ndarray, shape(N,) or (N, K)
         Least squares solution.  The shape of `x` depends on the shape of
         `b`.
    residues : ndarray, shape(), (1,), or (K,)
        Sums of residues; squared Euclidian norm for each column in
        `b - a x`.
        If the rank of `a` is < N or > M, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : integer
        Rank of matrix `a`.
    s : ndarray, shape(min(M,N),)
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    Notes
    -----
    If `b` is a matrix, then all array results returned as
    matrices.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cuts the y-axis at more-or-less -1.

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
    real_t = _linalgRealType(t)
    bstar = zeros((ldb, n_rhs), t)
    bstar[:b.shape[0],:n_rhs] = b.copy()
    a, bstar = _fastCopyAndTranspose(t, a, bstar)
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
    resids = array([], t)
    if is_1d:
        x = array(ravel(bstar)[:n], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            resids = array([sum((ravel(bstar)[n:])**2)], dtype=result_t)
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            resids = sum((transpose(bstar)[n:,:])**2, axis=0).astype(result_t)
    st = s[:min(n, m)].copy().astype(_realType(result_t))
    return wrap(x), wrap(resids), results['rank'], st

def norm(x, ord=None):
    """
    Matrix or vector norm.

    Parameters
    ----------
    x : array_like, shape (M,) or (M, N)
        Input array.
    ord : {int, 1, -1, 2, -2, inf, -inf, 'fro'}
        Order of the norm (see table under ``Notes``).

    Returns
    -------
    n : float
        Norm of the matrix or vector

    Notes
    -----
    For values ord < 0, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    """
    x = asarray(x)
    nd = len(x.shape)
    if ord is None: # check the default case first and handle it immediately
        return sqrt(add.reduce((x.conj() * x).ravel().real))

    if nd == 1:
        if ord == Inf:
            return abs(x).max()
        elif ord == -Inf:
            return abs(x).min()
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
