"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
contains high-level Python interface to the LAPACK library.  The lite
version only accesses the following LAPACK functions: dgesv, zgesv,
dgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,
zgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.
"""

__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv',
           'inv', 'cholesky',
           'eigvals',
           'eigvalsh', 'pinv',
           'det', 'svd',
           'eig', 'eigh','lstsq', 'norm',
           'qr',
           'cond',
           'LinAlgError'
           ]

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

def _castCopyAndTranspose(type, *arrays):
    if len(arrays) == 1:
        return transpose(arrays[0]).astype(type)
    else:
        return [transpose(a).astype(type) for a in arrays]

# _fastCopyAndTranpose is an optimized version of _castCopyAndTranspose.
# It assumes the input is 2D (as all the calls in here are).

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
    """Solve the tensor equation a x = b for x

    It is assumed that all indices of x are summed over in the product,
    together with the rightmost indices of a, similarly as in
    tensordot(a, x, axes=len(b.shape)).

    Parameters
    ----------
    a : array, shape b.shape+Q
        Coefficient tensor. Shape Q of the rightmost indices of a must
        be such that a is 'square', ie., prod(Q) == prod(b.shape).
    b : array, any shape
        Right-hand tensor.
    axes : tuple of integers
        Axes in a to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : array, shape Q

    Examples
    --------
    >>> from numpy import *
    >>> a = eye(2*3*4)
    >>> a.shape = (2*3,4,  2,3,4)
    >>> b = random.randn(2*3,4)
    >>> x = linalg.tensorsolve(a, b)
    >>> x.shape
    (2, 3, 4)
    >>> allclose(tensordot(a, x, axes=3), b)
    True

    """
    a = asarray(a)
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
    res = solve(a, b)
    res.shape = oldshape
    return res

def solve(a, b):
    """Solve the equation a x = b

    Parameters
    ----------
    a : array, shape (M, M)
    b : array, shape (M,)

    Returns
    -------
    x : array, shape (M,)

    Raises LinAlgError if a is singular or not square

    """
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
        return b.ravel().astype(result_t)
    else:
        return b.transpose().astype(result_t)


def tensorinv(a, ind=2):
    """Find the 'inverse' of a N-d array

    The result is an inverse corresponding to the operation
    tensordot(a, b, ind), ie.,

        x == tensordot(tensordot(tensorinv(a), a, ind), x, ind)
          == tensordot(tensordot(a, tensorinv(a), ind), x, ind)

    for all x (up to floating-point accuracy).

    Parameters
    ----------
    a : array
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
    >>> from numpy import *
    >>> a = eye(4*6)
    >>> a.shape = (4,6,8,3)
    >>> ainv = linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)
    >>> b = random.randn(4,6)
    >>> allclose(tensordot(ainv, b), linalg.tensorsolve(a, b))
    True

    >>> a = eye(4*6)
    >>> a.shape = (24,8,3)
    >>> ainv = linalg.tensorinv(a, ind=1)
    >>> ainv.shape
    (8, 3, 24)
    >>> b = random.randn(24)
    >>> allclose(tensordot(ainv, b, 1), linalg.tensorsolve(a, b))
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
    """Compute the inverse of a matrix.

    Parameters
    ----------
    a : array-like, shape (M, M)
        Matrix to be inverted

    Returns
    -------
    ainv : array-like, shape (M, M)
        Inverse of the matrix a

    Raises LinAlgError if a is singular or not square

    Examples
    --------
    >>> from numpy import array, inv, dot
    >>> a = array([[1., 2.], [3., 4.]])
    >>> inv(a)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> dot(a, inv(a))
    array([[ 1.,  0.],
           [ 0.,  1.]])

    """
    a, wrap = _makearray(a)
    return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))


# Cholesky decomposition

def cholesky(a):
    """Compute the Cholesky decomposition of a matrix.

    Returns the Cholesky decomposition, :lm:`A = L L^*` of a Hermitian
    positive-definite matrix :lm:`A`.

    Parameters
    ----------
    a : array, shape (M, M)
        Matrix to be decomposed

    Returns
    -------
    L : array, shape (M, M)
        Lower-triangular Cholesky factor of A

    Raises LinAlgError if decomposition fails

    Examples
    --------
    >>> from numpy import array, linalg
    >>> a = array([[1,-2j],[2j,5]])
    >>> L = linalg.cholesky(a)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> dot(L, L.T.conj())
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])

    """
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
        return s.astype(result_t)
    return s

# QR decompostion

def qr(a, mode='full'):
    """Compute QR decomposition of a matrix.

    Calculate the decomposition :lm:`A = Q R` where Q is orthonormal
    and R upper triangular.

    Parameters
    ----------
    a : array, shape (M, N)
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

    Raises LinAlgError if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, and zungqr.

    Examples
    --------
    >>> from numpy import *
    >>> a = random.randn(9, 6)
    >>> q, r = linalg.qr(a)
    >>> allclose(a, dot(q, r))
    True
    >>> r2 = linalg.qr(a, mode='r')
    >>> r3 = linalg.qr(a, mode='economic')
    >>> allclose(r, r2)
    True
    >>> allclose(r, triu(r3[:6,:6], k=0))
    True

    """
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

    return q, r


# Eigenvalues


def eigvals(a):
    """Compute the eigenvalues of a general matrix.

    Parameters
    ----------
    a : array, shape (M, M)
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.

    Returns
    -------
    w : double or complex array, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered, nor are they necessarily
        real for real matrices.

    Raises LinAlgError if eigenvalue computation does not converge

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
    """Compute the eigenvalues of a Hermitean or real symmetric matrix.

    Parameters
    ----------
    a : array, shape (M, M)
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    UPLO : {'L', 'U'}
        Specifies whether the pertinent array data is taken from the upper
        or lower triangular part of a. Possible values are 'L', and 'U' for
        upper and lower respectively. Default is 'L'.

    Returns
    -------
    w : double array, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered.

    Raises LinAlgError if eigenvalue computation does not converge

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
    """Compute eigenvalues and right eigenvectors of a general matrix.

    Parameters
    ----------
    a : array, shape (M, M)
        A complex or real 2-d array whose eigenvalues and eigenvectors
        will be computed.

    Returns
    -------
    w : double or complex array, shape (M,)
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered, nor are they
        necessarily real for real matrices.
    v : double or complex array, shape (M, M)
        The normalized eigenvector corresponding to the eigenvalue w[i] is
        the column v[:,i].

    Raises LinAlgError if eigenvalue computation does not converge

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

    The number w is an eigenvalue of a if there exists a vector v
    satisfying the equation dot(a,v) = w*v. Alternately, if w is a root of
    the characteristic equation det(a - w[i]*I) = 0, where det is the
    determinant and I is the identity matrix. The arrays a, w, and v
    satisfy the equation dot(a,v[i]) = w[i]*v[:,i].

    The array v of eigenvectors may not be of maximum rank, that is, some
    of the columns may be dependent, although roundoff error may obscure
    that fact. If the eigenvalues are all different, then theoretically the
    eigenvectors are independent. Likewise, the matrix of eigenvectors is
    unitary if the matrix a is normal, i.e., if dot(a, a.H) = dot(a.H, a).

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
    """Compute eigenvalues for a Hermitian or real symmetric matrix.

    Parameters
    ----------
    a : array, shape (M, M)
        A complex Hermitian or symmetric real matrix whose eigenvalues
        and eigenvectors will be computed.
    UPLO : {'L', 'U'}
        Specifies whether the pertinent array date is taken from the upper
        or lower triangular part of a. Possible values are 'L', and 'U'.
        Default is 'L'.

    Returns
    -------
    w : double array, shape (M,)
        The eigenvalues. The eigenvalues are not necessarily ordered.
    v : double or complex double array, shape (M, M)
        The normalized eigenvector corresponding to the eigenvalue w[i] is
        the column v[:,i].

    Raises LinAlgError if eigenvalue computation does not converge

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
    """Singular Value Decomposition.

    Factorizes the matrix a into two unitary matrices U and Vh and
    an 1d-array s of singular values (real, non-negative) such that
    a == U S Vh  if S is an suitably shaped matrix of zeros whose
    main diagonal is s.

    Parameters
    ----------
    a : array, shape (M, N)
        Matrix to decompose
    full_matrices : boolean
        If true,  U, Vh are shaped  (M,M), (N,N)
        If false, the shapes are    (M,K), (K,N) where K = min(M,N)
    compute_uv : boolean
        Whether to compute also U, Vh in addition to s

    Returns
    -------
    U:  array, shape (M,M) or (M,K) depending on full_matrices
    s:  array, shape (K,)
        The singular values, sorted so that s[i] >= s[i+1]
        K = min(M, N)
    Vh: array, shape (N,N) or (K,N) depending on full_matrices

    For compute_uv = False, only s is returned.

    Raises LinAlgError if SVD computation does not converge

    Examples
    --------
    >>> a = random.randn(9, 6) + 1j*random.randn(9, 6)
    >>> U, s, Vh = linalg.svd(a)
    >>> U.shape, Vh.shape, s.shape
    ((9, 9), (6, 6), (6,))

    >>> U, s, Vh = linalg.svd(a, full_matrices=False)
    >>> U.shape, Vh.shape, s.shape
    ((9, 6), (6, 6), (6,))
    >>> S = diag(s)
    >>> allclose(a, dot(U, dot(S, Vh)))
    True

    >>> s2 = linalg.svd(a, compute_uv=False)
    >>> allclose(s, s2)
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

def cond(x,p=None):
    """Compute the condition number of a matrix.

    The condition number of x is the norm of x times the norm 
    of the inverse of x.  The norm can be the usual L2 
    (root-of-sum-of-squares) norm or a number of other matrix norms.

    Parameters
    ----------
    x : array, shape (M, N)
        The matrix whose condition number is sought.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}
        Order of the norm:

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
    if p is None:
        s = svd(x,compute_uv=False)
        return s[0]/s[-1]
    else:
        return norm(x,p)*norm(inv(x),p)

# Generalized inverse

def pinv(a, rcond=1e-15 ):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.

    Parameters
    ----------
    a : array, shape (M, N)
        Matrix to be pseudo-inverted
    rcond : float
        Cutoff for 'small' singular values.
        Singular values smaller than rcond*largest_singular_value are
        considered zero.

    Returns
    -------
    B : array, shape (N, M)

    Raises LinAlgError if SVD computation does not converge

    Examples
    --------
    >>> from numpy import *
    >>> a = random.randn(9, 6)
    >>> B = linalg.pinv(a)
    >>> allclose(a, dot(a, dot(B, a)))
    True
    >>> allclose(B, dot(B, dot(a, B)))
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
    return wrap(dot(transpose(vt),
                       multiply(s[:, newaxis],transpose(u))))

# Determinant

def det(a):
    """Compute the determinant of a matrix

    Parameters
    ----------
    a : array, shape (M, M)

    Returns
    -------
    det : float or complex
        Determinant of a

    Notes
    -----
    The determinant is computed via LU factorization, LAPACK routine z/dgetrf.
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
    """Compute least-squares solution to equation :m:`a x = b`

    Compute a vector x such that the 2-norm :m:`|b - a x|` is minimised.

    Parameters
    ----------
    a : array, shape (M, N)
    b : array, shape (M,) or (M, K)
    rcond : float
        Cutoff for 'small' singular values.
        Singular values smaller than rcond*largest_singular_value are
        considered zero.

    Raises LinAlgError if computation does not converge

    Returns
    -------
    x : array, shape (N,) or (N, K) depending on shape of b
        Least-squares solution
    residues : array, shape () or (1,) or (K,)
        Sums of residues, squared 2-norm for each column in :m:`b - a x`
        If rank of matrix a is < N or > M this is an empty array.
        If b was 1-d, this is an (1,) shape array, otherwise the shape is (K,)
    rank : integer
        Rank of matrix a
    s : array, shape (min(M,N),)
        Singular values of a
    """
    import math
    a = asarray(a)
    b, wrap = _makearray(b)
    one_eq = len(b.shape) == 1
    if one_eq:
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
    if one_eq:
        x = array(ravel(bstar)[:n], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            resids = array([sum((ravel(bstar)[n:])**2)], dtype=result_t)
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            resids = sum((transpose(bstar)[n:,:])**2, axis=0).astype(result_t)
    st = s[:min(n, m)].copy().astype(_realType(result_t))
    return wrap(x), resids, results['rank'], st

def norm(x, ord=None):
    """Matrix or vector norm.

    Parameters
    ----------
    x : array, shape (M,) or (M, N)
    ord : number, or {None, 1, -1, 2, -2, inf, -inf, 'fro'}
        Order of the norm:

        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                -
        inf    max(sum(abs(x), axis=1))      max(abs(x))
        -inf   min(sum(abs(x), axis=1))      min(abs(x))
        1      max(sum(abs(x), axis=0))      as below
        -1     min(sum(abs(x), axis=0))      as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  -                             sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

    Returns
    -------
    n : float
        Norm of the matrix or vector

    Notes
    -----
    For values ord < 0, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for numerical
    purposes.

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
