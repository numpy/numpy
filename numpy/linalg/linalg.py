"""Lite version of scipy.linalg.

This module is a lite version of the linalg.py module in SciPy which contains
high-level Python interface to the LAPACK library.  The lite version
only accesses the following LAPACK functions: dgesv, zgesv, dgeev,
zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf, dpotrf.
"""

__all__ = ['solve', 'tensorsolve', 'tensorinv',
           'inv', 'cholesky',
           'eigvals',
           'eigvalsh', 'pinv',
           'det', 'svd',
           'eig', 'eigh','lstsq', 'norm',
           'qr',
           'LinAlgError'
           ]

from numpy.core import array, asarray, zeros, empty, transpose, \
        intc, single, double, csingle, cdouble, inexact, complexfloating, \
        newaxis, ravel, all, Inf, dot, add, multiply, identity, sqrt, \
        maximum, flatnonzero, diagonal, arange, fastCopyAndTranspose, sum, \
        argsort
from numpy.lib import triu, iterable
from numpy.linalg import lapack_lite

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
            raise LinAlgError, 'Array must be two-dimensional'

def _assertSquareness(*arrays):
    for a in arrays:
        if max(a.shape) != min(a.shape):
            raise LinAlgError, 'Array must be square'

# Linear equations

def tensorsolve(a, b, axes=None):
    """Solves the tensor equation a x = b for x

    where it is assumed that all the indices of x are summed over in
    the product.

    a can be N-dimensional.  x will have the dimensions of A subtracted from
    the dimensions of b.
    """
    a = asarray(a)
    b = asarray(b)
    an = a.ndim

    if axes is not None:
        allaxes = range(0,an)
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)

    oldshape = a.shape[-(an-b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k

    a = a.reshape(-1,prod)
    b = b.ravel()
    res = solve(a, b)
    res.shape = oldshape
    return res

def solve(a, b):
    """Return the solution of a*x = b
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

    ind must be a positive integer specifying
    how many indices at the front of the array are involved
    in the inverse sum.

    the result is ainv with shape a.shape[ind:] + a.shape[:ind]

    tensordot(ainv, a, ind) is an identity operator

    and so is

    tensordot(a, ainv, a.shape-newind)

    Example:

       a = rand(4,6,8,3)
       ainv = tensorinv(a)
       # ainv.shape is (8,3,4,6)
       # suppose b has shape (4,6)
       tensordot(ainv, b) # produces same (8,3)-shaped output as
       tensorsolve(a, b)

       a = rand(24,8,3)
       ainv = tensorinv(a,1)
       # ainv.shape is (8,3,24)
       # suppose b has shape (24,)
       tensordot(ainv, b, 1)  # produces the same (8,3)-shaped output as
       tensorsolve(a, b)

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
    a = a.reshape(prod,-1)
    ia = inv(a)
    return ia.reshape(*invshape)


# Matrix inversion

def inv(a):
    a, wrap = _makearray(a)
    return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))

# Cholesky decomposition

def cholesky(a):
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
        raise LinAlgError, 'Matrix is not positive definite - Cholesky decomposition cannot be computed'
    s = triu(a, k=0).transpose()
    if (s.dtype != result_t):
        return s.astype(result_t)
    return s

# QR decompostion

def qr(a, mode='full'):
    """cacluates A=QR, Q orthonormal, R upper triangular

    mode:  'full' --> (Q,R)
           'r'    --> R
           'economic' --> A2 where the diagonal + upper triangle
                   part of A2 is R. This is faster if you only need R
    """
    _assertRank2(a)
    m,n = a.shape
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
    mn = min(m,n)
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
    results=lapack_routine(m, n, a, m, tau, work, -1, 0)
    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    # do qr decomposition
    lwork = int(abs(work[0]))
    work = zeros((lwork,),t)
    results=lapack_routine(m, n, a, m, tau, work, lwork, 0)

    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    #  economic mode. Isn't actually economic.
    if mode[0]=='e':
        if t != result_t :
            a = a.astype(result_t)
        return a.T

    #  generate r
    r = _fastCopyAndTranspose(result_t, a[:,:mn])
    for i in range(mn):
        r[i,:i].fill(0.0)

    #  'r'-mode, that is, calculate only r
    if mode[0]=='r':
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
    work=zeros((lwork,), t)
    results=lapack_routine(m,mn,mn, a, m, tau, work, -1, 0)
    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    # compute q
    lwork = int(abs(work[0]))
    work=zeros((lwork,), t)
    results=lapack_routine(m,mn,mn, a, m, tau, work, lwork, 0)
    if results['info'] != 0:
        raise LinAlgError, '%s returns %d' % (routine_name, results['info'])

    q = _fastCopyAndTranspose(result_t, a[:mn,:])

    return q,r


# Eigenvalues
def eigvals(a):
    _assertRank2(a)
    _assertSquareness(a)
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
        rwork = zeros((lrwork,),real_t)
        results = lapack_routine('N', UPLO, n, a, n, w, work, -1,
                                 rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,),real_t)
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
    """eig(a) returns u,v  where u is the eigenvalues and
v is a matrix of eigenvectors with vector v[:,i] corresponds to
eigenvalue u[i].  Satisfies the equation dot(a, v[:,i]) = u[i]*v[:,i]
"""
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    a, t, result_t = _convertarray(a) # convert to double or cdouble type
    real_t = _linalgRealType(t)
    n = a.shape[0]
    dummy = zeros((1,), t)
    if isComplexType(t):
        # Complex routines take different arguments
        lapack_routine = lapack_lite.zgeev
        w = zeros((n,), t)
        v = zeros((n,n), t)
        lwork = 1
        work = zeros((lwork,),t)
        rwork = zeros((2*n,),real_t)
        results = lapack_routine('N', 'V', n, a, n, w,
                                 dummy, 1, v, n, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,),t)
        results = lapack_routine('N', 'V', n, a, n, w,
                                 dummy, 1, v, n, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = zeros((n,), t)
        wi = zeros((n,), t)
        vr = zeros((n,n), t)
        lwork = 1
        work = zeros((lwork,),t)
        results = lapack_routine('N', 'V', n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, -1, 0)
        lwork = int(work[0])
        work = zeros((lwork,),t)
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
    """Compute eigenvalues for a Hermitian-symmetric matrix.
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
        rwork = zeros((lrwork,),real_t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, -1,
                                 rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, lwork,
                                 rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,),t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, -1,
                iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,),t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, lwork,
                iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    at = a.transpose().astype(result_t)
    return w.astype(_realType(result_t)), wrap(at)


# Singular value decomposition

def svd(a, full_matrices=1, compute_uv=1):
    """Singular Value Decomposition.

    u,s,vh = svd(a)

    If a is an M x N array, then the svd produces a factoring of the array
    into two unitary (orthogonal) 2-d arrays u (MxM) and vh (NxN) and a
    min(M,N)-length array of singular values such that

                     a == dot(u,dot(S,vh))

    where S is an MxN array of zeros whose main diagonal is s.

    if compute_uv == 0, then return only the singular values
    if full_matrices == 0, then only part of either u or vh is
                           returned so that it is MxN
    """
    a, wrap = _makearray(a)
    _assertRank2(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    real_t = _linalgRealType(t)
    a = _fastCopyAndTranspose(t, a)
    s = zeros((min(n,m),), real_t)
    if compute_uv:
        if full_matrices:
            nu = m
            nvt = n
            option = 'A'
        else:
            nu = min(n,m)
            nvt = min(n,m)
            option = 'S'
        u = zeros((nu, m), t)
        vt = zeros((n, nvt), t)
    else:
        option = 'N'
        nu = 1
        nvt = 1
        u = empty((1,1), t)
        vt = empty((1,1), t)

    iwork = zeros((8*min(m,n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgesdd
        rwork = zeros((5*min(m,n)*min(m,n) + 5*min(m,n),), real_t)
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
        if option == 'N' and lwork==1:
            # there seems to be a bug in dgesdd of lapack
            #   (NNemec, 060310)
            # returning the wrong lwork size for option == 'N'
            results = lapack_routine('A', m, n, a, m, s, u, m, vt, n,
                                     work, -1, iwork, 0)
            lwork = int(work[0])
            assert lwork > 1

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

# Generalized inverse

def pinv(a, rcond=1e-15 ):
    """Return the (Moore-Penrose) pseudo-inverse of a 2-d array

    This method computes the generalized inverse using the
    singular-value decomposition and all singular values larger than
    rcond of the largest.
    """
    a, wrap = _makearray(a)
    a = a.conjugate()
    u, s, vt = svd(a, 0)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond*maximum.reduce(s)
    for i in range(min(n,m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.;
    return wrap(dot(transpose(vt),
                       multiply(s[:, newaxis],transpose(u))))

# Determinant

def det(a):
    "The determinant of the 2-d array a"
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
    return (1.-2.*sign)*multiply.reduce(diagonal(a),axis=-1)

# Linear Least Squares

def lstsq(a, b, rcond=-1):
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
    ldb = max(n,m)
    if m != b.shape[0]:
        raise LinAlgError, 'Incompatible dimensions'
    t, result_t = _commonType(a, b)
    real_t = _linalgRealType(t)
    bstar = zeros((ldb,n_rhs),t)
    bstar[:b.shape[0],:n_rhs] = b.copy()
    a, bstar = _fastCopyAndTranspose(t, a, bstar)
    s = zeros((min(m,n),),real_t)
    nlvl = max( 0, int( math.log( float(min( m,n ))/2. ) ) + 1 )
    iwork = zeros((3*min(m,n)*nlvl+11*min(m,n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = zeros((lwork,), real_t)
        work = zeros((lwork,),t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        rwork = zeros((lwork,),real_t)
        a_real = zeros((m,n),real_t)
        bstar_real = zeros((ldb,n_rhs,),real_t)
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
        if results['rank']==n and m>n:
            resids = array([sum((ravel(bstar)[n:])**2)], dtype=result_t)
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank']==n and m>n:
            resids = sum((transpose(bstar)[n:,:])**2,axis=0).astype(result_t)
    st = s[:min(n,m)].copy().astype(_realType(result_t))
    return wrap(x), resids, results['rank'], st

def norm(x, ord=None):
    """ norm(x, ord=None) -> n

    Matrix or vector norm.

    Inputs:

      x -- a rank-1 (vector) or rank-2 (matrix) array
      ord -- the order of the norm.

     Comments:
       For arrays of any rank, if ord is None:
         calculate the square norm (Euclidean norm for vectors,
         Frobenius norm for matrices)

       For vectors ord can be any real number including Inf or -Inf.
         ord = Inf, computes the maximum of the magnitudes
         ord = -Inf, computes minimum of the magnitudes
         ord is finite, computes sum(abs(x)**ord,axis=0)**(1.0/ord)

       For matrices ord can only be one of the following values:
         ord = 2 computes the largest singular value
         ord = -2 computes the smallest singular value
         ord = 1 computes the largest column sum of absolute values
         ord = -1 computes the smallest column sum of absolute values
         ord = Inf computes the largest row sum of absolute values
         ord = -Inf computes the smallest row sum of absolute values
         ord = 'fro' computes the frobenius norm sqrt(sum(diag(X.H * X),axis=0))

       For values ord < 0, the result is, strictly speaking, not a
       mathematical 'norm', but it may still be useful for numerical purposes.
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
            return svd(x,compute_uv=0).max()
        elif ord == -2:
            return svd(x,compute_uv=0).min()
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
