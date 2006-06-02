
"""Lite version of scipy.linalg.
"""
# This module is a lite version of the linalg.py module in SciPy which contains
# high-level Python interface to the LAPACK library.  The lite version
# only accesses the following LAPACK functions: dgesv, zgesv, dgeev,
# zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf, dpotrf.

__all__ = ['solve',
           'inv', 'cholesky',
           'eigvals',
           'eigvalsh', 'pinv',
           'det', 'svd',
           'eig', 'eigh','lstsq', 'norm',
           'LinAlgError'
           ]

from numpy.core import *
from numpy.lib import *
import lapack_lite

# Error object
class LinAlgError(Exception):
    pass

# Helper routines
_array_kind = {'i':0, 'l': 0, 'q': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'q': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]

def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap

def _commonType(*arrays):
    kind = 0
#    precision = 0
#   force higher precision in lite version
    precision = 1
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]

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
        if a.dtype.char == type:
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

def solve(a, b):
    one_eq = len(b.shape) == 1
    if one_eq:
        b = b[:, newaxis]
    _assertRank2(a, b)
    _assertSquareness(a)
    n_eq = a.shape[0]
    n_rhs = b.shape[1]
    if n_eq != b.shape[0]:
        raise LinAlgError, 'Incompatible dimensions'
    t =_commonType(a, b)
#    lapack_routine = _findLapackRoutine('gesv', t)
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgesv
    else:
        lapack_routine = lapack_lite.dgesv
    a, b = _fastCopyAndTranspose(t, a, b)
    pivots = zeros(n_eq, 'i')
    results = lapack_routine(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Singular matrix'
    if one_eq:
        return ravel(b) # I see no need to copy here
    else:
        return transpose(b) # no need to copy


# Matrix inversion

def inv(a):
    a, wrap = _makearray(a)
    return wrap(solve(a, identity(a.shape[0])))

# Cholesky decomposition

def cholesky(a):
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    a = _castCopyAndTranspose(t, a)
    m = a.shape[0]
    n = a.shape[1]
    if _array_kind[t] == 1:
        lapack_routine = lapack_lite.zpotrf
    else:
        lapack_routine = lapack_lite.dpotrf
    results = lapack_routine('L', n, a, m, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Matrix is not positive definite - Cholesky decomposition cannot be computed'
    return transpose(triu(a,k=0)).copy()


# Eigenvalues
def eigvals(a):
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _fastCopyAndTranspose(t, a)
    n = a.shape[0]
    dummy = zeros((1,), t)
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgeev
        w = zeros((n,), t)
        rwork = zeros((n,),real_t)
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
        if logical_and.reduce(equal(wi, 0.)):
            w = wr
        else:
            w = wr+1j*wi
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w


def eigvalsh(a, UPLO='L'):
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _castCopyAndTranspose(t, a)
    n = a.shape[0]
    liwork = 5*n+3
    iwork = zeros((liwork,),'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zheevd
        w = zeros((n,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        lrwork = 1
        rwork = zeros((lrwork,),real_t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, -1, rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,),real_t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, lwork, rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, -1, iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, lwork, iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w

def _convertarray(a):
    if issubclass(a.dtype.type, complexfloating):
        if a.dtype.char == 'D':
            a = _fastCT(a)
        else:
            a = _fastCT(a.astype('D'))
    else:
        if a.dtype.char == 'd':
            a = _fastCT(a)
        else:
            a = _fastCT(a.astype('d'))
    return a, a.dtype.char

# Eigenvectors

def eig(a):
    """eig(a) returns u,v  where u is the eigenvalues and
v is a matrix of eigenvectors with vector v[:,i] corresponds to
eigenvalue u[i].  Satisfies the equation dot(a, v[:,i]) = u[i]*v[:,i]
"""
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    a,t = _convertarray(a) # convert to float_ or complex_ type
    real_t = 'd'
    n = a.shape[0]
    dummy = zeros((1,), t)
    if t == 'D': # Complex routines take different arguments
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
        if logical_and.reduce(equal(wi, 0.)):
            w = wr
            v = vr
        else:
            w = wr+1j*wi
            v = array(vr,Complex)
            ind = nonzero(
                          equal(
                              equal(wi,0.0) # true for real e-vals
                                       ,0)          # true for complex e-vals
                                 )                  # indices of complex e-vals
            for i in range(len(ind)/2):
                v[ind[2*i]] = vr[ind[2*i]] + 1j*vr[ind[2*i+1]]
                v[ind[2*i+1]] = vr[ind[2*i]] - 1j*vr[ind[2*i+1]]
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w,wrap(v.transpose())


def eigh(a, UPLO='L'):
    a, wrap = _makearray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _castCopyAndTranspose(t, a)
    n = a.shape[0]
    liwork = 5*n+3
    iwork = zeros((liwork,),'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zheevd
        w = zeros((n,), real_t)
        lwork = 1
        work = zeros((lwork,), t)
        lrwork = 1
        rwork = zeros((lrwork,),real_t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, -1, rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = zeros((lrwork,),real_t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, lwork, rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = zeros((n,), t)
        lwork = 1
        work = zeros((lwork,),t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, -1, iwork, liwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,),t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, lwork, iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w,wrap(a.transpose())


# Singular value decomposition

def svd(a, full_matrices=1, compute_uv=1):
    a, wrap = _makearray(a)
    _assertRank2(a)
    m, n = a.shape
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
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
        u = empty((1,1),t) 
        vt = empty((1,1),t) 

    iwork = zeros((8*min(m,n),), 'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
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
    if compute_uv:
        return wrap(transpose(u)), s, \
               wrap(transpose(vt)) # why copy here?
    else:
        return s

# Generalized inverse

def pinv(a, rcond = 1.e-10):
    a, wrap = _makearray(a)
    if a.dtype.char in typecodes['Complex']:
        a = conjugate(a)
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
    a = asarray(a)
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    a = _fastCopyAndTranspose(t, a)
    n = a.shape[0]
    if _array_kind[t] == 1:
        lapack_routine = lapack_lite.zgetrf
    else:
        lapack_routine = lapack_lite.dgetrf
    pivots = zeros((n,), 'i')
    results = lapack_routine(n, n, a, n, pivots, 0)
    sign = add.reduce(not_equal(pivots, arange(1, n+1))) % 2
    return (1.-2.*sign)*multiply.reduce(diagonal(a),axis=-1)

# Linear Least Squares

def lstsq(a, b, rcond=1.e-10):
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
    t =_commonType(a, b)
    real_t = _array_type[0][_array_precision[t]]
    bstar = zeros((ldb,n_rhs),t)
    bstar[:b.shape[0],:n_rhs] = b.copy()
    a,bstar = _castCopyAndTranspose(t, a, bstar)
    s = zeros((min(m,n),),real_t)
    nlvl = max( 0, int( math.log( float(min( m,n ))/2. ) ) + 1 )
    iwork = zeros((3*min(m,n)*nlvl+11*min(m,n),), 'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = zeros((lwork,), real_t)
        work = zeros((lwork,),t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,-1,rwork,iwork,0 )
        lwork = int(abs(work[0]))
        rwork = zeros((lwork,),real_t)
        a_real = zeros((m,n),real_t)
        bstar_real = zeros((ldb,n_rhs,),real_t)
        results = lapack_lite.dgelsd( m, n, n_rhs, a_real, m, bstar_real,ldb , s, rcond,
                        0,rwork,-1,iwork,0 )
        lrwork = int(rwork[0])
        work = zeros((lwork,), t)
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,lwork,rwork,iwork,0 )
    else:
        lapack_routine = lapack_lite.dgelsd
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,-1,iwork,0 )
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,lwork,iwork,0 )
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge in Linear Least Squares'
    resids = array([],t)
    if one_eq:
        x = ravel(bstar)[:n].copy()
        if (results['rank']==n) and (m>n):
            resids = array([sum((ravel(bstar)[n:])**2)])
    else:
        x = transpose(bstar)[:n,:].copy()
        if (results['rank']==n) and (m>n):
            resids = sum((transpose(bstar)[n:,:])**2).copy()
    return wrap(x),resids,results['rank'],s[:min(n,m)].copy()

def norm(x, ord=None):
    """ norm(x, ord=None) -> n

    Matrix or vector norm.

    Inputs:

      x -- a rank-1 (vector) or rank-2 (matrix) array
      ord -- the order of the norm.

     Comments:
       For arrays of any rank, if ord is None:
         calculate the square norm (Euclidean norm for vectors, Frobenius norm for matrices)

       For vectors ord can be any real number including Inf or -Inf.
         ord = Inf, computes the maximum of the magnitudes
         ord = -Inf, computes minimum of the magnitudes
         ord is finite, computes sum(abs(x)**ord)**(1.0/ord)

       For matrices ord can only be one of the following values:
         ord = 2 computes the largest singular value
         ord = -2 computes the smallest singular value
         ord = 1 computes the largest column sum of absolute values
         ord = -1 computes the smallest column sum of absolute values
         ord = Inf computes the largest row sum of absolute values
         ord = -Inf computes the smallest row sum of absolute values
         ord = 'fro' computes the frobenius norm sqrt(sum(diag(X.H * X)))

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

if __name__ == '__main__':
    def test(a, b):

        print "All numbers printed should be (almost) zero:"

        x = solve(a, b)
        check = b - matrixmultiply(a, x)
        print check


        a_inv = inv(a)
        check = matrixmultiply(a, a_inv)-identity(a.shape[0])
        print check


        ev = eigvals(a)

        evalues, evectors = eig(a)
        check = ev-evalues
        print check

        evectors = transpose(evectors)
        check = matrixmultiply(a, evectors)-evectors*evalues
        print check


        u, s, vt = svd(a,0)
        check = a - matrixmultiply(u*s, vt)
        print check


        a_ginv = pinv(a)
        check = matrixmultiply(a, a_ginv)-identity(a.shape[0])
        print check


        det = det(a)
        check = det-multiply.reduce(evalues)
        print check

        x, residuals, rank, sv = lstsq(a, b)
        check = b - matrixmultiply(a, x)
        print check
        print rank-a.shape[0]
        print sv-s

    a = array([[1.,2.], [3.,4.]])
    b = array([2., 1.])
    test(a, b)

    a = a+0j
    b = b+0j
    test(a, b)
