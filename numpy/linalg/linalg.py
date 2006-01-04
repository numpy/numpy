"""Lite version of numpy.linalg.
"""
# This module is a lite version of LinAlg.py module which contains
# high-level Python interface to the LAPACK library.  The lite version
# only accesses the following LAPACK functions: dgesv, zgesv, dgeev,
# zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf, dpotrf.

__all__ = ['LinAlgError', 'solve_linear_equations', 'solve',
           'inverse', 'inv', 'cholesky_decomposition', 'cholesky', 'eigenvalues',
           'eigvals', 'Heigenvalues', 'eigvalsh', 'generalized_inverse', 'pinv',
           'determinant', 'det', 'singular_value_decomposition', 'svd',
           'eigenvectors', 'eig', 'Heigenvectors', 'eigh','lstsq', 'linear_least_squares'
           ]
           
from numpy.core import *
import lapack_lite
import math

# Error object
class LinAlgError(Exception):
    pass

# Helper routines
_lapack_type = {'f': 0, 'd': 1, 'F': 2, 'D': 3}
_lapack_letter = ['s', 'd', 'c', 'z']
_array_kind = {'i':0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]

def _commonType(*arrays):
    kind = 0
#    precision = 0
#   force higher precision in lite version
    precision = 1
    for a in arrays:
        t = a.dtypechar
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]

def _castCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        cast_arrays = cast_arrays + (transpose(a).astype(type),)
    if len(cast_arrays) == 1:
            return cast_arrays[0]
    else:
        return cast_arrays

# _fastCopyAndTranpose is an optimized version of _castCopyAndTranspose.
# It assumes the input is 2D (as all the calls in here are).

_fastCT = fastCopyAndTranspose

def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtypechar == type:
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

def solve_linear_equations(a, b):
    one_eq = len(b.shape) == 1
    if one_eq:
        b = b[:, NewAxis]
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

def inverse(a):
    return solve_linear_equations(a, identity(a.shape[0]))


# Cholesky decomposition

def cholesky_decomposition(a):
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

def eigenvalues(a):
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


def Heigenvalues(a, UPLO='L'):
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
    if issubclass(a.dtype, complexfloating):
        if a.dtypechar == 'D':
            a = _fastCT(a)
        else:
            a = _fastCT(a.astype('D'))
    else:
        if a.dtypechar == 'd':
            a = _fastCT(a)
        else:
            a = _fastCT(a.astype('d'))
    return a, a.dtypechar

# Eigenvectors

def eig(a):
    """eig(a) returns u,v  where u is the eigenvalues and
v is a matrix of eigenvectors with vector v[:,i] corresponds to
eigenvalue u[i].  Satisfies the equation dot(a, v[:,i]) = u[i]*v[:,i]
"""
    a = asarray(a)    
    _assertRank2(a)
    _assertSquareness(a)
    a,t = _convertarray(a) # convert to float_ or complex_ type
    wrap = a.__array_wrap__
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
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _castCopyAndTranspose(t, a)
    wrap = a.__array_wrap__
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

def svd(a, full_matrices = 1):
    _assertRank2(a)
    n = a.shape[1]
    m = a.shape[0]
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _fastCopyAndTranspose(t, a)
    wrap = a.__array_wrap__
    if full_matrices:
        nu = m
        nvt = n
        option = 'A'
    else:
        nu = min(n,m)
        nvt = min(n,m)
        option = 'S'
    s = zeros((min(n,m),), real_t)
    u = zeros((nu, m), t)
    vt = zeros((n, nvt), t)
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
        work = zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge'
    return wrap(transpose(u)), s, \
           wrap(transpose(vt)) # why copy here?


# Generalized inverse

def generalized_inverse(a, rcond = 1.e-10):
    a = array(a, copy=0)
    if a.dtypechar in typecodes['Complex']:
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
    wrap = a.__array_wrap__
    return wrap(dot(transpose(vt),                        
                       multiply(s[:, NewAxis],transpose(u))))

# Determinant

def determinant(a):
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
    sign = add.reduce(not_equal(pivots,
                                                arrayrange(1, n+1))) % 2
    return (1.-2.*sign)*multiply.reduce(diagonal(a),axis=-1)

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
    a = asarray(a)
    b = asarray(b)
    one_eq = len(b.shape) == 1
    if one_eq:
        b = b[:, NewAxis]
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
    return x,resids,results['rank'],s[:min(n,m)].copy()

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

if __name__ == '__main__':
    def test(a, b):

        print "All numbers printed should be (almost) zero:"

        x = solve_linear_equations(a, b)
        check = b - matrixmultiply(a, x)
        print check


        a_inv = inverse(a)
        check = matrixmultiply(a, a_inv)-identity(a.shape[0])
        print check


        ev = eigenvalues(a)

        evalues, evectors = eig(a)
        check = ev-evalues
        print check

        evectors = transpose(evectors)
        check = matrixmultiply(a, evectors)-evectors*evalues
        print check


        u, s, vt = svd(a,0)
        check = a - matrixmultiply(u*s, vt)
        print check


        a_ginv = generalized_inverse(a)
        check = matrixmultiply(a, a_ginv)-identity(a.shape[0])
        print check


        det = determinant(a)
        check = det-multiply.reduce(evalues)
        print check

        x, residuals, rank, sv = linear_least_squares(a, b)
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


