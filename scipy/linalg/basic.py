# This module is a lite version of LinAlg.py module which contains
# high-level Python interface to the LAPACK library.  The lite version
# only accesses the following LAPACK functions: dgesv, zgesv, dgeev,
# zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf, dpotrf.

import Numeric
import copy
import lapack_lite
import math
import MLab
import multiarray

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
        t = a.typecode()
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]

def _castCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.typecode() == type:
            cast_arrays = cast_arrays + (copy.copy(Numeric.transpose(a)),)
        else:
            cast_arrays = cast_arrays + (copy.copy(
                                       Numeric.transpose(a).astype(type)),)
    if len(cast_arrays) == 1:
            return cast_arrays[0]
    else:
        return cast_arrays

# _fastCopyAndTranpose is an optimized version of _castCopyAndTranspose.
# It assumes the input is 2D (as all the calls in here are).

_fastCT = multiarray._fastCopyAndTranspose

def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.typecode() == type:
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
        b = b[:, Numeric.NewAxis]
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
    pivots = Numeric.zeros(n_eq, 'i')
    results = lapack_routine(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Singular matrix'
    if one_eq:
        return Numeric.ravel(b) # I see no need to copy here
    else:
        return multiarray.transpose(b) # no need to copy


# Matrix inversion

def inverse(a):
    return solve_linear_equations(a, Numeric.identity(a.shape[0]))

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
    return copy.copy(Numeric.transpose(MLab.triu(a,k=0)))


# Eigenvalues

def eigenvalues(a):
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _fastCopyAndTranspose(t, a)
    n = a.shape[0]
    dummy = Numeric.zeros((1,), t)
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgeev
        w = Numeric.zeros((n,), t)
        rwork = Numeric.zeros((n,),real_t)
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, w,
                                 dummy, 1, dummy, 1, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, w,
                                 dummy, 1, dummy, 1, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = Numeric.zeros((n,), t)
        wi = Numeric.zeros((n,), t)
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, wr, wi,
                                 dummy, 1, dummy, 1, work, -1, 0)
        lwork = int(work[0])
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine('N', 'N', n, a, n, wr, wi,
                                 dummy, 1, dummy, 1, work, lwork, 0)
        if Numeric.logical_and.reduce(Numeric.equal(wi, 0.)):
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
    iwork = Numeric.zeros((liwork,),'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zheevd
        w = Numeric.zeros((n,), real_t)
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        lrwork = 1
        rwork = Numeric.zeros((lrwork,),real_t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, -1, rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = Numeric.zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = Numeric.zeros((lrwork,),real_t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, lwork, rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = Numeric.zeros((n,), t)
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, -1, iwork, liwork, 0)
        lwork = int(work[0])
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine('N', UPLO, n, a, n,w, work, lwork, iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w

# Eigenvectors

def eigenvectors(a):
    """eigenvectors(a) returns u,v  where u is the eigenvalues and
v is a matrix of eigenvectors with vector v[i] corresponds to
eigenvalue u[i].  Satisfies the equation dot(a, v[i]) = u[i]*v[i]
"""
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _fastCopyAndTranspose(t, a)
    n = a.shape[0]
    dummy = Numeric.zeros((1,), t)
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgeev
        w = Numeric.zeros((n,), t)
        v = Numeric.zeros((n,n), t)
        lwork = 1
        work = Numeric.zeros((lwork,),t)
        rwork = Numeric.zeros((2*n,),real_t)
        results = lapack_routine('N', 'V', n, a, n, w,
                                  dummy, 1, v, n, work, -1, rwork, 0)
        lwork = int(abs(work[0]))
        work = Numeric.zeros((lwork,),t)
        results = lapack_routine('N', 'V', n, a, n, w,
                                  dummy, 1, v, n, work, lwork, rwork, 0)
    else:
        lapack_routine = lapack_lite.dgeev
        wr = Numeric.zeros((n,), t)
        wi = Numeric.zeros((n,), t)
        vr = Numeric.zeros((n,n), t)
        lwork = 1
        work = Numeric.zeros((lwork,),t)
        results = lapack_routine('N', 'V', n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, -1, 0)
        lwork = int(work[0])
        work = Numeric.zeros((lwork,),t)
        results = lapack_routine('N', 'V', n, a, n, wr, wi,
                                  dummy, 1, vr, n, work, lwork, 0)
        if Numeric.logical_and.reduce(Numeric.equal(wi, 0.)):
            w = wr
            v = vr
        else:
            w = wr+1j*wi
            v = Numeric.array(vr,Numeric.Complex)
            ind = Numeric.nonzero(
                          Numeric.equal(
                              Numeric.equal(wi,0.0) # true for real e-vals
                                       ,0)          # true for complex e-vals
                                 )                  # indices of complex e-vals
            for i in range(len(ind)/2):
                v[ind[2*i]] = vr[ind[2*i]] + 1j*vr[ind[2*i+1]]
                v[ind[2*i+1]] = vr[ind[2*i]] - 1j*vr[ind[2*i+1]]
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return w,v


def Heigenvectors(a, UPLO='L'):
    _assertRank2(a)
    _assertSquareness(a)
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _castCopyAndTranspose(t, a)
    n = a.shape[0]
    liwork = 5*n+3
    iwork = Numeric.zeros((liwork,),'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zheevd
        w = Numeric.zeros((n,), real_t)
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        lrwork = 1
        rwork = Numeric.zeros((lrwork,),real_t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, -1, rwork, -1, iwork, liwork,  0)
        lwork = int(abs(work[0]))
        work = Numeric.zeros((lwork,), t)
        lrwork = int(rwork[0])
        rwork = Numeric.zeros((lrwork,),real_t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, lwork, rwork, lrwork, iwork, liwork,  0)
    else:
        lapack_routine = lapack_lite.dsyevd
        w = Numeric.zeros((n,), t)
        lwork = 1
        work = Numeric.zeros((lwork,),t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, -1, iwork, liwork, 0)
        lwork = int(work[0])
        work = Numeric.zeros((lwork,),t)
        results = lapack_routine('V', UPLO, n, a, n,w, work, lwork, iwork, liwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'Eigenvalues did not converge'
    return (w,a)


# Singular value decomposition

def singular_value_decomposition(a, full_matrices = 0):
    _assertRank2(a)
    n = a.shape[1]
    m = a.shape[0]
    t =_commonType(a)
    real_t = _array_type[0][_array_precision[t]]
    a = _fastCopyAndTranspose(t, a)
    if full_matrices:
        nu = m
        nvt = n
        option = 'A'
    else:
        nu = min(n,m)
        nvt = min(n,m)
        option = 'S'
    s = Numeric.zeros((min(n,m),), real_t)
    u = Numeric.zeros((nu, m), t)
    vt = Numeric.zeros((n, nvt), t)
    iwork = Numeric.zeros((8*min(m,n),), 'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgesdd
        rwork = Numeric.zeros((5*min(m,n)*min(m,n) + 5*min(m,n),), real_t)
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, lwork, rwork, iwork, 0)
    else:
        lapack_routine = lapack_lite.dgesdd
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, -1, iwork, 0)
        lwork = int(work[0])
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine(option, m, n, a, m, s, u, m, vt, nvt,
                                 work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge'
    return multiarray.transpose(u), s, multiarray.transpose(vt) # why copy here?


# Generalized inverse

def generalized_inverse(a, rcond = 1.e-10):
    a = Numeric.array(a, copy=0)
    if a.typecode() in Numeric.typecodes['Complex']:
        a = Numeric.conjugate(a)
    u, s, vt = singular_value_decomposition(a, 0)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond*Numeric.maximum.reduce(s)
    for i in range(min(n,m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.;
    return Numeric.dot(Numeric.transpose(vt),
                       s[:, Numeric.NewAxis]*Numeric.transpose(u))

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
    pivots = Numeric.zeros((n,), 'i')
    results = lapack_routine(n, n, a, n, pivots, 0)
    sign = Numeric.add.reduce(Numeric.not_equal(pivots,
                                                Numeric.arrayrange(1, n+1))) % 2
    return (1.-2.*sign)*Numeric.multiply.reduce(Numeric.diagonal(a))

# Linear Least Squares

def linear_least_squares(a, b, rcond=1.e-10):
    """solveLinearLeastSquares(a,b) returns x,resids,rank,s
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
    one_eq = len(b.shape) == 1
    if one_eq:
        b = b[:, Numeric.NewAxis]
    _assertRank2(a, b)
    m  = a.shape[0]
    n  = a.shape[1]
    n_rhs = b.shape[1]
    ldb = max(n,m)
    if m != b.shape[0]:
        raise LinAlgError, 'Incompatible dimensions'
    t =_commonType(a, b)
    real_t = _array_type[0][_array_precision[t]]
    bstar = Numeric.zeros((ldb,n_rhs),t)
    bstar[:b.shape[0],:n_rhs] = copy.copy(b)
    a,bstar = _castCopyAndTranspose(t, a, bstar)
    s = Numeric.zeros((min(m,n),),real_t)
    nlvl = max( 0, int( math.log( float(min( m,n ))/2. ) ) + 1 )
    iwork = Numeric.zeros((3*min(m,n)*nlvl+11*min(m,n),), 'i')
    if _array_kind[t] == 1: # Complex routines take different arguments
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = Numeric.zeros((lwork,), real_t)
        work = Numeric.zeros((lwork,),t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,-1,rwork,iwork,0 )
        lwork = int(abs(work[0]))
        rwork = Numeric.zeros((lwork,),real_t)
        a_real = Numeric.zeros((m,n),real_t)
        bstar_real = Numeric.zeros((ldb,n_rhs,),real_t)
        results = lapack_lite.dgelsd( m, n, n_rhs, a_real, m, bstar_real,ldb , s, rcond,
                        0,rwork,-1,iwork,0 )
        lrwork = int(rwork[0])
        work = Numeric.zeros((lwork,), t)
        rwork = Numeric.zeros((lrwork,), real_t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,lwork,rwork,iwork,0 )
    else:
        lapack_routine = lapack_lite.dgelsd
        lwork = 1
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,-1,iwork,0 )
        lwork = int(work[0])
        work = Numeric.zeros((lwork,), t)
        results = lapack_routine( m, n, n_rhs, a, m, bstar,ldb , s, rcond,
                        0,work,lwork,iwork,0 )
    if results['info'] > 0:
        raise LinAlgError, 'SVD did not converge in Linear Least Squares'
    resids = Numeric.array([],t)
    if one_eq:
        x = copy.copy(Numeric.ravel(bstar)[:n])
        if (results['rank']==n) and (m>n):
            resids = Numeric.array([Numeric.sum((Numeric.ravel(bstar)[n:])**2)])
    else:
        x = copy.copy(Numeric.transpose(bstar)[:n,:])
        if (results['rank']==n) and (m>n):
            resids = copy.copy(Numeric.sum((Numeric.transpose(bstar)[n:,:])**2))
    return x,resids,results['rank'],copy.copy(s[:min(n,m)])


if __name__ == '__main__':
    from Numeric import *

    def test(a, b):

        print "All numbers printed should be (almost) zero:"

        x = solve_linear_equations(a, b)
        check = b - matrixmultiply(a, x)
        print check


        a_inv = inverse(a)
        check = matrixmultiply(a, a_inv)-identity(a.shape[0])
        print check


        ev = eigenvalues(a)

        evalues, evectors = eigenvectors(a)
        check = ev-evalues
        print check

        evectors = transpose(evectors)
        check = matrixmultiply(a, evectors)-evectors*evalues
        print check


        u, s, vt = singular_value_decomposition(a)
        check = a - Numeric.matrixmultiply(u*s, vt)
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
