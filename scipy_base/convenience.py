"""Contains basic routines of common interest.  Always imported first.
   Basically MLab minus the LinearAlgebra-dependent functions.

   But max is changed to amax (array max)
    and min is changed to amin (array min)
   so that the builtin max and min are still available.
"""


__all__ = ['logspace','linspace','round','any','all','fix','mod','fftshift',
           'ifftshift','fftfreq','cont_ft','toeplitz','hankel','real','imag',
           'iscomplex','isreal','array_iscomplex','array_isreal','isposinf',
           'isneginf','nan_to_num','eye','tri','diag','fliplr','flipud',
           'rot90','tril','triu','amax','amin','ptp','cumsum','prod','cumprod',
           'diff','squeeze','sinc','angle','unwrap','real_if_close',
           'sort_complex']

import Numeric

def logspace(start,stop,num=50,endpoint=1):
    """Evenly spaced samples on a logarithmic scale.

    Return num evenly spaced samples from 10**start to 10**stop.  If
    endpoint=1 then last sample is 10**stop.
    """
    if endpoint:
        step = (stop-start)/float((num-1))
        y = Numeric.arange(0,num) * step + start
    else:
        step = (stop-start)/float(num)
        y = Numeric.arange(0,num) * step + start
    return Numeric.power(10.0,y)

def linspace(start,stop,num=50,endpoint=1,retstep=0):
    """Evenly spaced samples.
    
    Return num evenly spaced samples from start to stop.  If endpoint=1 then
    last sample is stop. If retstep is 1 then return the step value used.
    """
    if endpoint:
        step = (stop-start)/float((num-1))
        y = Numeric.arange(0,num) * step + start        
    else:
        step = (stop-start)/float(num)
        y = Numeric.arange(0,num) * step + start
    if retstep:
        return y, step
    else:
        return y

#def round(arr):
#    return Numeric.floor(arr+0.5)
round = Numeric.around
any = Numeric.sometrue
all = Numeric.alltrue

def fix(x):
    """Round x to nearest integer towards zero.
    """
    x = Numeric.asarray(x)
    y = Numeric.floor(x)
    return Numeric.where(x<0,y+1,y)

def mod(x,y):
    """x - y*floor(x/y)
    
    For numeric arrays, x % y has the same sign as x while
    mod(x,y) has the same sign as y.
    """
    return x - y*Numeric.floor(x*1.0/y)

def fftshift(x,axes=None):
    """Shift the result of an FFT operation.

    Return a shifted version of x (useful for obtaining centered spectra).
    This function swaps "half-spaces" for all axes listed (defaults to all)
    """
    ndim = len(x.shape)
    if axes == None:
        axes = range(ndim)
    y = x
    for k in axes:
        N = x.shape[k]
        p2 = int(Numeric.ceil(N/2.0))
        mylist = Numeric.concatenate((Numeric.arange(p2,N),Numeric.arange(p2)))
        y = Numeric.take(y,mylist,k)
    return y

def ifftshift(x,axes=None):
    """Reverse the effect of fftshift.
    """
    ndim = len(x.shape)
    if axes == None:
        axes = range(ndim)
    y = x
    for k in axes:
        N = x.shape[k]
        p2 = int(Numeric.floor(N/2.0))
        mylist = Numeric.concatenate((Numeric.arange(p2,N),Numeric.arange(p2)))
        y = Numeric.take(y,mylist,k)
    return y

def fftfreq(N,sample=1.0):
    """FFT sample frequencies
    
    Return the frequency bins in cycles/unit (with zero at the start) given a
    window length N and a sample spacing.
    """
    N = int(N)
    sample = float(sample)
    return Numeric.concatenate((Numeric.arange(0,(N-1)/2+1,1,'d'),Numeric.arange(-(N-1)/2,0,1,'d')))/N/sample

def cont_ft(gn,fr,delta=1.0,n=None):
    """Compute the (scaled) DFT of gn at frequencies fr.

    If the gn are alias-free samples of a continuous time function then the
    correct value for the spacing, delta, will give the properly scaled,
    continuous Fourier spectrum.

    The DFT is obtained when delta=1.0
    """
    if n is None:
        n = Numeric.arange(len(gn))
    dT = delta
    trans_kernel = Numeric.exp(-2j*Numeric.pi*fr[:,Numeric.NewAxis]*dT*n)
    return dT*Numeric.dot(trans_kernel,gn)

def toeplitz(c,r=None):
    """Construct a toeplitz matrix (i.e. a matrix with constant diagonals).

    Description:

       toeplitz(c,r) is a non-symmetric Toeplitz matrix with c as its first
       column and r as its first row.

       toeplitz(c) is a symmetric (Hermitian) Toeplitz matrix (r=c). 

    See also: hankel
    """
    if isscalar(c) or isscalar(r):
        return c   
    if r is None:
        r = c
        r[0] = Numeric.conjugate(r[0])
        c = Numeric.conjugate(c)
    r,c = map(Numeric.asarray,(r,c))
    r,c = map(Numeric.ravel,(r,c))
    rN,cN = map(len,(r,c))
    if r[0] != c[0]:
        print "Warning: column and row values don't agree; column value used."
    vals = r_[r[rN-1:0:-1], c]
    cols = grid[0:cN]
    rows = grid[rN:0:-1]
    indx = cols[:,Numeric.NewAxis]*Numeric.ones((1,rN)) + \
           rows[Numeric.NewAxis,:]*Numeric.ones((cN,1)) - 1
    return Numeric.take(vals, indx)


def hankel(c,r=None):
    """Construct a hankel matrix (i.e. matrix with constant anti-diagonals).

    Description:

      hankel(c,r) is a Hankel matrix whose first column is c and whose
      last row is r.

      hankel(c) is a square Hankel matrix whose first column is C.
      Elements below the first anti-diagonal are zero.

    See also:  toeplitz
    """
    if isscalar(c) or isscalar(r):
        return c   
    if r is None:
        r = Numeric.zeros(len(c))
    elif r[0] != c[-1]:
        print "Warning: column and row values don't agree; column value used."
    r,c = map(Numeric.asarray,(r,c))
    r,c = map(Numeric.ravel,(r,c))
    rN,cN = map(len,(r,c))
    vals = r_[c, r[1:rN]]
    cols = grid[1:cN+1]
    rows = grid[0:rN]
    indx = cols[:,Numeric.NewAxis]*Numeric.ones((1,rN)) + \
           rows[Numeric.NewAxis,:]*Numeric.ones((cN,1)) - 1
    return Numeric.take(vals, indx)


def real(val):
    aval = asarray(val)
    if aval.typecode() in ['F', 'D']:
        return aval.real
    else:
        return aval

def imag(val):
    aval = asarray(val)
    if aval.typecode() in ['F', 'D']:
        return aval.imag
    else:
        return array(0,aval.typecode())*aval

def iscomplex(x):
    return imag(x) != Numeric.zeros(asarray(x).shape)

def isreal(x):
    return imag(x) == Numeric.zeros(asarray(x).shape)

def array_iscomplex(x):
    return asarray(x).typecode() in ['F', 'D']

def array_isreal(x):
    return not asarray(x).typecode() in ['F', 'D']

def isposinf(val):
    # complex not handled currently (and potentially ambiguous)
    return Numeric.logical_and(isinf(val),val > 0)

def isneginf(val):
    # complex not handled currently (and potentially ambiguous)
    return Numeric.logical_and(isinf(val),val < 0)
    
def nan_to_num(x):
    # mapping:
    #    NaN -> 0
    #    Inf -> scipy.limits.double_max
    #   -Inf -> scipy.limits.double_min
    # complex not handled currently
    import limits
    try:
        t = x.typecode()
    except AttributeError:
        t = type(x)
    if t in [ComplexType,'F','D']:    
        y = nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    else:    
        x = Numeric.asarray(x)
        are_inf = isposinf(x)
        are_neg_inf = isneginf(x)
        are_nan = isnan(x)
        choose_array = are_neg_inf + are_nan * 2 + are_inf * 3
        y = Numeric.choose(choose_array,
                   (x,scipy.limits.double_min, 0., scipy.limits.double_max))
    return y

# These are from Numeric
from Numeric import *
import Numeric
import Matrix
from utility import isscalar
from fastumath import *


# Elementary Matrices

# zeros is from matrixmodule in C
# ones is from Numeric.py


def eye(N, M=None, k=0, typecode=None):
    """eye(N, M=N, k=0, typecode=None) returns a N-by-M matrix where the 
    k-th diagonal is all ones, and everything else is zeros.
    """
    if M is None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)

def tri(N, M=None, k=0, typecode=None):
    """tri(N, M=N, k=0, typecode=None) returns a N-by-M matrix where all
    the diagonals starting from lower left corner up to the k-th are all ones.
    """
    if M is None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)
    
# Matrix manipulation

def diag(v, k=0):
    """diag(v,k=0) returns the k-th diagonal if v is a matrix or
    returns a matrix with v as the k-th diagonal if v is a vector.
    """
    v = asarray(v)
    s = v.shape
    if len(s)==1:
        n = s[0]+abs(k)
        if k > 0:
            v = concatenate((zeros(k, v.typecode()),v))
        elif k < 0:
            v = concatenate((v,zeros(-k, v.typecode())))
        return eye(n, k=k)*v
    elif len(s)==2:
        v = add.reduce(eye(s[0], s[1], k=k)*v)
        if k > 0: return v[k:]
        elif k < 0: return v[:k]
        else: return v
    else:
            raise ValueError, "Input must be 1- or 2-D."
    
def fliplr(m):
    """fliplr(m) returns a 2-D matrix m with the rows preserved and
    columns flipped in the left/right direction.  Only works with 2-D
    arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    return m[:, ::-1]

def flipud(m):
    """flipud(m) returns a 2-D matrix with the columns preserved and
    rows flipped in the up/down direction.  Only works with 2-D arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    return m[::-1]
    
# reshape(x, m, n) is not used, instead use reshape(x, (m, n))

def rot90(m, k=1):
    """rot90(m,k=1) returns the matrix found by rotating m by k*90 degrees
    in the counterclockwise direction.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    k = k % 4
    if k == 0: return m
    elif k == 1: return transpose(fliplr(m))
    elif k == 2: return fliplr(flipud(m))
    else: return fliplr(transpose(m))  # k==3

def tril(m, k=0):
    """tril(m,k=0) returns the elements on and below the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    svsp = m.spacesaver()
    m = asarray(m,savespace=1)
    out = tri(m.shape[0], m.shape[1], k=k, typecode=m.typecode())*m
    out.savespace(svsp)
    return out

def triu(m, k=0):
    """triu(m,k=0) returns the elements on and above the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    svsp = m.spacesaver()
    m = asarray(m,savespace=1)
    out = (1-tri(m.shape[0], m.shape[1], k-1, m.typecode()))*m
    out.savespace(svsp)
    return out

# Data analysis

# Basic operations
def amax(m,axis=-1):
    """Returns the maximum of m along dimension axis. 
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return maximum.reduce(m,axis)

def amin(m,axis=-1):
    """Returns the minimum of m along dimension axis.
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:        
        m = asarray(m)
    return minimum.reduce(m,axis)

# Actually from Basis, but it fits in so naturally here...

def ptp(m,axis=-1):
    """Returns the maximum - minimum along the the given dimension
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return amax(m,axis)-amin(m,axis)

def cumsum(m,axis=-1):
    """Returns the cumulative sum of the elements along the given axis
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return add.accumulate(m,axis)

def prod(m,axis=-1):
    """Returns the product of the elements along the given axis
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return multiply.reduce(m,axis)

def cumprod(m,axis=-1):
    """Returns the cumulative product of the elments along the given axis
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return multiply.accumulate(m,axis)

def diff(x, n=1,axis=-1):
    """Calculates the nth order, discrete difference along given axis.
    """
    x = asarray(x)
    nd = len(x.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1,None)
    slice2[axis] = slice(None,-1)
    if n > 1:
        return diff(x[slice1]-x[slice2], n-1, axis=axis)
    else:
        return x[slice1]-x[slice2]

def squeeze(a):
    "Returns a with any ones from the shape of a removed"
    a = asarray(a)
    b = asarray(a.shape)
    return reshape (a, tuple (compress (not_equal (b, 1), b)))

def sinc(x):
    """Returns sin(pi*x)/(pi*x) at all points of array x.
    """
    w = asarray(x*pi)
    return where(x==0, 1.0, sin(w)/w)

def angle(z,deg=0):
    """Return the angle of complex argument z."""
    if deg:
        fact = 180/pi
    else:
        fact = 1.0
    z = asarray(z)
    if z.typecode() in ['D','F']:
       zimag = z.imag
       zreal = z.real
    else:
       zimag = 0
       zreal = z
    return arctan2(zimag,zreal) * fact

import copy
def unwrap(p,discont=pi,axis=-1):
    """unwrap(p,discont=pi,axis=-1)

    unwraps radian phase p by changing absolute jumps greater than discont to
    their 2*pi complement along the given axis.
    """
    p = asarray(p)
    nd = len(p.shape)
    dd = diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = mod(dd+pi,2*pi)-pi
    putmask(ddmod,(ddmod==-pi) & (dd > 0),pi)
    ph_correct = ddmod - dd;
    putmask(ph_correct,abs(dd)<discont,0)
    up = array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct,axis)
    return up
 


def real_if_close(a,tol=1e-13):
    a = Numeric.asarray(a)
    if a.typecode() in ['F','D'] and Numeric.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a

def sort_complex(a):
    """ Doesn't currently work for integer arrays -- only float or complex.
    """
    a = asarray(a,typecode=a.typecode().upper())
    def complex_cmp(x,y):
        res = cmp(x.real,y.real)
        if res == 0:
            res = cmp(x.imag,y.imag)
        return res
    l = a.tolist()                
    l.sort(complex_cmp)
    return array(l)




def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)



