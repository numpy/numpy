import types
import numerix as _nx
from numerix import ravel, nonzero, array, choose, ones, zeros, \
     sometrue, alltrue, reshape, alter_numeric, restore_numeric, arraymap, \
     pi, _insert, multiply, add, arctan2, maximum, minimum, any, all
from type_check import ScalarType, isscalar, asarray
from shape_base import squeeze, atleast_1d

__all__ = ['round','logspace','linspace','fix','mod',
           'select','trim_zeros','amax','amin', 'alen',
           'ptp','cumsum','take', 'copy',
           'prod','cumprod','diff','gradient','angle','unwrap','sort_complex',
           'disp','unique','extract','insert','nansum','nanmax','nanargmax',
           'nanargmin','nanmin','sum','vectorize','asarray_chkfinite',
           'alter_numeric', 'restore_numeric','isaltered']

def isaltered():
    val = str(type(_nx.array([1])))
    return 'scipy' in val

round = _nx.around

def asarray_chkfinite(x):
    """Like asarray except it checks to be sure no NaNs or Infs are present.
    """
    x = asarray(x)
    if not all(_nx.isfinite(x)):
        raise ValueError, "Array must not contain infs or nans."
    return x    

# Need this to change array type for low precision values
def sum(x,axis=0):  # could change default axis here
    x = asarray(x)
    if x.typecode() in ['1','s','b','w']:
        x = x.astype('l')
    return _nx.sum(x,axis)
    

def logspace(start,stop,num=50,endpoint=1):
    """ Evenly spaced samples on a logarithmic scale.

        Return num evenly spaced samples from 10**start to 10**stop.  If
        endpoint=1 then last sample is 10**stop.
    """
    if num <= 0: return array([])
    if endpoint:
        step = (stop-start)/float((num-1))
        y = _nx.arange(0,num) * step + start
    else:
        step = (stop-start)/float(num)
        y = _nx.arange(0,num) * step + start
    return _nx.power(10.0,y)

def linspace(start,stop,num=50,endpoint=1,retstep=0):
    """ Evenly spaced samples.
    
        Return num evenly spaced samples from start to stop.  If endpoint=1 then
        last sample is stop. If retstep is 1 then return the step value used.
    """
    if num <= 0: return array([])
    if endpoint:
        step = (stop-start)/float((num-1))
        y = _nx.arange(0,num) * step + start        
    else:
        step = (stop-start)/float(num)
        y = _nx.arange(0,num) * step + start
    if retstep:
        return y, step
    else:
        return y

def fix(x):
    """ Round x to nearest integer towards zero.
    """
    x = asarray(x)
    y = _nx.floor(x)
    return _nx.where(x<0,y+1,y)

def mod(x,y):
    """ x - y*floor(x/y)
    
        For numeric arrays, x % y has the same sign as x while
        mod(x,y) has the same sign as y.
    """
    return x - y*_nx.floor(x*1.0/y)

def select(condlist, choicelist, default=0):
    """ Returns an array comprised from different elements of choicelist
        depending on the list of conditions.

        condlist is a list of condition arrays containing ones or zeros
    
        choicelist is a list of choice matrices (of the "same" size as the
        arrays in condlist).  The result array has the "same" size as the
        arrays in choicelist.  If condlist is [c0,...,cN-1] then choicelist
        must be of length N.  The elements of the choicelist can then be
        represented as [v0,...,vN-1]. The default choice if none of the
        conditions are met is given as the default argument. 
    
        The conditions are tested in order and the first one statisfied is
        used to select the choice. In other words, the elements of the
        output array are found from the following tree (notice the order of
        the conditions matters):
    
        if c0: v0
        elif c1: v1
        elif c2: v2
        ...
        elif cN-1: vN-1
        else: default
    
        Note, that one of the condition arrays must be large enough to handle
        the largest array in the choice list.
    """
    n = len(condlist)
    n2 = len(choicelist)
    if n2 != n:
        raise ValueError, "List of cases, must be same length as the list of conditions."
    choicelist.insert(0,default)    
    S = 0
    pfac = 1
    for k in range(1,n+1):
        S += k * pfac * asarray(condlist[k-1])
        if k < n:
            pfac *= (1-asarray(condlist[k-1]))
    # handle special case of a 1-element condition but
    #  a multi-element choice
    if type(S) in ScalarType or max(asarray(S).shape)==1:
        pfac = asarray(1)
        for k in range(n2+1):
            pfac = pfac + asarray(choicelist[k])            
        S = S*ones(asarray(pfac).shape)
    return choose(S, tuple(choicelist))

def _asarray1d(arr):
    """Ensure 1d array for one array.
    """
    m = asarray(arr)
    if len(m.shape)==0:
        m = reshape(m,(1,))
    return m

def copy(a):
    """Return an array copy of the object.
    """
    return array(a,copy=1)

def take(a, indices, axis=0):
    """Selects the elements in indices from array a along given axis.
    """
    try:
        a = _nx.take(a,indices,axis)
    except ValueError:  # a is scalar
        pass
    return a

def _no_axis_is_all(function, m, axis):
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = _asarray1d(m)
    if _nx.which[0] == "numeric":
        r = function(m, axis)
    else:
        import numarray as _na
        _na.Error.pushMode(overflow="raise")
        try:
            r = function(m, axis)
        finally:
            _na.Error.popMode()
    return r
    
# Basic operations
def amax(m,axis=-1): 
    """Returns the maximum of m along dimension axis. 
    """
    return _no_axis_is_all(maximum.reduce, m, axis)

def amin(m,axis=-1):
    """Returns the minimum of m along dimension axis.
    """
    return _no_axis_is_all(minimum.reduce, m, axis)

def alen(m):
    """Returns the length of a Python object interpreted as an array
    """
    return len(asarray(m))

# Actually from Basis, but it fits in so naturally here...

def _amin_amax(m, axis):
    return amax(m,axis)-amin(m,axis)

def ptp(m,axis=-1):
    """Returns the maximum - minimum along the the given dimension
    """
    return _no_axis_is_all(_amin_amax, m, axis)

def cumsum(m,axis=-1):
    """Returns the cumulative sum of the elements along the given axis
    """
    return _no_axis_is_all(add.accumulate, m, axis)

def prod(m,axis=-1):
    """Returns the product of the elements along the given axis
    """
    return _no_axis_is_all(multiply.reduce, m, axis)

def cumprod(m,axis=-1):
    """Returns the cumulative product of the elments along the given axis
    """
    return _no_axis_is_all(multiply.accumulate, m, axis)

def gradient(f,*varargs):
    """Calculate the gradient of an N-dimensional scalar function.

    Uses central differences on the interior and first differences on boundaries
    to give the same shape.

    Inputs:

      f -- An N-dimensional array giving samples of a scalar function

      varargs -- 0, 1, or N scalars giving the sample distances in each direction

    Outputs:

      N arrays of the same shape as f giving the derivative of f with respect
       to each dimension.
    """
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n==0:
        dx = [1.0]*N
    elif n==1:
        dx = [varargs[0]]*N
    elif n==N:
        dx = list(varargs)
    else:
        raise SyntaxError, "Invalid number of arguments"

    # use central differences on interior and first differences on endpoints

    print dx
    outvals = []

    # create slice objects --- initially all are [:,:,...,:]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N

    otype = f.typecode()
    if otype not in ['f','d','F','D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = zeros(f.shape, f.typecode())
        slice1[axis] = slice(1,-1)
        slice2[axis] = slice(2,None)
        slice3[axis] = slice(None,-2)
        # 1d equivalent -- out[1:-1] = (f[2:] - f[:-2])/2.0
        out[slice1] = (f[slice2] - f[slice3])/2.0   
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        # 1d equivalent -- out[0] = (f[1] - f[0])
        out[slice1] = (f[slice2] - f[slice3])
        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        # 1d equivalent -- out[-1] = (f[-1] - f[-2])
        out[slice1] = (f[slice2] - f[slice3])
        
        # divide by step size
        outvals.append(out / dx[axis])
        
        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals
    

def diff(x, n=1,axis=-1):
    """Calculates the nth order, discrete difference along given axis.
    """
    if n==0:
        return x
    if n<0:
        raise ValueError,'Order must be non-negative but got ' + `n`
    x = _asarray1d(x)
    nd = len(x.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1,None)
    slice2[axis] = slice(None,-1)
    if n > 1:
        return diff(x[slice1]-x[slice2], n-1, axis=axis)
    else:
        return x[slice1]-x[slice2]
    
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

def unwrap(p,discont=pi,axis=-1):
    """unwraps radian phase p by changing absolute jumps greater than
       discont to their 2*pi complement along the given axis.
    """
    p = asarray(p)
    nd = len(p.shape)
    dd = diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = mod(dd+pi,2*pi)-pi
    _nx.putmask(ddmod,(ddmod==-pi) & (dd > 0),pi)
    ph_correct = ddmod - dd;
    _nx.putmask(ph_correct,abs(dd)<discont,0)
    up = array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct,axis)
    return up

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

def trim_zeros(filt,trim='fb'):
    """ Trim the leading and trailing zeros from a 1D array.
    
        Example:
            >>> import scipy
            >>> a = array((0,0,0,1,2,3,2,1,0))
            >>> scipy.trim_zeros(a)
            array([1, 2, 3, 2, 1])
    """
    first = 0
    if 'f' in trim or 'F' in trim:
        for i in filt:
            if i != 0.: break
            else: first = first + 1
    last = len(filt)
    if 'b' in trim or 'B' in trim:
        for i in filt[::-1]:
            if i != 0.: break
            else: last = last - 1
    return filt[first:last]

def unique(inseq):
    """Returns unique items in 1-dimensional sequence.
    """
    set = {}
    for item in inseq:
        set[item] = None
    return asarray(set.keys())

def where(condition,x=None,y=None):
    """If x and y are both None, then return the (1-d equivalent) indices
    where condition is true.  Otherwise, return an array shaped like
    condition with elements of x and y in the places where condition is
    true or false respectively.
    """
    if (x is None) and (y is None):
             # Needs work for multidimensional arrays
        return nonzero(ravel(condition))
    else:
        return choose(not_equal(condition, 0), (y,x))
    
def extract(condition, arr):
    """Elements of ravel(condition) where ravel(condition) is true (1-d)

    Equivalent of compress(ravel(condition), ravel(arr))
    """
    return _nx.take(ravel(arr), nonzero(ravel(condition)))

def insert(arr, mask, vals):
    """Similar to putmask arr[mask] = vals but 1d array vals has the
    same number of elements as the non-zero values of mask. Inverse of extract.
    """
    return _nx._insert(arr, mask, vals)

def nansum(x,axis=-1):
    """Sum the array over the given axis treating nans as missing values.
    """
    x = _asarray1d(x).copy()
    _nx.putmask(x,isnan(x),0)
    return _nx.sum(x,axis)

def nanmin(x,axis=-1):
    """Find the minimium over the given axis ignoring nans.
    """
    x = _asarray1d(x).copy()
    _nx.putmask(x,isnan(x),inf)
    return amin(x,axis)

def nanargmin(x,axis=-1):
    """Find the indices of the minimium over the given axis ignoring nans.
    """
    x = _asarray1d(x).copy()
    _nx.putmask(x,isnan(x),inf)
    return argmin(x,axis)
    

def nanmax(x,axis=-1):
    """Find the maximum over the given axis ignoring nans.
    """
    x = _asarray1d(x).copy()
    _nx.putmask(x,isnan(x),-inf)
    return amax(x,axis)

def nanargmax(x,axis=-1):
    """Find the maximum over the given axis ignoring nans.
    """
    x = _asarray1d(x).copy()
    _nx.putmask(x,isnan(x),-inf)
    return argmax(x,axis)

def disp(mesg, device=None, linefeed=1):
    """Display a message to device (default is sys.stdout) with(out) linefeed.
    """
    if device is None:
        import sys
        device = sys.stdout
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    device.flush()
    return

class vectorize:
    """
 vectorize(somefunction)  Generalized Function class.

  Description:
 
    Define a vectorized function which takes nested sequence
    objects or numerix arrays as inputs and returns a
    numerix array as output, evaluating the function over successive
    tuples of the input arrays like the python map function except it uses
    the broadcasting rules of numerix Python.

  Input:

    somefunction -- a Python function or method

  Example:

    def myfunc(a,b):
        if a > b:
            return a-b
        else
            return a+b

    vfunc = vectorize(myfunc)

    >>> vfunc([1,2,3,4],2)
    array([3,4,1,2])

    """
    def __init__(self,pyfunc,otypes=None,doc=None):
        if not callable(pyfunc) or type(pyfunc) is types.ClassType:
            raise TypeError, "Object is not a callable Python object."
        self.thefunc = pyfunc
        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc
        if otypes is None:
            self.otypes=''
        else:
            if isinstance(otypes,types.StringType):
                self.otypes=otypes
            else:
                raise ValueError, "Output types must be a string."

    def __call__(self,*args):
        try:
            return squeeze(arraymap(self.thefunc,args,self.otypes))
        except IndexError:
            return self.zerocall(*args)

    def zerocall(self,*args):
        # one of the args was a zeros array
        #  return zeros for each output
        #  first --- find number of outputs
        #  get it from self.otypes if possible
        #  otherwise evaluate function at 0.9
        N = len(self.otypes)
        if N==1:
            return zeros((0,),'d')
        elif N !=0:
            return (zeros((0,),'d'),)*N
        newargs = []
        args = atleast_1d(args)
        for arg in args:
            if arg.typecode() != 'O':
                newargs.append(0.9)
            else:
                newargs.append(arg[0])
        newargs = tuple(newargs)
        try:
            res = self.thefunc(*newargs)
        except:
            raise ValueError, "Zerocall is failing.  "\
                  "Try using otypes in vectorize."
        if isscalar(res):
            return zeros((0,),'d')
        else:
            return (zeros((0,),'d'),)*len(res)

