import types
import math, operator
import numeric as _nx
from numeric import ones, zeros, arange, concatenate, array, asarray, empty
from numeric import ScalarType
from umath import pi, multiply, add, arctan2, maximum, minimum, frompyfunc, \
     isnan
from oldnumeric import ravel, nonzero, choose, \
     sometrue, alltrue, reshape, any, all, typecodes 
from type_check import ScalarType, isscalar
from shape_base import squeeze, atleast_1d
from _compiled_base import digitize, bincount, _insert

__all__ = ['round','logspace','linspace','fix','mod',
           'select','piecewise','trim_zeros','alen','amax', 'amin', 'ptp',
           'copy', 'iterable', 'base_repr', 'binary_repr', 
           'prod','cumprod', 'diff','gradient','angle','unwrap','sort_complex',
           'disp','unique','extract','insert','nansum','nanmax','nanargmax',
           'nanargmin','nanmin', 'vectorize','asarray_chkfinite',
           'average','histogram','bincount','digitize']

_lkup = {'0':'000',
         '1':'001',
         '2':'010',
         '3':'011',
         '4':'100',
         '5':'101',
         '6':'110',
         '7':'111',
         'L':''}

def binary_repr(num):
    """Return the binary representation of the input number as a string.

    This is abuut 25x faster than using base_repr with base 2.
    """
    ostr = oct(num)
    bin = ''
    for ch in ostr[1:]:
        bin += _lkup[ch]
    ind = 0
    while bin[ind] == '0':
        ind += 1
    return bin[ind:]

def base_repr (number, base=2, padding=0):
    """Return the representation of a number in any given base.
    """
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    lnb = math.log(base)
    res = padding*chars[0]
    if number == 0:
        return res + chars[0]
    exponent = int (math.log (number)/lnb)
    while(exponent >= 0):
        term = long(base)**exponent
        lead_digit = int(number / term)
        res += chars[lead_digit]
        number -= term*lead_digit
        exponent -= 1
    return res
#end Fernando's utilities



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

def iterable(y):
    try: iter(y)
    except: return 0
    return 1

def histogram(x, bins=10, range=None, normed=0):
    x = asarray(x).ravel()
    if not iterable(bins):
        if range is None:
            range = (x.min(), x.max())
        mn, mx = [x+0.0 for x in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins)

    n = x.sort().searchsorted(bins)
    n = concatenate([n, [len(x)]])
    n = n[1:]-n[:-1]

    if normed:
        db = bins[1] - bins[0]
        return 1.0/(x.size*db) * n, bins
    else:
        return n, bins

def average (a, axis=0, weights=None, returned=0):
    """average(a, axis=0, weights=None)
       Computes average along indicated axis. 
       If axis is None, average over the entire array.
       Inputs can be integer or floating types; result is type Float.
   
       If weights are given, result is:
           sum(a*weights)/(sum(weights))
       weights must have a's shape or be the 1-d with length the size
       of a in the given axis. Integer weights are converted to Float.

       Not supplying weights is equivalent to supply weights that are
       all 1.

       If returned, return a tuple: the result and the sum of the weights 
       or count of values. The shape of these two results will be the same.

       raises ZeroDivisionError if appropriate when result is scalar.
       (The version in MA does not -- it returns masked values).
    """
    if axis is None:
        a = array(a).ravel()
        if weights is None:
            n = add.reduce(a)
            d = len(a) * 1.0
        else:
            w = array(weights).ravel() * 1.0
            n = add.reduce(a*w)
            d = add.reduce(w) 
    else:
        a = array(a)
        ash = a.shape
        if ash == ():
            a.shape = (1,)
        if weights is None:
            n = add.reduce(a, axis) 
            d = ash[axis] * 1.0
            if returned:
                d = ones(shape(n)) * d
        else:
            w = array(weights, copy=0) * 1.0
            wsh = w.shape
            if wsh == ():
                wsh = (1,)
            if wsh == ash:
                n = add.reduce(a*w, axis)
                d = add.reduce(w, axis) 
            elif wsh == (ash[axis],):
                ni = ash[axis]
                r = [newaxis]*ni
                r[axis] = slice(None,None,1)
                w1 = eval("w["+repr(tuple(r))+"]*ones(ash, Float)")
                n = add.reduce(a*w1, axis)
                d = add.reduce(w1, axis)
            else:
                raise ValueError, 'average: weights wrong shape.'
            
    if not isinstance(d, ArrayType):
        if d == 0.0: 
            raise ZeroDivisionError, 'Numeric.average, zero denominator'
    if returned:
        return n/d, d
    else:
        return n/d


def isaltered():
    val = str(type(_nx.array([1])))
    return 'scipy' in val

round = _nx.around

def asarray_chkfinite(x):
    """Like asarray except it checks to be sure no NaNs or Infs are present.
    """
    x = asarray(x)
    if (x.dtypechar in _nx.typecodes['AllFloat']) \
           and (_nx.isnan(x).any() or _nx.isinf(x).any()):
        raise ValueError, "Array must not contain infs or nans."
    return x    


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

def piecewise(x, condlist, funclist, *args, **kw):
    """Returns a piecewise-defined function.

    x is the domain

    condlist is a list of boolean arrays or a single boolean array
      The length of the condition list must be n2 or n2-1 where n2
      is the length of the function list.  If len(condlist)==n2-1, then
      an 'otherwise' condition is formed by |'ing all the conditions
      and inverting. 

    funclist is a list of functions to call of length (n2).
      Each function should return an array output for an array input
      Each function can take (the same set) of extra arguments and
      keyword arguments which are passed in after the function list.

    The output is the same shape and type as x and is found by
      calling the functions on the appropriate portions of x.

    Note: This is similar to choose or select, except
          the the functions are only evaluated on elements of x
          that satisfy the corresponding condition.

    The result is
           |--
           |  f1(x)  for condition1
     y = --|  f2(x)  for condition2
           |   ...
           |  fn(x)  for conditionn
           |--
        
    """
    n2 = len(funclist)
    if not isinstance(condlist, type([])):
        condlist = [condlist]
    n = len(condlist)
    if n == n2-1:  # compute the "otherwise" condition.
        totlist = condlist[0]
        for k in range(1,n):
            totlist |= condlist
        condlist.append(~totlist)
        n += 1
    if (n != n2):
        raise ValueError, "function list and condition list must be the same."
    y = empty(x.shape, x.dtype)
    for k in range(n):
        item = funclist[k]
        if not callable(item):
            y[condlist[k]] = item
        else:
            y[condlist[k]] = item(x[condlist[k]], *args, **kw)
    return y

def select(condlist, choicelist, default=0):
    """ Returns an array comprised from different elements of choicelist
        depending on the list of conditions.

        condlist is a list of condition arrays containing ones or zeros
    
        choicelist is a list of choice arrays (of the "same" size as the
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

def _asarray1d(arr,copy=0):
    """Ensure 1d array for one array.
    """
    if copy:
        return asarray(arr).flatten()
    else:
        return asarray(arr).ravel()

def copy(a):
    """Return an array copy of the object.
    """
    return array(a,copy=1)
    
# Basic operations
def amax(m,axis=-1): 
    """Returns the maximum of m along dimension axis. 
    """
    return asarray(m).max(axis)

def amin(m,axis=-1):
    """Returns the minimum of m along dimension axis.
    """
    return asarray(m).min(axis)

def alen(m):
    """Returns the length of a Python object interpreted as an array
    """
    return len(asarray(m))

def ptp(m,axis=-1):
    """Returns the maximum - minimum along the the given dimension
    """
    return asarray(m).ptp(axis)

def prod(m,axis=-1):
    """Returns the product of the elements along the given axis
    """
    return asarray(m).prod(axis)

def cumprod(m,axis=-1):
    """Returns the cumulative product of the elments along the given axis
    """
    return asarray(m).cumprod(axis)

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

    otype = f.dtypechar
    if otype not in ['f','d','F','D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = zeros(f.shape, f.dtypechar)
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
    x = x.ravel()
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
    if (issubclass(z.dtype, _nx.complexfloating)):
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
    """ Sort a as a complex array using real part first and then
    imaginary part if the real part is the same.
    This is the default for complex arrays and so this is a
    wrapper ensuring complex return type
    """
    b = asarray(a).sort()
    if not issubclass(b.dtype, _nx.complexfloating):
        if b.dtypechar in 'bhBH':
            return b.astype('F')
        elif b.dtypechar == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b
    
def trim_zeros(filt,trim='fb'):
    """ Trim the leading and trailing zeros from a 1D array.
    
        Example:
            >>> import scipy
            >>> a = array((0,0,0,1,2,3,2,1,0))
            >>> scipy.trim_zeros(a)
            array([1, 2, 3, 2, 1])
    """
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filt:
            if i != 0.: break
            else: first = first + 1
    last = len(filt)
    if 'B' in trim:
        for i in filt[::-1]:
            if i != 0.: break
            else: last = last - 1
    return filt[first:last]

def unique(inseq):
    """Returns unique items in 1-dimensional sequence.
    """
    # Dictionary setting is quite fast.
    set = {}
    for item in inseq:
        set[item] = None
    return asarray(set.keys())
    
def extract(condition, arr):
    """Elements of ravel(arr) where ravel(condition) is true (1-d)

    Equivalent of compress(ravel(condition), ravel(arr))
    """
    return _nx.take(ravel(arr), nonzero(ravel(condition)))

def insert(arr, mask, vals):
    """Similar to putmask arr[mask] = vals but 1d array vals has the
    same number of elements as the non-zero values of mask. Inverse of extract.
    """
    return _nx._insert(arr, mask, vals)

def nansum(x,axis=-1):
    """Sum the array over the given axis treating nans as 0
    """
    y = array(x)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(x)] = 0
    return y.sum(axis)

def nanmin(x,axis=-1):
    """Find the minimium over the given axis ignoring nans.
    """
    y = array(x)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(x)] = nx.inf
    return y.min(axis)

def nanargmin(x,axis=-1):
    """Find the indices of the minimium over the given axis ignoring nans.
    """
    y = array(x)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(x)] = nx.inf    
    return y.argmin(axis)    

def nanmax(x,axis=-1):
    """Find the maximum over the given axis ignoring nans.
    """
    y = array(x)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(x)] = -nx.inf    
    return y.max(axis)    

def nanargmax(x,axis=-1):
    """Find the maximum over the given axis ignoring nans.
    """
    y = array(x)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(x)] = -nx.inf    
    return y.argmax(axis)    

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
 vectorize(somefunction, otypes=None, doc=None)
 Generalized Function class.

  Description:
 
    Define a vectorized function which takes nested sequence
    objects or scipy arrays as inputs and returns a
    scipy array as output, evaluating the function over successive
    tuples of the input arrays like the python map function except it uses
    the broadcasting rules of scipy. 

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
    def __init__(self,pyfunc,otypes='',doc=None):
        try:
            fcode = pyfunc.func_code
        except AttributeError:
            raise TypeError, "Object is not a callable Python object"

        self.thefunc = pyfunc
        self.ufunc = None
        self.nin = len(fcode.co_varnames)
        self.nout = None
        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc
        if isinstance(otypes,types.StringType):
            self.otypes=otypes
        else:
            raise ValueError, "Output types must be a string."
        for char in self.otypes:
            if char not in typecodes['All']:
                raise ValueError, "Invalid typecode specified"

    def __call__(self,*args):
        # get number of outputs and output types by calling
        #  the function on the first entries of args
        if len(args) != self.nin:
            raise ValueError, "mismatch between python function inputs"\
                  " and received arguments"
        if self.nout is None or self.otypes == '':
            newargs = []
            for arg in args:
                newargs.append(asarray(arg).flat[0])
            theout = self.thefunc(*newargs)
            if isinstance(theout, types.TupleType):
                self.nout = len(theout)
            else:
                self.nout = 1
                theout = (theout,)
            if self.otypes == '':
                otypes = []
                for k in range(self.nout):
                    otypes.append(asarray(theout[k]).dtypechar)
                self.otypes = ''.join(otypes)

        if self.ufunc is None:
            self.ufunc = frompyfunc(self.thefunc, self.nin, self.nout)

        if self.nout == 1:
            return self.ufunc(*args).astype(self.otypes[0])
        else:
            return tuple([x.astype(c) for x,c in zip(self.ufunc(*args), self.otypes)])


            
