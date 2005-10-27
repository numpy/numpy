import types
import math, operator
import numeric as _nx
from numeric import ones, zeros, arange, concatenate, array, asarray, empty
from numeric import ScalarType
from umath import pi, multiply, add, arctan2, maximum, minimum, frompyfunc, \
     isnan, absolute
from oldnumeric import ravel, nonzero, choose, \
     sometrue, alltrue, reshape, any, all, typecodes, ArrayType
from type_check import ScalarType, isscalar
from shape_base import squeeze, atleast_1d
from _compiled_base import digitize, bincount, _insert
from ufunclike import sign

__all__ = ['logspace', 'linspace', 'round_',
           'select', 'piecewise', 'trim_zeros', 'alen', 'amax', 'amin', 'ptp',
           'copy', 'iterable', 'base_repr', 'binary_repr', 'prod', 'cumprod',
           'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'disp',
           'unique', 'extract', 'insert', 'nansum', 'nanmax', 'nanargmax',
           'nanargmin', 'nanmin', 'vectorize', 'asarray_chkfinite', 'average',
           'histogram', 'bincount', 'digitize']

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

    This is equivalent to using base_repr with base 2, but about 25x
    faster.
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



def logspace(start, stop, num=50, endpoint=True):
    """ Return evenly spaced samples on a logarithmic scale.

        Return 'num' evenly spaced samples from 10**start to 10**stop.
        If 'endpoint' is True then the last sample is 10**stop.
    """
    if num <= 0: return array([])
    if endpoint:
        step = (stop-start)/float((num-1))
        y = _nx.arange(0, num) * step + start
    else:
        step = (stop-start)/float(num)
        y = _nx.arange(0, num) * step + start
    return _nx.power(10.0, y)

def linspace(start, stop, num=50, endpoint=True, retstep=False):
    """ Return 'num' evenly spaced samples from 'start' to 'stop'.  If
        'endpoint' is True, the last sample is 'stop'. If 'retstep' is
        True then return the step value used.
    """
    if num <= 0: return array([])
    if endpoint:
        step = (stop-start)/float((num-1))
        y = _nx.arange(0, num) * step + start        
    else:
        step = (stop-start)/float(num)
        y = _nx.arange(0, num) * step + start
    if retstep:
        return y, step
    else:
        return y

def iterable(y):
    try: iter(y)
    except: return 0
    return 1

def histogram(a, bins=10, range=None, normed=False):
    a = asarray(a).ravel()
    if not iterable(bins):
        if range is None:
            range = (a.min(), a.max())
        mn, mx = [a+0.0 for a in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins)

    n = a.sort().searchsorted(bins)
    n = concatenate([n, [len(a)]])
    n = n[1:]-n[:-1]

    if normed:
        db = bins[1] - bins[0]
        return 1.0/(a.size*db) * n, bins
    else:
        return n, bins

def average(a, axis=0, weights=None, returned=False):
    """average(a, axis=0, weights=None, returned=False)

    Average the array over the given axis.  If the axis is None, average
    over all dimensions of the array.  Equivalent to a.mean(axis), but
    with a default axis of 0 instead of None.

    If an integer axis is given, this equals:
        a.sum(axis) * 1.0 / len(a)
    
    If axis is None, this equals:
        a.sum(axis) * 1.0 / product(a.shape)

    If weights are given, result is:
        sum(a * weights) / sum(weights),
    where the weights must have a's shape or be 1D with length the 
    size of a in the given axis. Integer weights are converted to 
    Float.  Not specifying weights is equivalent to specifying 
    weights that are all 1.

    If 'returned' is True, return a tuple: the result and the sum of
    the weights or count of values. The shape of these two results
    will be the same.

    Raises ZeroDivisionError if appropriate.  (The version in MA does
    not -- it returns masked values).
    """
    if axis is None:
        a = array(a).ravel()
        if weights is None:
            n = add.reduce(a)
            d = len(a) * 1.0
        else:
            w = array(weights).ravel() * 1.0
            n = add.reduce(multiply(a, w))
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
            w = array(weights, copy=False) * 1.0
            wsh = w.shape
            if wsh == ():
                wsh = (1,)
            if wsh == ash:
                n = add.reduce(a*w, axis)
                d = add.reduce(w, axis) 
            elif wsh == (ash[axis],):
                ni = ash[axis]
                r = [newaxis]*ni
                r[axis] = slice(None, None, 1)
                w1 = eval("w["+repr(tuple(r))+"]*ones(ash, Float)")
                n = add.reduce(a*w1, axis)
                d = add.reduce(w1, axis)
            else:
                raise ValueError, 'averaging weights have wrong shape'
            
    if not isinstance(d, ArrayType):
        if d == 0.0: 
            raise ZeroDivisionError, 'zero denominator in average()'
    if returned:
        return n/d, d
    else:
        return n/d


def isaltered():
    val = str(type(_nx.array([1])))
    return 'scipy' in val


def asarray_chkfinite(a):
    """Like asarray, but check that no NaNs or Infs are present.
    """
    a = asarray(a)
    if (a.dtypechar in _nx.typecodes['AllFloat']) \
           and (_nx.isnan(a).any() or _nx.isinf(a).any()):
        raise ValueError, "array must not contain infs or NaNs"
    return a    




def piecewise(x, condlist, funclist, *args, **kw):
    """Return a piecewise-defined function.

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
        for k in range(1, n):
            totlist |= condlist
        condlist.append(~totlist)
        n += 1
    if (n != n2):
        raise ValueError, "function list and condition list must be the same"
    y = empty(x.shape, x.dtype)
    for k in range(n):
        item = funclist[k]
        if not callable(item):
            y[condlist[k]] = item
        else:
            y[condlist[k]] = item(x[condlist[k]], *args, **kw)
    return y

def select(condlist, choicelist, default=0):
    """ Return an array composed of different elements of choicelist
        depending on the list of conditions.

        condlist is a list of condition arrays containing ones or zeros
    
        choicelist is a list of choice arrays (of the "same" size as the
        arrays in condlist).  The result array has the "same" size as the
        arrays in choicelist.  If condlist is [c0, ..., cN-1] then choicelist
        must be of length N.  The elements of the choicelist can then be
        represented as [v0, ..., vN-1]. The default choice if none of the
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
    
        Note that one of the condition arrays must be large enough to handle
        the largest array in the choice list.
    """
    n = len(condlist)
    n2 = len(choicelist)
    if n2 != n:
        raise ValueError, "list of cases must be same length as list of conditions"
    choicelist.insert(0, default)    
    S = 0
    pfac = 1
    for k in range(1, n+1):
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

def _asarray1d(arr, copy=False):
    """Ensure 1D array for one array.
    """
    if copy:
        return asarray(arr).flatten()
    else:
        return asarray(arr).ravel()

def copy(a):
    """Return an array copy of the given object.
    """
    return array(a, copy=True)
    
# Basic operations
def amax(a, axis=-1): 
    """Return the maximum of 'a' along dimension axis. 
    """
    return asarray(a).max(axis)

def amin(a, axis=-1):
    """Return the minimum of a along dimension axis.
    """
    return asarray(a).min(axis)

def alen(a):
    """Return the length of a Python object interpreted as an array
    """
    return len(asarray(a))

def ptp(a, axis=-1):
    """Return maximum - minimum along the the given dimension
    """
    return asarray(a).ptp(axis)

def prod(a, axis=-1):
    """Return the product of the elements along the given axis
    """
    return asarray(a).prod(axis)

def cumprod(a, axis=-1):
    """Return the cumulative product of the elments along the given axis
    """
    return asarray(a).cumprod(axis)

def gradient(f, *varargs):
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
        raise SyntaxError, "invalid number of arguments"

    # use central differences on interior and first differences on endpoints

    print dx
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N

    otype = f.dtypechar
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = zeros(f.shape, f.dtypechar)
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(2, None)
        slice3[axis] = slice(None, -2)
        # 1D equivalent -- out[1:-1] = (f[2:] - f[:-2])/2.0
        out[slice1] = (f[slice2] - f[slice3])/2.0   
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        # 1D equivalent -- out[0] = (f[1] - f[0])
        out[slice1] = (f[slice2] - f[slice3])
        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        # 1D equivalent -- out[-1] = (f[-1] - f[-2])
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
    

def diff(a, n=1, axis=-1):
    """Calculate the nth order discrete difference along given axis.
    """
    if n==0:
        return a
    if n<0:
        raise ValueError, 'order must be non-negative but got ' + `n`
    a = asarray(a)
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return diff(a[slice1]-a[slice2], n-1, axis=axis)
    else:
        return a[slice1]-a[slice2]
    
def angle(z, deg=0):
    """Return the angle of the complex argument z.
    """
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
    return arctan2(zimag, zreal) * fact

def unwrap(p, discont=pi, axis=-1):
    """Unwrap radian phase p by changing absolute jumps greater than
       'discont' to their 2*pi complement along the given axis.
    """
    p = asarray(p)
    nd = len(p.shape)
    dd = diff(p, axis=axis)
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    ddmod = mod(dd+pi, 2*pi)-pi
    _nx.putmask(ddmod, (ddmod==-pi) & (dd > 0), pi)
    ph_correct = ddmod - dd;
    _nx.putmask(ph_correct, abs(dd)<discont, 0)
    up = array(p, copy=True, typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct, axis)
    return up

def sort_complex(a):
    """ Sort 'a' as a complex array using the real part first and then
    the imaginary part if the real part is equal (the default sort order
    for complex arrays).  This function is a wrapper ensuring a complex
    return type.
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
    
def trim_zeros(filt, trim='fb'):
    """ Trim the leading and trailing zeros from a 1D array.
    
    Example:
        >>> import scipy
        >>> a = array((0, 0, 0, 1, 2, 3, 2, 1, 0))
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
    """Return unique items from a 1-dimensional sequence.
    """
    # Dictionary setting is quite fast.
    set = {}
    for item in inseq:
        set[item] = None
    return asarray(set.keys())
    
def extract(condition, arr):
    """Return the elements of ravel(arr) where ravel(condition) is True
    (in 1D).

    Equivalent to compress(ravel(condition), ravel(arr)).
    """
    return _nx.take(ravel(arr), nonzero(ravel(condition)))

def insert(arr, mask, vals):
    """Similar to putmask arr[mask] = vals but the 1D array vals has the
    same number of elements as the non-zero values of mask. Inverse of
    extract.
    """
    return _nx._insert(arr, mask, vals)

def nansum(a, axis=-1):
    """Sum the array over the given axis, treating NaNs as 0.
    """
    y = array(a)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(a)] = 0
    return y.sum(axis)

def nanmin(a, axis=-1):
    """Find the minimium over the given axis, ignoring NaNs.
    """
    y = array(a)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(a)] = _nx.inf
    return y.min(axis)

def nanargmin(a, axis=-1):
    """Find the indices of the minimium over the given axis ignoring NaNs.
    """
    y = array(a)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(a)] = _nx.inf    
    return y.argmin(axis)    

def nanmax(a, axis=-1):
    """Find the maximum over the given axis ignoring NaNs.
    """
    y = array(a)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(a)] = -_nx.inf    
    return y.max(axis)    

def nanargmax(a, axis=-1):
    """Find the maximum over the given axis ignoring NaNs.
    """
    y = array(a)
    if not issubclass(y.dtype, _nx.integer):
        y[isnan(a)] = -_nx.inf    
    return y.argmax(axis)    

def disp(mesg, device=None, linefeed=True):
    """Display a message to the given device (default is sys.stdout)
    with or without a linefeed.
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

    def myfunc(a, b):
        if a > b:
            return a-b
        else
            return a+b

    vfunc = vectorize(myfunc)

    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    """
    def __init__(self, pyfunc, otypes='', doc=None):
        try:
            fcode = pyfunc.func_code
        except AttributeError:
            raise TypeError, "object is not a callable Python object"

        self.thefunc = pyfunc
        self.ufunc = None
        self.nin = len(fcode.co_varnames)
        self.nout = None
        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc
        if isinstance(otypes, types.StringType):
            self.otypes=otypes
        else:
            raise ValueError, "output types must be a string"
        for char in self.otypes:
            if char not in typecodes['All']:
                raise ValueError, "invalid typecode specified"

    def __call__(self, *args):
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
            return tuple([x.astype(c) for x, c in zip(self.ufunc(*args), self.otypes)])


def round_(a, decimals=0):
    """Round 'a' to the given number of decimal places.  Rounding
    behaviour is equivalent to Python.

    Return 'a' if the array is not floating point.  Round both the real
    and imaginary parts separately if the array is complex.
    """
    a = asarray(a)
    if not issubclass(a.dtype, _nx.inexact):
        return a
    if issubclass(a.dtype, _nx.complexfloating):
        return round_(a.real, decimals) + 1j*round_(a.imag, decimals)
    if decimals is not 0:
        decimals = asarray(decimals)
    s = sign(a)
    if decimals is not 0:
        a = absolute(multiply(a, 10.**decimals))
    else:
        a = absolute(a)
    rem = a-asarray(a).astype(_nx.intp)
    a = _nx.where(_nx.less(rem, 0.5), _nx.floor(a), _nx.ceil(a))
    # convert back
    if decimals is not 0:
        return multiply(a, s/(10.**decimals))
    else:
        return multiply(a, s)

