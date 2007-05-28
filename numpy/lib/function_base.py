__docformat__ = "restructuredtext en"
__all__ = ['logspace', 'linspace',
           'select', 'piecewise', 'trim_zeros',
           'copy', 'iterable', #'base_repr', 'binary_repr',
           'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'disp',
           'unique', 'extract', 'place', 'nansum', 'nanmax', 'nanargmax',
           'nanargmin', 'nanmin', 'vectorize', 'asarray_chkfinite', 'average',
           'histogram', 'histogramdd', 'bincount', 'digitize', 'cov',
           'corrcoef', 'msort', 'median', 'sinc', 'hamming', 'hanning',
           'bartlett', 'blackman', 'kaiser', 'trapz', 'i0', 'add_newdoc',
           'add_docstring', 'meshgrid', 'delete', 'insert', 'append',
           'interp'
           ]

import types
import numpy.core.numeric as _nx
from numpy.core.numeric import ones, zeros, arange, concatenate, array, \
     asarray, asanyarray, empty, empty_like, asanyarray, ndarray, around
from numpy.core.numeric import ScalarType, dot, where, newaxis, intp, \
     integer, isscalar
from numpy.core.umath import pi, multiply, add, arctan2,  \
     frompyfunc, isnan, cos, less_equal, sqrt, sin, mod, exp, log10
from numpy.core.fromnumeric import ravel, nonzero, choose, sort
from numpy.core.numerictypes import typecodes
from numpy.lib.shape_base import atleast_1d, atleast_2d
from numpy.lib.twodim_base import diag
from _compiled_base import _insert, add_docstring
from _compiled_base import digitize, bincount, interp
from arraysetops import setdiff1d

#end Fernando's utilities

def linspace(start, stop, num=50, endpoint=True, retstep=False):
    """Return evenly spaced numbers.

    Return num evenly spaced samples from start to stop.  If
    endpoint is True, the last sample is stop. If retstep is
    True then return the step value used.
    """
    num = int(num)
    if num <= 0:
        return array([], float)
    if endpoint:
        if num == 1:
            return array([float(start)])
        step = (stop-start)/float((num-1))
        y = _nx.arange(0, num) * step + start
        y[-1] = stop
    else:
        step = (stop-start)/float(num)
        y = _nx.arange(0, num) * step + start
    if retstep:
        return y, step
    else:
        return y

def logspace(start,stop,num=50,endpoint=True,base=10.0):
    """Evenly spaced numbers on a logarithmic scale.

    Computes int(num) evenly spaced exponents from base**start to
    base**stop. If endpoint=True, then last number is base**stop
    """
    y = linspace(start,stop,num=num,endpoint=endpoint)
    return _nx.power(base,y)

def iterable(y):
    try: iter(y)
    except: return 0
    return 1

def histogram(a, bins=10, range=None, normed=False):
    """Compute the histogram from a set of data.

    Parameters:

        a : array
            The data to histogram. n-D arrays will be flattened.

        bins : int or sequence of floats
            If an int, then the number of equal-width bins in the given range.
            Otherwise, a sequence of the lower bound of each bin.

        range : (float, float)
            The lower and upper range of the bins. If not provided, then
            (a.min(), a.max()) is used. Values outside of this range are
            allocated to the closest bin.

        normed : bool
            If False, the result array will contain the number of samples in
            each bin.  If True, the result array is the value of the
            probability *density* function at the bin normalized such that the
            *integral* over the range is 1. Note that the sum of all of the
            histogram values will not usually be 1; it is not a probability
            *mass* function.

    Returns:

        hist : array
            The values of the histogram. See `normed` for a description of the
            possible semantics.

        lower_edges : float array
            The lower edges of each bin.

    SeeAlso:

        histogramdd

    """
    a = asarray(a).ravel()
    if not iterable(bins):
        if range is None:
            range = (a.min(), a.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins, endpoint=False)

    # best block size probably depends on processor cache size
    block = 65536
    n = sort(a[:block]).searchsorted(bins)
    for i in xrange(block, a.size, block):
        n += sort(a[i:i+block]).searchsorted(bins)
    n = concatenate([n, [len(a)]])
    n = n[1:]-n[:-1]

    if normed:
        db = bins[1] - bins[0]
        return 1.0/(a.size*db) * n, bins
    else:
        return n, bins

def histogramdd(sample, bins=10, range=None, normed=False, weights=None):
    """histogramdd(sample, bins=10, range=None, normed=False, weights=None)

    Return the N-dimensional histogram of the sample.

    Parameters:

        sample : sequence or array
            A sequence containing N arrays or an NxM array. Input data.

        bins : sequence or scalar
            A sequence of edge arrays, a sequence of bin counts, or a scalar
            which is the bin count for all dimensions. Default is 10.

        range : sequence
            A sequence of lower and upper bin edges. Default is [min, max].

        normed : boolean
            If False, return the number of samples in each bin, if True,
            returns the density.

        weights : array
            Array of weights.  The weights are normed only if normed is True.
            Should the sum of the weights not equal N, the total bin count will
            not be equal to the number of samples.

    Returns:

        hist : array
            Histogram array.

        edges : list
            List of arrays defining the lower bin edges.

    SeeAlso:

        histogram

    Example

        >>> x = random.randn(100,3)
        >>> hist3d, edges = histogramdd(x, bins = (5, 6, 7))

    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = atleast_2d(sample).T
        N, D = sample.shape

    nbin = empty(D, int)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        weights = asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise AttributeError, 'The dimension of bins must be a equal to the dimension of the sample x.'
    except TypeError:
        bins = D*[bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = atleast_1d(array(sample.min(0), float))
        smax = atleast_1d(array(sample.max(0), float))
    else:
        smin = zeros(D)
        smax = zeros(D)
        for i in arange(D):
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in arange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in arange(D):
        if isscalar(bins[i]):
            nbin[i] = bins[i] + 2 # +2 for outlier bins
            edges[i] = linspace(smin[i], smax[i], nbin[i]-1)
        else:
            edges[i] = asarray(bins[i], float)
            nbin[i] = len(edges[i])+1  # +1 for outlier bins
        dedges[i] = diff(edges[i])

    nbin =  asarray(nbin)

    # Compute the bin number each sample falls into.
    Ncount = {}
    for i in arange(D):
        Ncount[i] = digitize(sample[:,i], edges[i])

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    outliers = zeros(N, int)
    for i in arange(D):
        # Rounding precision
        decimal = int(-log10(dedges[i].min())) +6
        # Find which points are on the rightmost edge.
        on_edge = where(around(sample[:,i], decimal) == around(edges[i][-1], decimal))[0]
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Flattened histogram matrix (1D)
    hist = zeros(nbin.prod(), float)

    # Compute the sample indices in the flattened histogram matrix.
    ni = nbin.argsort()
    shape = []
    xy = zeros(N, int)
    for i in arange(0, D-1):
        xy += Ncount[ni[i]] * nbin[ni[i+1:]].prod()
    xy += Ncount[ni[-1]]

    # Compute the number of repetitions in xy and assign it to the flattened histmat.
    if len(xy) == 0:
        return zeros(nbin-2, int), edges

    flatcount = bincount(xy, weights)
    a = arange(len(flatcount))
    hist[a] = flatcount

    # Shape into a proper matrix
    hist = hist.reshape(sort(nbin))
    for i in arange(nbin.size):
        j = ni[i]
        hist = hist.swapaxes(i,j)
        ni[i],ni[j] = ni[j],ni[i]

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D*[slice(1,-1)]
    hist = hist[core]

    # Normalize if normed is True
    if normed:
        s = hist.sum()
        for i in arange(D):
            shape = ones(D, int)
            shape[i] = nbin[i]-2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    return hist, edges


def average(a, axis=None, weights=None, returned=False):
    """Average the array over the given axis.  If the axis is None,
    average over all dimensions of the array.  Equivalent to
    a.mean(axis) and to

      a.sum(axis) / size(a, axis)

    If weights are given, result is:
        sum(a * weights,axis) / sum(weights,axis),
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
                d = ones(n.shape) * d
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
                w1 = eval("w["+repr(tuple(r))+"]*ones(ash, float)")
                n = add.reduce(a*w1, axis)
                d = add.reduce(w1, axis)
            else:
                raise ValueError, 'averaging weights have wrong shape'

    if not isinstance(d, ndarray):
        if d == 0.0:
            raise ZeroDivisionError, 'zero denominator in average()'
    if returned:
        return n/d, d
    else:
        return n/d

def asarray_chkfinite(a):
    """Like asarray, but check that no NaNs or Infs are present.
    """
    a = asarray(a)
    if (a.dtype.char in typecodes['AllFloat']) \
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
      A constant may be used in funclist for a function that returns a
      constant (e.g. val  and lambda x: val are equivalent in a funclist).

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
    x = asanyarray(x)
    n2 = len(funclist)
    if not isinstance(condlist, type([])):
        condlist = [condlist]
    n = len(condlist)
    if n == n2-1:  # compute the "otherwise" condition.
        totlist = condlist[0]
        for k in range(1, n):
            totlist |= condlist[k]
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
    choicelist = [default] + choicelist
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
        if type(S) in ScalarType:
            S = S*ones(asarray(pfac).shape, type(S))
        else:
            S = S*ones(asarray(pfac).shape, S.dtype)
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
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == N:
        dx = list(varargs)
    else:
        raise SyntaxError, "invalid number of arguments"

    # use central differences on interior and first differences on endpoints

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = zeros(f.shape, f.dtype.char)
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
    if n == 0:
        return a
    if n < 0:
        raise ValueError, 'order must be non-negative but got ' + repr(n)
    a = asanyarray(a)
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

try:
    add_docstring(digitize,
r"""digitize(x,bins)

Return the index of the bin to which each value of x belongs.

Each index i returned is such that bins[i-1] <= x < bins[i] if
bins is monotonically increasing, or bins [i-1] > x >= bins[i] if
bins is monotonically decreasing.

Beyond the bounds of the bins 0 or len(bins) is returned as appropriate.

""")
except RuntimeError:
    pass

try:
    add_docstring(bincount,
r"""bincount(x,weights=None)

Return the number of occurrences of each value in x.

x must be a list of non-negative integers.  The output, b[i],
represents the number of times that i is found in x.  If weights
is specified, every occurrence of i at a position p contributes
weights[p] instead of 1.

See also: histogram, digitize, unique.

""")
except RuntimeError:
    pass

try:
    add_docstring(add_docstring,
r"""docstring(obj, docstring)

Add a docstring to a built-in obj if possible.
If the obj already has a docstring raise a RuntimeError
If this routine does not know how to add a docstring to the object
raise a TypeError

""")
except RuntimeError:
    pass

try:
    add_docstring(interp,
r"""interp(x, xp, fp, left=None, right=None)

Return the value of a piecewise-linear function at each value in x.

The piecewise-linear function, f, is defined by the known data-points fp=f(xp).
The xp points must be sorted in increasing order but this is not checked.

For values of x < xp[0] return the value given by left.  If left is None, then
return fp[0].
For values of x > xp[-1] return the value given by right. If right is None, then
return fp[-1].
"""
                  )
except RuntimeError:
    pass


def angle(z, deg=0):
    """Return the angle of the complex argument z.
    """
    if deg:
        fact = 180/pi
    else:
        fact = 1.0
    z = asarray(z)
    if (issubclass(z.dtype.type, _nx.complexfloating)):
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
    up = array(p, copy=True, dtype='d')
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up

def sort_complex(a):
    """ Sort 'a' as a complex array using the real part first and then
    the imaginary part if the real part is equal (the default sort order
    for complex arrays).  This function is a wrapper ensuring a complex
    return type.

    """
    b = array(a,copy=True)
    b.sort()
    if not issubclass(b.dtype.type, _nx.complexfloating):
        if b.dtype.char in 'bhBH':
            return b.astype('F')
        elif b.dtype.char == 'g':
            return b.astype('G')
        else:
            return b.astype('D')
    else:
        return b

def trim_zeros(filt, trim='fb'):
    """ Trim the leading and trailing zeros from a 1D array.

    Example:
        >>> import numpy
        >>> a = array((0, 0, 0, 1, 2, 3, 2, 1, 0))
        >>> numpy.trim_zeros(a)
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

import sys
if sys.hexversion < 0x2040000:
    from sets import Set as set

def unique(x):
    """Return sorted unique items from an array or sequence.

    Example:
    >>> unique([5,2,4,0,4,4,2,2,1])
    array([0, 1, 2, 4, 5])

    """
    try:
        tmp = x.flatten()
        if tmp.size == 0:
            return tmp
        tmp.sort()
        idx = concatenate(([True],tmp[1:]!=tmp[:-1]))
        return tmp[idx]
    except AttributeError:
        items = list(set(x))
        items.sort()
        return asarray(items)

def extract(condition, arr):
    """Return the elements of ravel(arr) where ravel(condition) is True
    (in 1D).

    Equivalent to compress(ravel(condition), ravel(arr)).
    """
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])

def place(arr, mask, vals):
    """Similar to putmask arr[mask] = vals but the 1D array vals has the
    same number of elements as the non-zero values of mask. Inverse of
    extract.

    """
    return _insert(arr, mask, vals)

def nansum(a, axis=None):
    """Sum the array over the given axis, treating NaNs as 0.
    """
    y = array(a,subok=True)
    if not issubclass(y.dtype.type, _nx.integer):
        y[isnan(a)] = 0
    return y.sum(axis)

def nanmin(a, axis=None):
    """Find the minimium over the given axis, ignoring NaNs.
    """
    y = array(a,subok=True)
    if not issubclass(y.dtype.type, _nx.integer):
        y[isnan(a)] = _nx.inf
    return y.min(axis)

def nanargmin(a, axis=None):
    """Find the indices of the minimium over the given axis ignoring NaNs.
    """
    y = array(a, subok=True)
    if not issubclass(y.dtype.type, _nx.integer):
        y[isnan(a)] = _nx.inf
    return y.argmin(axis)

def nanmax(a, axis=None):
    """Find the maximum over the given axis ignoring NaNs.
    """
    y = array(a, subok=True)
    if not issubclass(y.dtype.type, _nx.integer):
        y[isnan(a)] = -_nx.inf
    return y.max(axis)

def nanargmax(a, axis=None):
    """Find the maximum over the given axis ignoring NaNs.
    """
    y = array(a,subok=True)
    if not issubclass(y.dtype.type, _nx.integer):
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

# return number of input arguments and
#  number of default arguments
import re
def _get_nargs(obj):
    if not callable(obj):
        raise TypeError, "Object is not callable."
    if hasattr(obj,'func_code'):
        fcode = obj.func_code
        nargs = fcode.co_argcount
        if obj.func_defaults is not None:
            ndefaults = len(obj.func_defaults)
        else:
            ndefaults = 0
        if isinstance(obj, types.MethodType):
            nargs -= 1
        return nargs, ndefaults
    terr = re.compile(r'.*? takes exactly (?P<exargs>\d+) argument(s|) \((?P<gargs>\d+) given\)')
    try:
        obj()
        return 0, 0
    except TypeError, msg:
        m = terr.match(str(msg))
        if m:
            nargs = int(m.group('exargs'))
            ndefaults = int(m.group('gargs'))
            if isinstance(obj, types.MethodType):
                nargs -= 1
            return nargs, ndefaults
    raise ValueError, 'failed to determine the number of arguments for %s' % (obj)


class vectorize(object):
    """
 vectorize(somefunction, otypes=None, doc=None)
 Generalized Function class.

  Description:

    Define a vectorized function which takes nested sequence
    of objects or numpy arrays as inputs and returns a
    numpy array as output, evaluating the function over successive
    tuples of the input arrays like the python map function except it uses
    the broadcasting rules of numpy.

    Data-type of output of vectorized is determined by calling the function
    with the first element of the input.  This can be avoided by specifying
    the otypes argument as either a string of typecode characters or a list
    of data-types specifiers.  There should be one data-type specifier for
    each output.

  Input:

    somefunction -- a Python function or method

  Example:

    >>> def myfunc(a, b):
    ...    if a > b:
    ...        return a-b
    ...    else:
    ...        return a+b

    >>> vfunc = vectorize(myfunc)

    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    """
    def __init__(self, pyfunc, otypes='', doc=None):
        self.thefunc = pyfunc
        self.ufunc = None
        nin, ndefault = _get_nargs(pyfunc)
        if nin == 0 and ndefault == 0:
            self.nin = None
            self.nin_wo_defaults = None
        else:
            self.nin = nin
            self.nin_wo_defaults = nin - ndefault
        self.nout = None
        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc
        if isinstance(otypes, types.StringType):
            self.otypes = otypes
            for char in self.otypes:
                if char not in typecodes['All']:
                    raise ValueError, "invalid otype specified"
        elif iterable(otypes):
            self.otypes = ''.join([_nx.dtype(x).char for x in otypes])
        else:
            raise ValueError, "output types must be a string of typecode characters or a list of data-types"
        self.lastcallargs = 0

    def __call__(self, *args):
        # get number of outputs and output types by calling
        #  the function on the first entries of args
        nargs = len(args)
        if self.nin:
            if (nargs > self.nin) or (nargs < self.nin_wo_defaults):
                raise ValueError, "mismatch between python function inputs"\
                      " and received arguments"

        if (self.lastcallargs != nargs):
            self.lastcallargs = nargs
            self.ufunc = None
            self.nout = None

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
                    otypes.append(asarray(theout[k]).dtype.char)
                self.otypes = ''.join(otypes)

        if (self.ufunc is None):
            self.ufunc = frompyfunc(self.thefunc, nargs, self.nout)

        if self.nout == 1:
            _res = array(self.ufunc(*args),copy=False).astype(self.otypes[0])
        else:
            _res = tuple([array(x,copy=False).astype(c) \
                          for x, c in zip(self.ufunc(*args), self.otypes)])
        return _res

def cov(m, y=None, rowvar=1, bias=0):
    """Estimate the covariance matrix.

    If m is a vector, return the variance.  For matrices return the
    covariance matrix.

    If y is given it is treated as an additional (set of)
    variable(s).

    Normalization is by (N-1) where N is the number of observations
    (unbiased estimate).  If bias is 1 then normalization is by N.

    If rowvar is non-zero (default), then each row is a variable with
    observations in the columns, otherwise each column
    is a variable and the observations are in the rows.
    """

    X = array(m, ndmin=2, dtype=float)
    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
        tup = (slice(None),newaxis)
    else:
        axis = 1
        tup = (newaxis, slice(None))


    if y is not None:
        y = array(y, copy=False, ndmin=2, dtype=float)
        X = concatenate((X,y),axis)

    X -= X.mean(axis=1-axis)[tup]
    if rowvar:
        N = X.shape[1]
    else:
        N = X.shape[0]

    if bias:
        fact = N*1.0
    else:
        fact = N-1.0

    if not rowvar:
        return (dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (dot(X, X.T.conj()) / fact).squeeze()

def corrcoef(x, y=None, rowvar=1, bias=0):
    """The correlation coefficients
    """
    c = cov(x, y, rowvar, bias)
    try:
        d = diag(c)
    except ValueError: # scalar covariance
        return 1
    return c/sqrt(multiply.outer(d,d))

def blackman(M):
    """blackman(M) returns the M-point Blackman window.
    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0,M)
    return 0.42-0.5*cos(2.0*pi*n/(M-1)) + 0.08*cos(4.0*pi*n/(M-1))

def bartlett(M):
    """bartlett(M) returns the M-point Bartlett window.
    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0,M)
    return where(less_equal(n,(M-1)/2.0),2.0*n/(M-1),2.0-2.0*n/(M-1))

def hanning(M):
    """hanning(M) returns the M-point Hanning window.
    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0,M)
    return 0.5-0.5*cos(2.0*pi*n/(M-1))

def hamming(M):
    """hamming(M) returns the M-point Hamming window.
    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1,float)
    n = arange(0,M)
    return 0.54-0.46*cos(2.0*pi*n/(M-1))

## Code from cephes for i0

_i0A = [
-4.41534164647933937950E-18,
 3.33079451882223809783E-17,
-2.43127984654795469359E-16,
 1.71539128555513303061E-15,
-1.16853328779934516808E-14,
 7.67618549860493561688E-14,
-4.85644678311192946090E-13,
 2.95505266312963983461E-12,
-1.72682629144155570723E-11,
 9.67580903537323691224E-11,
-5.18979560163526290666E-10,
 2.65982372468238665035E-9,
-1.30002500998624804212E-8,
 6.04699502254191894932E-8,
-2.67079385394061173391E-7,
 1.11738753912010371815E-6,
-4.41673835845875056359E-6,
 1.64484480707288970893E-5,
-5.75419501008210370398E-5,
 1.88502885095841655729E-4,
-5.76375574538582365885E-4,
 1.63947561694133579842E-3,
-4.32430999505057594430E-3,
 1.05464603945949983183E-2,
-2.37374148058994688156E-2,
 4.93052842396707084878E-2,
-9.49010970480476444210E-2,
 1.71620901522208775349E-1,
-3.04682672343198398683E-1,
 6.76795274409476084995E-1]

_i0B = [
-7.23318048787475395456E-18,
-4.83050448594418207126E-18,
 4.46562142029675999901E-17,
 3.46122286769746109310E-17,
-2.82762398051658348494E-16,
-3.42548561967721913462E-16,
 1.77256013305652638360E-15,
 3.81168066935262242075E-15,
-9.55484669882830764870E-15,
-4.15056934728722208663E-14,
 1.54008621752140982691E-14,
 3.85277838274214270114E-13,
 7.18012445138366623367E-13,
-1.79417853150680611778E-12,
-1.32158118404477131188E-11,
-3.14991652796324136454E-11,
 1.18891471078464383424E-11,
 4.94060238822496958910E-10,
 3.39623202570838634515E-9,
 2.26666899049817806459E-8,
 2.04891858946906374183E-7,
 2.89137052083475648297E-6,
 6.88975834691682398426E-5,
 3.36911647825569408990E-3,
 8.04490411014108831608E-1]

def _chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0

    for i in xrange(1,len(vals)):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + vals[i]

    return 0.5*(b0 - b2)

def _i0_1(x):
    return exp(x) * _chbevl(x/2.0-2, _i0A)

def _i0_2(x):
    return exp(x) * _chbevl(32.0/x - 2.0, _i0B) / sqrt(x)

def i0(x):
    x = atleast_1d(x).copy()
    y = empty_like(x)
    ind = (x<0)
    x[ind] = -x[ind]
    ind = (x<=8.0)
    y[ind] = _i0_1(x[ind])
    ind2 = ~ind
    y[ind2] = _i0_2(x[ind2])
    return y.squeeze()

## End of cephes code for i0

def kaiser(M,beta):
    """kaiser(M, beta) returns a Kaiser window of length M with shape parameter
    beta.
    """
    from numpy.dual import i0
    n = arange(0,M)
    alpha = (M-1)/2.0
    return i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/i0(beta)

def sinc(x):
    """sinc(x) returns sin(pi*x)/(pi*x) at all points of array x.
    """
    y = pi* where(x == 0, 1.0e-20, x)
    return sin(y)/y

def msort(a):
    b = array(a,subok=True,copy=True)
    b.sort(0)
    return b

def median(m):
    """median(m) returns a median of m along the first dimension of m.
    """
    sorted = msort(m)
    index = int(sorted.shape[0]/2)
    if sorted.shape[0] % 2 == 1:
        return sorted[index]
    else:
        return (sorted[index-1]+sorted[index])/2.0

def trapz(y, x=None, dx=1.0, axis=-1):
    """Integrate y(x) using samples along the given axis and the composite
    trapezoidal rule.  If x is None, spacing given by dx is assumed.
    """
    y = asarray(y)
    if x is None:
        d = dx
    else:
        d = diff(x,axis=axis)
    nd = len(y.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1,None)
    slice2[axis] = slice(None,-1)
    return add.reduce(d * (y[slice1]+y[slice2])/2.0,axis)

#always succeed
def add_newdoc(place, obj, doc):
    """Adds documentation to obj which is in module place.

    If doc is a string add it to obj as a docstring

    If doc is a tuple, then the first element is interpreted as
       an attribute of obj and the second as the docstring
          (method, docstring)

    If doc is a list, then each element of the list should be a
       sequence of length two --> [(method1, docstring1),
       (method2, docstring2), ...]

    This routine never raises an error.
       """
    try:
        new = {}
        exec 'from %s import %s' % (place, obj) in new
        if isinstance(doc, str):
            add_docstring(new[obj], doc.strip())
        elif isinstance(doc, tuple):
            add_docstring(getattr(new[obj], doc[0]), doc[1].strip())
        elif isinstance(doc, list):
            for val in doc:
                add_docstring(getattr(new[obj], val[0]), val[1].strip())
    except:
        pass


# From matplotlib
def meshgrid(x,y):
    """
    For vectors x, y with lengths Nx=len(x) and Ny=len(y), return X, Y
    where X and Y are (Ny, Nx) shaped arrays with the elements of x
    and y repeated to fill the matrix

    EG,

      [X, Y] = meshgrid([1,2,3], [4,5,6,7])

       X =
         1   2   3
         1   2   3
         1   2   3
         1   2   3


       Y =
         4   4   4
         5   5   5
         6   6   6
         7   7   7
  """
    x = asarray(x)
    y = asarray(y)
    numRows, numCols = len(y), len(x)  # yes, reversed
    x = x.reshape(1,numCols)
    X = x.repeat(numRows, axis=0)

    y = y.reshape(numRows,1)
    Y = y.repeat(numCols, axis=1)
    return X, Y

def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted.

    Return a new array with the sub-arrays (i.e. rows or columns)
    deleted along the given axis as specified by obj

    obj may be a slice_object (s_[3:5:2]) or an integer
    or an array of integers indicated which sub-arrays to
    remove.

    If axis is None, then ravel the array first.

    Example:
    >>> arr = [[3,4,5],
    ...       [1,2,3],
    ...       [6,7,8]]

    >>> delete(arr, 1, 1)
    array([[3, 5],
           [1, 3],
           [6, 8]])
    >>> delete(arr, 1, 0)
    array([[3, 4, 5],
           [6, 7, 8]])
    """
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass


    arr = asarray(arr)
    ndim = arr.ndim
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim;
        axis = ndim-1;
    if ndim == 0:
        if wrap:
            return wrap(arr)
        else:
            return arr.copy()
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, (int, long, integer)):
        if (obj < 0): obj += N
        if (obj < 0 or obj >=N):
            raise ValueError, "invalid entry"
        newshape[axis]-=1;
        new = empty(newshape, arr.dtype, arr.flags.fnc)
        slobj[axis] = slice(None, obj)
        new[slobj] = arr[slobj]
        slobj[axis] = slice(obj,None)
        slobj2 = [slice(None)]*ndim
        slobj2[axis] = slice(obj+1,None)
        new[slobj] = arr[slobj2]
    elif isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        numtodel = len(xrange(start, stop, step))
        if numtodel <= 0:
            if wrap:
                return wrap(new)
            else:
                return arr.copy()
        newshape[axis] -= numtodel
        new = empty(newshape, arr.dtype, arr.flags.fnc)
        # copy initial chunk
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[slobj] = arr[slobj]
        # copy end chunck
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop-numtodel,None)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(stop, None)
            new[slobj] = arr[slobj2]
        # copy middle pieces
        if step == 1:
            pass
        else:  # use array indexing.
            obj = arange(start, stop, step, dtype=intp)
            all = arange(start, stop, dtype=intp)
            obj = setdiff1d(all, obj)
            slobj[axis] = slice(start, stop-numtodel)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = obj
            new[slobj] = arr[slobj2]
    else: # default behavior
        obj = array(obj, dtype=intp, copy=0, ndmin=1)
        all = arange(N, dtype=intp)
        obj = setdiff1d(all, obj)
        slobj[axis] = obj
        new = arr[slobj]
    if wrap:
        return wrap(new)
    else:
        return new

def insert(arr, obj, values, axis=None):
    """Return a new array with values inserted along the given axis
    before the given indices

    If axis is None, then ravel the array first.

    The obj argument can be an integer, a slice, or a sequence of
    integers.

    Example:
    >>> a = array([[1,2,3],
    ...            [4,5,6],
    ...            [7,8,9]])

    >>> insert(a, [1,2], [[4],[5]], axis=0)
    array([[1, 2, 3],
           [4, 4, 4],
           [4, 5, 6],
           [5, 5, 5],
           [7, 8, 9]])
    """
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass

    arr = asarray(arr)
    ndim = arr.ndim
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim
        axis = ndim-1
    if (ndim == 0):
        arr = arr.copy()
        arr[...] = values
        if wrap:
            return wrap(arr)
        else:
            return arr
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, (int, long, integer)):
        if (obj < 0): obj += N
        if obj < 0 or obj > N:
            raise ValueError, "index (%d) out of range (0<=index<=%d) "\
                  "in dimension %d" % (obj, N, axis)
        newshape[axis] += 1;
        new = empty(newshape, arr.dtype, arr.flags.fnc)
        slobj[axis] = slice(None, obj)
        new[slobj] = arr[slobj]
        slobj[axis] = obj
        new[slobj] = values
        slobj[axis] = slice(obj+1,None)
        slobj2 = [slice(None)]*ndim
        slobj2[axis] = slice(obj,None)
        new[slobj] = arr[slobj2]
        if wrap:
            return wrap(new)
        return new

    elif isinstance(obj, slice):
        # turn it into a range object
        obj = arange(*obj.indices(N),**{'dtype':intp})

    # get two sets of indices
    #  one is the indices which will hold the new stuff
    #  two is the indices where arr will be copied over

    obj = asarray(obj, dtype=intp)
    numnew = len(obj)
    index1 = obj + arange(numnew)
    index2 = setdiff1d(arange(numnew+N),index1)
    newshape[axis] += numnew
    new = empty(newshape, arr.dtype, arr.flags.fnc)
    slobj2 = [slice(None)]*ndim
    slobj[axis] = index1
    slobj2[axis] = index2
    new[slobj] = values
    new[slobj2] = arr

    if wrap:
        return wrap(new)
    return new

def append(arr, values, axis=None):
    """Append to the end of an array along axis (ravel first if None)
    """
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim-1
    return concatenate((arr, values), axis=axis)
