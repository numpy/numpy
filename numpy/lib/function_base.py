__docformat__ = "restructuredtext en"
__all__ = ['logspace', 'linspace',
           'select', 'piecewise', 'trim_zeros',
           'copy', 'iterable',
           'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'disp',
           'unique', 'extract', 'place', 'nansum', 'nanmax', 'nanargmax',
           'nanargmin', 'nanmin', 'vectorize', 'asarray_chkfinite', 'average',
           'histogram', 'histogramdd', 'bincount', 'digitize', 'cov',
           'corrcoef', 'msort', 'median', 'sinc', 'hamming', 'hanning',
           'bartlett', 'blackman', 'kaiser', 'trapz', 'i0', 'add_newdoc',
           'add_docstring', 'meshgrid', 'delete', 'insert', 'append',
           'interp'
           ]
import warnings

import types
import numpy.core.numeric as _nx
from numpy.core.numeric import ones, zeros, arange, concatenate, array, \
     asarray, asanyarray, empty, empty_like, ndarray, around
from numpy.core.numeric import ScalarType, dot, where, newaxis, intp, \
     integer, isscalar
from numpy.core.umath import pi, multiply, add, arctan2,  \
     frompyfunc, isnan, cos, less_equal, sqrt, sin, mod, exp, log10
from numpy.core.fromnumeric import ravel, nonzero, choose, sort, mean
from numpy.core.numerictypes import typecodes, number
from numpy.lib.shape_base import atleast_1d, atleast_2d
from numpy.lib.twodim_base import diag
from _compiled_base import _insert, add_docstring
from _compiled_base import digitize, bincount, interp as compiled_interp
from arraysetops import setdiff1d
import numpy as np

#end Fernando's utilities

def linspace(start, stop, num=50, endpoint=True, retstep=False):
    """
    Return evenly spaced numbers.

    `linspace` returns `num` evenly spaced samples, calculated over the
    interval ``[start, stop]``.  The endpoint of the interval can optionally
    be excluded.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int
        Number of samples to generate. Default is 50.
    endpoint : bool
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.

    Returns
    -------
    samples : ndarray
        `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float (only if `retstep` is True)
        Size of spacing between samples.


    See Also
    --------
    arange : Similiar to `linspace`, but uses a step size (instead of the
             number of samples).  Note that, when used with a float
             endpoint, the endpoint may or may not be included.
    logspace : Samples uniformly distributed in log space.

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
        array([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
        (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    >>> plt.plot(x2, y + 0.5, 'o')
    >>> plt.ylim([-0.5, 1])
    >>> plt.show()

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
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    Parameters
    ----------
    start : float
        ``base ** start`` is the starting value of the sequence.
    stop : float
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length ``num``) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similiar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.

    Notes
    -----
    Logspace is equivalent to the code

    >>> y = linspace(start, stop, num=num, endpoint=endpoint)
    >>> power(base, y)

    Examples
    --------
    >>> np.logspace(2.0, 3.0, num=4)
        array([  100.        ,   215.443469  ,   464.15888336,  1000.        ])
    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
        array([ 100.        ,  177.827941  ,  316.22776602,  562.34132519])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
        array([ 4.        ,  5.0396842 ,  6.34960421,  8.        ])

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
    >>> y = np.zeros(N)
    >>> plt.plot(x1, y, 'o')
    >>> plt.plot(x2, y + 0.5, 'o')
    >>> plt.ylim([-0.5, 1])
    >>> plt.show()

    """
    y = linspace(start,stop,num=num,endpoint=endpoint)
    return _nx.power(base,y)

def iterable(y):
    try: iter(y)
    except: return 0
    return 1

def histogram(a, bins=10, range=None, normed=False, weights=None, new=None):
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : array_like
        Input data.
    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a sequence,
        it defines the bin edges, including the rightmost edge, allowing
        for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. Note that with `new` set to False, values below
        the range are ignored, while those above the range are tallied
        in the rightmost bin.
    normed : bool, optional
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will often not be equal to 1; it is not a
        probability *mass* function.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in `a`
        only contributes its associated weight towards the bin count
        (instead of 1).  If `normed` is True, the weights are normalized,
        so that the integral of the density over the range remains 1.
        The `weights` keyword is only available with `new` set to True.
    new : {None, True, False}, optional
        Whether to use the new semantics for histogram:
          * None : the new behaviour is used, and a warning is printed,
          * True : the new behaviour is used and no warning is printed,
          * False : the old behaviour is used and a message is printed
              warning about future deprecation.

    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
        With ``new=False``, return the left bin edges (``length(hist)``).


    See Also
    --------
    histogramdd

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and the
    second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which *includes*
    4.

    Examples
    --------
    >>> np.histogram([1,2,1], bins=[0,1,2,3], new=True)
    (array([0, 2, 1]), array([0, 1, 2, 3]))

    """
    # Old behavior
    if new == False:
        warnings.warn("""
        The original semantics of histogram is scheduled to be
        deprecated in NumPy 1.3. The new semantics fixes
        long-standing issues with outliers handling. The main
        changes concern
        1. the definition of the bin edges,
           now including the rightmost edge, and
        2. the handling of upper outliers,
           now ignored rather than tallied in the rightmost bin.

        Please read the docstring for more information.
        """, Warning)

        a = asarray(a).ravel()

        if (range is not None):
            mn, mx = range
            if (mn > mx):
                raise AttributeError, \
                    'max must be larger than min in range parameter.'

        if not iterable(bins):
            if range is None:
                range = (a.min(), a.max())
            mn, mx = [mi+0.0 for mi in range]
            if mn == mx:
                mn -= 0.5
                mx += 0.5
            bins = linspace(mn, mx, bins, endpoint=False)
        else:
            if normed:
                raise ValueError, 'Use new=True to pass bin edges explicitly.'
                raise ValueError, 'Use new=True to pass bin edges explicitly.'
            bins = asarray(bins)
            if (np.diff(bins) < 0).any():
                raise AttributeError, 'bins must increase monotonically.'


        if weights is not None:
            raise ValueError, 'weights are only available with new=True.'

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



    # New behavior
    elif new in [True, None]:
        if new is None:
            warnings.warn("""
            The semantics of histogram has been modified in
            the current release to fix long-standing issues with
            outliers handling. The main changes concern
            1. the definition of the bin edges,
               now including the rightmost edge, and
            2. the handling of upper outliers, now ignored rather
               than tallied in the rightmost bin.
            The previous behaviour is still accessible using
            `new=False`, but is scheduled to be deprecated in the
            next release (1.3).

            *This warning will not printed in the 1.3 release.*

            Use `new=True` to bypass this warning.

            Please read the docstring for more information.
            """, Warning)
        a = asarray(a)
        if weights is not None:
            weights = asarray(weights)
            if np.any(weights.shape != a.shape):
                raise ValueError, 'weights should have the same shape as a.'
            weights = weights.ravel()
        a =  a.ravel()

        if (range is not None):
            mn, mx = range
            if (mn > mx):
                raise AttributeError, \
                    'max must be larger than min in range parameter.'

        if not iterable(bins):
            if range is None:
                range = (a.min(), a.max())
            mn, mx = [mi+0.0 for mi in range]
            if mn == mx:
                mn -= 0.5
                mx += 0.5
            bins = linspace(mn, mx, bins+1, endpoint=True)
        else:
            bins = asarray(bins)
            if (np.diff(bins) < 0).any():
                raise AttributeError, 'bins must increase monotonically.'

        # Histogram is an integer or a float array depending on the weights.
        if weights is None:
            ntype = int
        else:
            ntype = weights.dtype
        n = np.zeros(bins.shape, ntype)

        block = 65536
        if weights is None:
            for i in arange(0, len(a), block):
                sa = sort(a[i:i+block])
                n += np.r_[sa.searchsorted(bins[:-1], 'left'), \
                    sa.searchsorted(bins[-1], 'right')]
        else:
            zero = array(0, dtype=ntype)
            for i in arange(0, len(a), block):
                tmp_a = a[i:i+block]
                tmp_w = weights[i:i+block]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = np.concatenate(([zero,], sw.cumsum()))
                bin_index = np.r_[sa.searchsorted(bins[:-1], 'left'), \
                    sa.searchsorted(bins[-1], 'right')]
                n += cw[bin_index]

        n = np.diff(n)

        if normed is False:
            return n, bins
        elif normed is True:
            db = array(np.diff(bins), float)
            return n/(n*db).sum(), bins


def histogramdd(sample, bins=10, range=None, normed=False, weights=None):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : array-like
      Data to histogram passed as a sequence of D arrays of length N, or
      as an (N,D) array.
    bins : sequence or int, optional
      The bin specification:

       * A sequence of arrays describing the bin edges along each dimension.
       * The number of bins for each dimension (nx, ny, ... =bins)
       * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
      A sequence of lower and upper bin edges to be used if the edges are
      not given explicitely in `bins`. Defaults to the minimum and maximum
      values along each dimension.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin hypervolume.
    weights : array-like (N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
      Weights are normalized to 1 if normed is True. If normed is False, the
      values of the returned histogram are equal to the sum of the weights
      belonging to the samples falling into each bin.

    Returns
    -------
    H : array
      The multidimensional histogram of sample x. See normed and weights for
      the different possible semantics.
    edges : list
      A list of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1D histogram
    histogram2d: 2D histogram

    Examples
    --------
    >>> r = np.random.randn(100,3)
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5,8,4), 6, 9, 5)

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
            raise AttributeError, 'The dimension of bins must be equal ' \
                                  'to the dimension of the sample x.'
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
        on_edge = where(around(sample[:,i], decimal) == around(edges[i][-1],
                                                               decimal))[0]
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

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    if len(xy) == 0:
        return zeros(nbin-2, int), edges

    flatcount = bincount(xy, weights)
    a = arange(len(flatcount))
    hist[a] = flatcount

    # Shape into a proper matrix
    hist = hist.reshape(sort(nbin))
    for i in arange(nbin.size):
        j = ni.argsort()[i]
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

    if (hist.shape != nbin-2).any():
        raise 'Internal Shape Error'
    return hist, edges


def average(a, axis=None, weights=None, returned=False):
    """
    Return the weighted average of array over the specified axis.


    Parameters
    ----------
    a : array_like
        Data to be averaged.
    axis : {None, integer}, optional
        Axis along which to average `a`. If `None`, averaging is done over the
        entire array irrespective of its shape.
    weights : {None, array_like}, optional
        The importance that each datum has in the computation of the average.
        The weights array can either be 1D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
    returned : {False, boolean}, optional
        If `True`, the tuple (`average`, `sum_of_weights`) is returned,
        otherwise only the average is returned. Note that if `weights=None`,
        `sum_of_weights` is equivalent to the number of elements over which
        the average is taken.

    Returns
    -------
    average, [sum_of_weights] : {array_type, double}
        Return the average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `Float`
        if `a` is of integer type, otherwise it is of the same type as `a`.
        `sum_of_weights` is of the same type as `average`.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When the length of 1D `weights` is not the same as the shape of `a`
        along axis.

    See Also
    --------
    ma.average : average for masked arrays

    Examples
    --------
    >>> data = range(1,5)
    >>> data
    [1, 2, 3, 4]
    >>> np.average(data)
    2.5
    >>> np.average(range(1,11), weights=range(10,0,-1))
    4.0

    """
    if not isinstance(a, np.matrix) :
        a = np.asarray(a)

    if weights is None :
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size/avg.size)
    else :
        a = a + 0.0
        wgt = np.array(weights, dtype=a.dtype, copy=0)

        # Sanity checks
        if a.shape != wgt.shape :
            if axis is None :
                raise TypeError, "Axis must be specified when shapes of a and weights differ."
            if wgt.ndim != 1 :
                raise TypeError, "1D weights expected when shapes of a and weights differ."
            if wgt.shape[0] != a.shape[axis] :
                raise ValueError, "Length of weights not compatible with specified axis."

            # setup wgt to broadcast along axis
            wgt = np.array(wgt, copy=0, ndmin=a.ndim).swapaxes(-1,axis)

        scl = wgt.sum(axis=axis)
        if (scl == 0.0).any():
            raise ZeroDivisionError, "Weights sum to zero, can't be normalized"

        avg = np.multiply(a,wgt).sum(axis)/scl

    if returned:
        scl = np.multiply(avg,0) + scl
        return avg, scl
    else:
        return avg

def asarray_chkfinite(a):
    """Like asarray, but check that no NaNs or Infs are present.
    """
    a = asarray(a)
    if (a.dtype.char in typecodes['AllFloat']) \
           and (_nx.isnan(a).any() or _nx.isinf(a).any()):
        raise ValueError, "array must not contain infs or NaNs"
    return a

def piecewise(x, condlist, funclist, *args, **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    x : (N,) ndarray
        The input domain.
    condlist : list of M (N,)-shaped boolean arrays
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x)` is used as the output value.

        Each boolean array in `condlist` selects a piece of `x`,
        and should therefore be of the same shape as `x`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if the length of `funclist` is
        M+1, then that extra function is the default value, used wherever
        all conditions are false.
    funclist : list of M or M+1 callables, f(x,*args,**kw), or values
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take an array as input and give an array
        or a scalar value as output.  If, instead of a callable,
        a value is provided then a constant function (``lambda x: value``) is
        assumed.
    args : tuple, optional
        Any further arguments given to `piecewise` are passed to the functions
        upon execution, i.e., if called ``piecewise(...,...,1,'a')``, then
        each function is called as ``f(x,1,'a')``.
    kw : dictionary, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(...,...,lambda=1)``, then each function is called as
        ``f(x,lambda=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have undefined values.

    Notes
    -----
    This is similar to choose or select, except that functions are
    evaluated on elements of `x` that satisfy the corresponding condition from
    `condlist`.

    The result is::

            |--
            |funclist[0](x[condlist[0]])
      out = |funclist[1](x[condlist[1]])
            |...
            |funclist[n2](x[condlist[n2]])
            |--

    Examples
    --------
    Define the sigma function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

    >>> x = np.arange(6) - 2.5 # x runs from -2.5 to 2.5 in steps of 1
    >>> np.piecewise(x, [x < 0, x >= 0.5], [-1,1])
    array([-1., -1., -1.,  1.,  1.,  1.])

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
    ``x >= 0``.

    >>> np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    array([ 2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    """
    x = asanyarray(x)
    n2 = len(funclist)
    if isscalar(condlist) or \
           not (isinstance(condlist[0], list) or
                isinstance(condlist[0], ndarray)):
        condlist = [condlist]
    condlist = [asarray(c, dtype=bool) for c in condlist]
    n = len(condlist)
    if n == n2-1:  # compute the "otherwise" condition.
        totlist = condlist[0]
        for k in range(1, n):
            totlist |= condlist[k]
        condlist.append(~totlist)
        n += 1
    if (n != n2):
        raise ValueError, "function list and condition list " \
                          "must be the same"
    zerod = False
    # This is a hack to work around problems with NumPy's
    #  handling of 0-d arrays and boolean indexing with
    #  numpy.bool_ scalars
    if x.ndim == 0:
        x = x[None]
        zerod = True
        newcondlist = []
        for k in range(n):
            if condlist[k].ndim == 0:
                condition = condlist[k][None]
            else:
                condition = condlist[k]
            newcondlist.append(condition)
        condlist = newcondlist

    y = zeros(x.shape, x.dtype)
    for k in range(n):
        item = funclist[k]
        if not callable(item):
            y[condlist[k]] = item
        else:
            vals = x[condlist[k]]
            if vals.size > 0:
                y[condlist[k]] = item(vals, *args, **kw)
    if zerod:
        y = y.squeeze()
    return y

def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of N boolean arrays of length M
            The conditions C_0 through C_(N-1) which determine
            from which vector the output elements are taken.
    choicelist : list of N arrays of length M
            Th vectors V_0 through V_(N-1), from which the output
            elements are chosen.

    Returns
    -------
    output : 1-dimensional array of length M
            The output at position m is the m-th element of the first
            vector V_n for which C_n[m] is non-zero.  Note that the
            output depends on the order of conditions, since the
            first satisfied condition is used.

    Notes
    -----
    Equivalent to:
    ::

                output = []
                for m in range(M):
                    output += [V[m] for V,C in zip(values,cond) if C[m]]
                              or [default]

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

def copy(a):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    Notes
    -----
    This is equivalent to

    >>> np.array(a, copy=True)

    Examples
    --------
    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> x[0] == y[0]
    True
    >>> x[0] == z[0]
    False

    """
    return array(a, copy=True)

# Basic operations

def gradient(f, *varargs):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using central differences in the interior
    and first differences at the boundaries. The returned gradient hence has
    the same shape as the input array.

    Parameters
    ----------
    f : array_like
      An N-dimensional array containing samples of a scalar function.
    `*varargs` : scalars
      0, 1, or N scalars specifying the sample distances in each direction,
      that is: `dx`, `dy`, `dz`, ... The default distance is 1.


    Returns
    -------
    g : ndarray
      N arrays of the same shape as `f` giving the derivative of `f` with
      respect to each dimension.

    Examples
    --------
    >>> np.gradient(np.array([[1,1],[3,4]]))
    [array([[ 2.,  3.],
           [ 2.,  3.]]),
     array([[ 0.,  0.],
           [ 1.,  1.]])]

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
    """
    Calculate the nth order discrete difference along given axis.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken.

    Returns
    -------
    out : ndarray
        The `n` order differences.  The shape of the output is the same as `a`
        except along `axis` where the dimension is `n` less.

    Examples
    --------
    >>> x = np.array([0,1,3,9,5,10])
    >>> np.diff(x)
    array([ 1,  2,  6, -4,  5])
    >>> np.diff(x,n=2)
    array([  1,   4, -10,   9])
    >>> x = np.array([[1,3,6,10],[0,5,6,8]])
    >>> np.diff(x)
    array([[2, 3, 4],
    [5, 1, 2]])
    >>> np.diff(x,axis=0)
    array([[-1,  2,  0, -2]])

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


def interp(x, xp, fp, left=None, right=None):
    """
    One-dimensional linear interpolation.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given values at discrete data-points.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the interpolated values.

    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing.

    fp : 1-D sequence of floats
        The y-coordinates of the data points, same length as `xp`.

    left : float, optional
        Value to return for `x < xp[0]`, default is `fp[0]`.

    right : float, optional
        Value to return for `x > xp[-1]`, defaults is `fp[-1]`.

    Returns
    -------
    y : {float, ndarray}
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length

    Notes
    -----
    Does not check that the x-coordinate sequence `xp` is increasing.
    If `xp` is not increasing, the results are nonsense.
    A simple check for increasingness is::

        np.all(np.diff(xp) > 0)


    Examples
    --------
    >>> xp = [1, 2, 3]
    >>> fp = [3, 2, 0]
    >>> np.interp(2.5, xp, fp)
    1.0
    >>> np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
    array([ 3. ,  3. ,  2.5,  0.56,  0. ])
    >>> UNDEF = -99.0
    >>> np.interp(3.14, xp, fp, right=UNDEF)
    -99.0

    """
    if isinstance(x, (float, int, number)):
        return compiled_interp([x], xp, fp, left, right).item()
    else:
        return compiled_interp(x, xp, fp, left, right)


def angle(z, deg=0):
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False.  Default is False.

    Returns
    -------
    angle : {ndarray, scalar}
        The angle is defined as counterclockwise from the positive real axis on
        the complex plane, with dtype as numpy.float64.

    Examples
    --------
    >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
    array([ 0.        ,  1.57079633,  0.78539816])
    >>> np.angle(1+1j, deg=True)                  # in degrees
    45.0

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
    """
    Unwrap by changing deltas between values to 2*pi complement.

    Unwrap radian phase `p` by changing absolute jumps greater than
    `discont` to their 2*pi complement along the given axis.

    Parameters
    ----------
    p : array_like
        Input array.
    discont : float
        Maximum discontinuity between values.
    axis : integer
        Axis along which unwrap will operate.

    Returns
    -------
    out : ndarray
        Output array

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
    """
    Sort a complex array using the real part first, then the imaginary part.

    Parameters
    ----------
    a : array_like
        Input array

    Returns
    -------
    out : complex ndarray
        Always returns a sorted complex array.

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
    """
    Trim the leading and trailing zeros from a 1D array.

    Parameters
    ----------
    filt : array_like
        Input array.
    trim : string, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back.

    Examples
    --------
    >>> a = np.array((0, 0, 0, 1, 2, 3, 2, 1, 0))
    >>> np.trim_zeros(a)
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
    """
    Return the sorted, unique elements of an array or sequence.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    y : ndarray
        The sorted, unique elements are returned in a 1-D array.

    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

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
    """
    Return the elements of an array that satisfy some condition.

    This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.  If
    `condition` is boolean ``np.extract`` is equivalent to ``arr[condition]``.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of `arr`
        to extract.
    arr : array_like
        Input array of the same size as `condition`.

    See Also
    --------
    take, put, putmask

    Examples
    --------
    >>> arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> condition = np.mod(arr, 3)==0
    >>> condition
    array([[False, False,  True, False],
           [False,  True, False, False],
           [ True, False, False,  True]], dtype=bool)
    >>> np.extract(condition, arr)
    array([ 3,  6,  9, 12])

    If `condition` is boolean:

    >>> arr[condition]
    array([ 3,  6,  9, 12])

    """
    return _nx.take(ravel(arr), nonzero(ravel(condition))[0])

def place(arr, mask, vals):
    """Similar to putmask arr[mask] = vals but the 1D array vals has the
    same number of elements as the non-zero values of mask. Inverse of
    extract.

    """
    return _insert(arr, mask, vals)

def _nanop(op, fill, a, axis=None):
    """
    General operation on arrays with not-a-number values.

    Parameters
    ----------
    op : callable
        Operation to perform.
    fill : float
        NaN values are set to fill before doing the operation.
    a : array-like
        Input array.
    axis : {int, None}, optional
        Axis along which the operation is computed.
        By default the input is flattened.

    Returns
    -------
    y : {ndarray, scalar}
        Processed data.

    """
    y = array(a,subok=True)
    mask = isnan(a)
    if mask.all():
        return np.nan

    if not issubclass(y.dtype.type, np.integer):
        y[mask] = fill

    return op(y, axis=axis)

def nansum(a, axis=None):
    """
    Sum the array along the given axis, treating NaNs as zero.

    Parameters
    ----------
    a : array-like
        Input array.
    axis : {int, None}, optional
        Axis along which the sum is computed. By default `a` is flattened.

    Returns
    -------
    y : {ndarray, scalar}
        The sum ignoring NaNs.

    Examples
    --------
    >>> np.nansum([np.nan, 1])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> np.nansum(a, axis=0)
    array([ 2.,  1.])

    """
    return _nanop(np.sum, 0, a, axis)

def nanmin(a, axis=None):
    """
    Find the minimum along the given axis, ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the minimum is computed. By default `a` is flattened.

    Returns
    -------
    y : {ndarray, scalar}
        The minimum ignoring NaNs.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmin(a)
    1.0
    >>> np.nanmin(a, axis=0)
    array([ 1.,  2.])
    >>> np.nanmin(a, axis=1)
    array([ 1.,  3.])

    """
    return _nanop(np.min, np.inf, a, axis)

def nanargmin(a, axis=None):
    """
    Return indices of the minimum values along the given axis of `a`,
    ignoring NaNs.

    Refer to `numpy.nanargmax` for detailed documentation.

    """
    return _nanop(np.argmin, np.inf, a, axis)

def nanmax(a, axis=None):
    """
    Find the maximum along the given axis, ignoring NaNs.

    Parameters
    ----------
    a : array-like
        Input array.
    axis : {int, None}, optional
        Axis along which the maximum is computed. By default `a` is flattened.

    Returns
    -------
    y : {ndarray, scalar}
        The maximum ignoring NaNs.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmax(a)
    3.0
    >>> np.nanmax(a, axis=0)
    array([ 3.,  2.])
    >>> np.nanmax(a, axis=1)
    array([ 2.,  3.])

    """
    return _nanop(np.max, -np.inf, a, axis)

def nanargmax(a, axis=None):
    """
    Return indices of the maximum values over the given axis of 'a',
    ignoring NaNs.

    Parameters
    ----------
    a : array-like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.

    Returns
    -------
    index_array : {ndarray, int}
        An array of indices or a single index value.

    See Also
    --------
    argmax

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmax(a)
    0
    >>> np.nanargmax(a)
    1
    >>> np.nanargmax(a, axis=0)
    array([1, 1])
    >>> np.nanargmax(a, axis=1)
    array([1, 0])

    """
    return _nanop(np.argmax, -np.inf, a, axis)

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

    Generalized function class.

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

    Parameters
    ----------
    f : callable
      A Python function or method.

    Examples
    --------
    >>> def myfunc(a, b):
    ...    if a > b:
    ...        return a-b
    ...    else:
    ...        return a+b

    >>> vfunc = np.vectorize(myfunc)

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
        if isinstance(otypes, str):
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

        # we need a new ufunc if this is being called with more arguments.
        if (self.lastcallargs != nargs):
            self.lastcallargs = nargs
            self.ufunc = None
            self.nout = None

        if self.nout is None or self.otypes == '':
            newargs = []
            for arg in args:
                newargs.append(asarray(arg).flat[0])
            theout = self.thefunc(*newargs)
            if isinstance(theout, tuple):
                self.nout = len(theout)
            else:
                self.nout = 1
                theout = (theout,)
            if self.otypes == '':
                otypes = []
                for k in range(self.nout):
                    otypes.append(asarray(theout[k]).dtype.char)
                self.otypes = ''.join(otypes)

        # Create ufunc if not already created
        if (self.ufunc is None):
            self.ufunc = frompyfunc(self.thefunc, nargs, self.nout)

        # Convert to object arrays first
        newargs = [array(arg,copy=False,subok=True,dtype=object) for arg in args]
        if self.nout == 1:
            _res = array(self.ufunc(*newargs),copy=False,
                         subok=True,dtype=self.otypes[0])
        else:
            _res = tuple([array(x,copy=False,subok=True,dtype=c) \
                          for x, c in zip(self.ufunc(*newargs), self.otypes)])
        return _res

def cov(m, y=None, rowvar=1, bias=0):
    """
    Estimate a covariance matrix, given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    Parameters
    ----------
    m : array-like
        A 1D or 2D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array-like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N-1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``.

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.vstack((x,y))
    >>> print np.cov(X)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print np.cov(x, y)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print np.cov(x)
    11.71

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
    """
    Correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, P, and the
    covariance matrix, C, is

    .. math:: P_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    The values of P are between -1 and 1.

    See Also
    --------
    cov : Covariance matrix

    """
    c = cov(x, y, rowvar, bias)
    try:
        d = diag(c)
    except ValueError: # scalar covariance
        return 1
    return c/sqrt(multiply.outer(d,d))

def blackman(M):
    """
    Return the Blackman window.

    The Blackman window is a taper formed by using the the first
    three terms of a summation of cosines. It was designed to have close
    to the minimal leakage possible.
    It is close to optimal, only slightly worse than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : array
        The window, normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, hamming, hanning, kaiser

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)


    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the kaiser window.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [3] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
           Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

    Examples
    --------
    >>> from numpy import blackman
    >>> blackman(12)
    array([ -1.38777878e-17,   3.26064346e-02,   1.59903635e-01,
             4.14397981e-01,   7.36045180e-01,   9.67046769e-01,
             9.67046769e-01,   7.36045180e-01,   4.14397981e-01,
             1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])


    Plot the window and the frequency response:

    >>> from numpy import clip, log10, array, bartlett
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = blackman(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.show()

    >>> A = fft(window, 2048) / 25.5
    >>> mag = abs(fftshift(A))
    >>> freq = linspace(-0.5,0.5,len(A))
    >>> response = 20*log10(mag)
    >>> response = clip(response,-100,100)
    >>> plt.plot(freq, response)
    >>> plt.title("Frequency response of Bartlett window")
    >>> plt.ylabel("Magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axis('tight'); plt.show()

    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0,M)
    return 0.42-0.5*cos(2.0*pi*n/(M-1)) + 0.08*cos(4.0*pi*n/(M-1))

def bartlett(M):
    """
    Return the Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : array
        The triangular window, normalized to one (the value one
        appears only if the number of samples is odd), with the first
        and last samples equal to zero.

    See Also
    --------
    blackman, hamming, hanning, kaiser

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \\frac{2}{M-1} \\left(
              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
              \\right)

    Most references to the Bartlett window come from the signal
    processing literature, where it is used as one of many windowing
    functions for smoothing values.  Note that convolution with this
    window produces linear interpolation.  It is also known as an
    apodization (which means"removing the foot", i.e. smoothing
    discontinuities at the beginning and end of the sampled signal) or
    tapering function. The fourier transform of the Bartlett is the product
    of two sinc functions.
    Note the excellent discussion in Kanasewich.

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.


    Examples
    --------
    >>> np.bartlett(12)
    array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273,
            0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,
            0.18181818,  0.        ])

    Plot the window and its frequency response (requires SciPy and matplotlib):

    >>> from numpy import clip, log10, array, bartlett
    >>> from numpy.fft import fft
    >>> import matplotlib.pyplot as plt

    >>> window = bartlett(51)
    >>> plt.plot(window)
    >>> plt.title("Bartlett window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.show()

    >>> A = fft(window, 2048) / 25.5
    >>> mag = abs(fftshift(A))
    >>> freq = linspace(-0.5,0.5,len(A))
    >>> response = 20*log10(mag)
    >>> response = clip(response,-100,100)
    >>> plt.plot(freq, response)
    >>> plt.title("Frequency response of Bartlett window")
    >>> plt.ylabel("Magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axis('tight'); plt.show()

    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0,M)
    return where(less_equal(n,(M-1)/2.0),2.0*n/(M-1),2.0-2.0*n/(M-1))

def hanning(M):
    """
    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : array
        The window, normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hamming, kaiser

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hanning was named for Julius van Hann, an Austrian meterologist. It is
    also known as the Cosine Bell. Some authors prefer that it be called a
    Hann window, to help avoid confusion with the very similar Hamming window.

    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> from numpy import hanning
    >>> hanning(12)
    array([ 0.        ,  0.07937323,  0.29229249,  0.57115742,  0.82743037,
            0.97974649,  0.97974649,  0.82743037,  0.57115742,  0.29229249,
            0.07937323,  0.        ])

    Plot the window and its frequency response:

    >>> from numpy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = np.hanning(51)
    >>> plt.subplot(121)
    >>> plt.plot(window)
    >>> plt.title("Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> A = fft(window, 2048) / 25.5
    >>> mag = abs(fftshift(A))
    >>> freq = np.linspace(-0.5,0.5,len(A))
    >>> response = 20*np.log10(mag)
    >>> response = np.clip(response,-100,100)
    >>> plt.subplot(122)
    >>> plt.plot(freq, response)
    >>> plt.title("Frequency response of the Hann window")
    >>> plt.ylabel("Magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axis('tight'); plt.show()

    """
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0,M)
    return 0.5-0.5*cos(2.0*pi*n/(M-1))

def hamming(M):
    """
    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The window, normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hanning, kaiser

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 + 0.46cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and
    is described in Blackman and Tukey. It was recommended for smoothing the
    truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> from numpy import hamming
    >>> hamming(12)
    array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594,
            0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
            0.15302337,  0.08      ])

    Plot the window and the frequency response:

    >>> from numpy import clip, log10, array, hamming
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = hamming(51)
    >>> plt.plot(window)
    >>> plt.title("Hamming window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.show()

    >>> A = fft(window, 2048) / 25.5
    >>> mag = abs(fftshift(A))
    >>> freq = linspace(-0.5,0.5,len(A))
    >>> response = 20*log10(mag)
    >>> response = clip(response,-100,100)
    >>> plt.plot(freq, response)
    >>> plt.title("Frequency response of Hamming window")
    >>> plt.ylabel("Magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axis('tight'); plt.show()

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
    """
    Modified Bessel function of the first kind, order 0, :math:`I_0`

    Parameters
    ----------
    x : array-like, dtype float or complex
        Argument of the Bessel function.

    Returns
    -------
    out : ndarray, shape z.shape, dtype z.dtype
        The modified Bessel function evaluated for all elements of `x`.

    Examples
    --------
    >>> np.i0([0.])
    array(1.0)
    >>> np.i0([0., 1. + 2j])
    array([ 1.00000000+0.j        ,  0.18785373+0.64616944j])

    """
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
    """
    Return the Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    beta : float
        Shape parameter for window.

    Returns
    -------
    out : array
        The window, normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hamming, hanning

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
               \\right)/I_0(\\beta)

    with

    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.

    The Kaiser can approximate many other windows by varying the beta
    parameter.

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hanning
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise nans will
    get returned.


    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1]  J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
            digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
            John Wiley and Sons, New York, (1966).
    .. [2]\tE.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
            University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function

    Examples
    --------
    >>> from numpy import kaiser
    >>> kaiser(12, 14)
    array([  7.72686684e-06,   3.46009194e-03,   4.65200189e-02,
             2.29737120e-01,   5.99885316e-01,   9.45674898e-01,
             9.45674898e-01,   5.99885316e-01,   2.29737120e-01,
             4.65200189e-02,   3.46009194e-03,   7.72686684e-06])


    Plot the window and the frequency response:

    >>> from numpy import clip, log10, array, kaiser
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = kaiser(51, 14)
    >>> plt.plot(window)
    >>> plt.title("Kaiser window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.show()

    >>> A = fft(window, 2048) / 25.5
    >>> mag = abs(fftshift(A))
    >>> freq = linspace(-0.5,0.5,len(A))
    >>> response = 20*log10(mag)
    >>> response = clip(response,-100,100)
    >>> plt.plot(freq, response)
    >>> plt.title("Frequency response of Kaiser window")
    >>> plt.ylabel("Magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axis('tight'); plt.show()

    """
    from numpy.dual import i0
    n = arange(0,M)
    alpha = (M-1)/2.0
    return i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/i0(beta)

def sinc(x):
    """
    Return the sinc function.

    The sinc function is :math:`\\sin(\\pi x)/(\\pi x)`.

    Parameters
    ----------
    x : ndarray
        Array (possibly multi-dimensional) of values for which to to
        calculate ``sinc(x)``.

    Returns
    -------
    out : ndarray
        ``sinc(x)``, which has the same shape as the input.

    Notes
    -----
    ``sinc(0)`` is the limit value 1.

    The name sinc is short for "sine cardinal" or "sinus cardinalis".

    The sinc function is used in various signal processing applications,
    including in anti-aliasing, in the construction of a
    Lanczos resampling filter, and in interpolation.

    For bandlimited interpolation of discrete-time signals, the ideal
    interpolation kernel is proportional to the sinc function.

    References
    ----------
    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
           Resource. http://mathworld.wolfram.com/SincFunction.html
    .. [2] Wikipedia, "Sinc function",
           http://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> x = np.arange(-20., 21.)/5.
    >>> np.sinc(x)
    array([ -3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02,
            -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,
             6.68206631e-02,   1.16434881e-01,   1.26137788e-01,
             8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,
            -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,
             3.89804309e-17,   2.33872321e-01,   5.04551152e-01,
             7.56826729e-01,   9.35489284e-01,   1.00000000e+00,
             9.35489284e-01,   7.56826729e-01,   5.04551152e-01,
             2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,
            -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,
            -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,
             1.16434881e-01,   6.68206631e-02,   3.89804309e-17,
            -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,
            -4.92362781e-02,  -3.89804309e-17])

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, sinc(x))
    >>> plt.title("Sinc Function")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("X")
    >>> plt.show()

    It works in 2-D as well:

    >>> x = np.arange(-200., 201.)/50.
    >>> xx = np.outer(x, x)
    >>> plt.imshow(sinc(xx))

    """
    y = pi* where(x == 0, 1.0e-20, x)
    return sin(y)/y

def msort(a):
    b = array(a,subok=True,copy=True)
    b.sort(0)
    return b

def median(a, axis=None, out=None, overwrite_input=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {None, int}, optional
        Axis along which the medians are computed. The default (axis=None) is to
        compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : {False, True}, optional
       If True, then allow use of memory of input array (a) for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. Note that, if `overwrite_input` is True and the input
       is not already an ndarray, an error will be raised.

    Returns
    -------
    median : ndarray
        A new array holding the result (unless `out` is specified, in
        which case that array is returned instead).  If the input contains
        integers, or floats of smaller precision than 64, then the output
        data-type is float64.  Otherwise, the output data-type is the same
        as that of the input.

    See Also
    --------
    mean

    Notes
    -----
    Given a vector V of length N, the median of V is the middle value of
    a sorted copy of V, ``V_sorted`` - i.e., ``V_sorted[(N-1)/2]``, when N is
    odd.  When N is even, it is the average of the two middle values of
    ``V_sorted``.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    3.5
    >>> np.median(a, axis=0)
    array([ 6.5,  4.5,  2.5])
    >>> np.median(a, axis=1)
    array([ 7.,  2.])
    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([ 6.5,  4.5,  2.5])
    >>> m
    array([ 6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([ 7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    3.5
    >>> assert not np.all(a==b)

    """
    if overwrite_input:
        if axis is None:
            sorted = a.ravel()
            sorted.sort()
        else:
            a.sort(axis=axis)
            sorted = a
    else:
        sorted = sort(a, axis=axis)
    if axis is None:
        axis = 0
    indexer = [slice(None)] * sorted.ndim
    index = int(sorted.shape[axis]/2)
    if sorted.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    # Use mean in odd and even case to coerce data type
    # and check, use out array.
    return mean(sorted[indexer], axis=axis, out=out)

def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Integrate `y` (`x`) along given axis.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : {array_like, None}, optional
        If `x` is None, then spacing between all `y` elements is 1.
    dx : scalar, optional
        If `x` is None, spacing given by `dx` is assumed.
    axis : int, optional
        Specify the axis.

    Examples
    --------
    >>> np.trapz([1,2,3])
    >>> 4.0
    >>> np.trapz([1,2,3], [4,6,8])
    >>> 8.0

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
    Return coordinate matrices from two coordinate vectors.




    Parameters
    ----------
    x, y : ndarray
        Two 1D arrays representing the x and y coordinates

    Returns
    -------
    X, Y : ndarray
        For vectors `x`, `y` with lengths Nx=len(`x`) and Ny=len(`y`),
        return `X`, `Y` where `X` and `Y` are (Ny, Nx) shaped arrays
        with the elements of `x` and y repeated to fill the matrix along
        the first dimension for `x`, the second for `y`.

    See Also
    --------
    numpy.mgrid : Construct a multi-dimensional "meshgrid"
                               using indexing notation.

    Examples
    --------
    >>> X, Y = numpy.meshgrid([1,2,3], [4,5,6,7])
    >>> X
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    >>> Y
    array([[4, 4, 4],
           [5, 5, 5],
           [6, 6, 6],
           [7, 7, 7]])

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
    """
    Return a new array with sub-arrays along an axis deleted.

    Parameters
    ----------
    arr : array-like
      Input array.
    obj : slice, integer or an array of integers
      Indicate which sub-arrays to remove.
    axis : integer or None
      The axis along which to delete the subarray defined by `obj`.
      If `axis` is None, `obj` is applied to the flattened array.

    See Also
    --------
    insert : Insert values into an array.
    append : Append values at the end of an array.

    Examples
    --------
    >>> arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, np.s_[::2], 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> np.delete(arr, [1,3,5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

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
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : {integer, slice, integer array_like}
        Insert `values` before `obj` indices.
    values :
        Values to insert into `arr`.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then ravel
        `arr` first.

    Examples
    --------
    >>> a = np.array([[1,2,3],
    ...               [4,5,6],
    ...               [7,8,9]])
    >>> np.insert(a, [1,2], [[4],[5]], axis=0)
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
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If `axis`
        is not specified, `values` can be any shape and will be flattened
        before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not given,
        both `arr` and `values` are flattened before use.

    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that `append`
        does not occur in-place: a new array is allocated and filled.

    Examples
    --------
    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    """
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim-1
    return concatenate((arr, values), axis=axis)
