"""
Generic statistics functions, with support to MA.

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $
:version: $Id: mstats.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'


import numpy
from numpy import bool_, float_, int_, \
    sqrt
from numpy import array as narray
import numpy.core.numeric as numeric
from numpy.core.numeric import concatenate

import numpy.ma
from numpy.ma.core import masked, nomask, MaskedArray, masked_array
from numpy.ma.extras import apply_along_axis, dot

__all__ = ['cov','meppf','plotting_positions','meppf','mmedian','mquantiles',
           'stde_median','trim_tail','trim_both','trimmed_mean','trimmed_stde',
           'winsorize']

#####--------------------------------------------------------------------------
#---- -- Trimming ---
#####--------------------------------------------------------------------------

def winsorize(data, alpha=0.2):
    """Returns a Winsorized version of the input array.

The (alpha/2.) lowest values are set to the (alpha/2.)th percentile, and
the (alpha/2.) highest values are set to the (1-alpha/2.)th percentile
Masked values are skipped.

*Parameters*:
    data : {ndarray}
        Input data to Winsorize. The data is first flattened.
    alpha : {float}, optional
        Percentage of total Winsorization : alpha/2. on the left, alpha/2. on the right
    """
    data = masked_array(data, copy=False).ravel()
    idxsort = data.argsort()
    (nsize, ncounts) = (data.size, data.count())
    ntrim = int(alpha * ncounts)
    (xmin,xmax) = data[idxsort[[ntrim, ncounts-nsize-ntrim-1]]]
    return masked_array(numpy.clip(data, xmin, xmax), mask=data._mask)

#..............................................................................
def trim_both(data, proportiontocut=0.2, axis=None):
    """Trims the data by masking the int(trim*n) smallest and int(trim*n) largest
values of data along the given axis, where n is the number of unmasked values.

*Parameters*:
    data : {ndarray}
        Data to trim.
    proportiontocut : {float}
        Percentage of trimming. If n is the number of unmasked values before trimming,
        the number of values after trimming is (1-2*trim)*n.
    axis : {integer}
        Axis along which to perform the trimming. If None, the input array is first
        flattened.
    """
    #...................
    def _trim_1D(data, trim):
        "Private function: return a trimmed 1D array."
        nsize = data.size
        ncounts = data.count()
        ntrim = int(trim * ncounts)
        idxsort = data.argsort()
        data[idxsort[:ntrim]] = masked
        data[idxsort[ncounts-nsize-ntrim:]] = masked
        return data
    #...................
    data = masked_array(data, copy=False, subok=True)
    data.unshare_mask()
    if (axis is None):
        return _trim_1D(data.ravel(), proportiontocut)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        return apply_along_axis(_trim_1D, axis, data, proportiontocut)

#..............................................................................
def trim_tail(data, proportiontocut=0.2, tail='left', axis=None):
    """Trims the data by masking int(trim*n) values from ONE tail of the data
along the given axis, where n is the number of unmasked values.

*Parameters*:
    data : {ndarray}
        Data to trim.
    proportiontocut : {float}
        Percentage of trimming. If n is the number of unmasked values before trimming,
        the number of values after trimming is (1-trim)*n.
    tail : {string}
        Trimming direction, in ('left', 'right'). If left, the proportiontocut
        lowest values are set to the corresponding percentile. If right, the
        proportiontocut highest values are used instead.
    axis : {integer}
        Axis along which to perform the trimming. If None, the input array is first
        flattened.
    """
    #...................
    def _trim_1D(data, trim, left):
        "Private function: return a trimmed 1D array."
        nsize = data.size
        ncounts = data.count()
        ntrim = int(trim * ncounts)
        idxsort = data.argsort()
        if left:
            data[idxsort[:ntrim]] = masked
        else:
            data[idxsort[ncounts-nsize-ntrim:]] = masked
        return data
    #...................
    data = masked_array(data, copy=False, subok=True)
    data.unshare_mask()
    #
    if not isinstance(tail, str):
        raise TypeError("The tail argument should be in ('left','right')")
    tail = tail.lower()[0]
    if tail == 'l':
        left = True
    elif tail == 'r':
        left=False
    else:
        raise ValueError("The tail argument should be in ('left','right')")
    #
    if (axis is None):
        return _trim_1D(data.ravel(), proportiontocut, left)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        return apply_along_axis(_trim_1D, axis, data, proportiontocut, left)

#..............................................................................
def trimmed_mean(data, proportiontocut=0.2, axis=None):
    """Returns the trimmed mean of the data along the given axis. Trimming is
performed on both ends of the distribution.

*Parameters*:
    data : {ndarray}
        Data to trim.
    proportiontocut : {float}
        Proportion of the data to cut from each side of the data .
        As a result, (2*proportiontocut*n) values are actually trimmed.
    axis : {integer}
        Axis along which to perform the trimming. If None, the input array is first
        flattened.
    """
    return trim_both(data, proportiontocut=proportiontocut, axis=axis).mean(axis=axis)

#..............................................................................
def trimmed_stde(data, proportiontocut=0.2, axis=None):
    """Returns the standard error of the trimmed mean for the input data,
along the given axis. Trimming is performed on both ends of the distribution.

*Parameters*:
    data : {ndarray}
        Data to trim.
    proportiontocut : {float}
        Proportion of the data to cut from each side of the data .
        As a result, (2*proportiontocut*n) values are actually trimmed.
    axis : {integer}
        Axis along which to perform the trimming. If None, the input array is first
        flattened.
    """
    #........................
    def _trimmed_stde_1D(data, trim=0.2):
        "Returns the standard error of the trimmed mean for a 1D input data."
        winsorized = winsorize(data)
        nsize = winsorized.count()
        winstd = winsorized.stdu()
        return winstd / ((1-2*trim) * numpy.sqrt(nsize))
    #........................
    data = masked_array(data, copy=False, subok=True)
    data.unshare_mask()
    if (axis is None):
        return _trimmed_stde_1D(data.ravel(), proportiontocut)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        return apply_along_axis(_trimmed_stde_1D, axis, data, proportiontocut)

#.............................................................................
def stde_median(data, axis=None):
    """Returns the McKean-Schrader estimate of the standard error of the sample
median along the given axis.


*Parameters*:
    data : {ndarray}
        Data to trim.
    axis : {integer}
        Axis along which to perform the trimming. If None, the input array is first
        flattened.
    """
    def _stdemed_1D(data):
        sorted = numpy.sort(data.compressed())
        n = len(sorted)
        z = 2.5758293035489004
        k = int(round((n+1)/2. - z * sqrt(n/4.),0))
        return ((sorted[n-k] - sorted[k-1])/(2.*z))
    #
    data = masked_array(data, copy=False, subok=True)
    if (axis is None):
        return _stdemed_1D(data)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        return apply_along_axis(_stdemed_1D, axis, data)


#####--------------------------------------------------------------------------
#---- --- Quantiles ---
#####--------------------------------------------------------------------------


def mquantiles(data, prob=list([.25,.5,.75]), alphap=.4, betap=.4, axis=None):
    """Computes empirical quantiles for a *1xN* data array.
Samples quantile are defined by:
*Q(p) = (1-g).x[i] +g.x[i+1]*
where *x[j]* is the jth order statistic,
with *i = (floor(n*p+m))*, *m=alpha+p*(1-alpha-beta)* and *g = n*p + m - i)*.

Typical values of (alpha,beta) are:

    - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)
    - (.5,.5)  : *p(k) = (k+1/2.)/n* : piecewise linear function (R, type 5)
    - (0,0)    : *p(k) = k/(n+1)* : (R type 6)
    - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].
      That's R default (R type 7)
    - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].
      The resulting quantile estimates are approximately median-unbiased
      regardless of the distribution of x. (R type 8)
    - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*. Blom.
      The resulting quantile estimates are approximately unbiased
      if x is normally distributed (R type 9)
    - (.4,.4)  : approximately quantile unbiased (Cunnane)
    - (.35,.35): APL, used with PWM

*Parameters*:
    x : {sequence}
        Input data, as a sequence or array of dimension at most 2.
    prob : {sequence}
        List of quantiles to compute.
    alpha : {float}
        Plotting positions parameter.
    beta : {float}
        Plotting positions parameter.
    axis : {integer}
        Axis along which to perform the trimming. If None, the input array is first
        flattened.
    """
    def _quantiles1D(data,m,p):
        x = numpy.sort(data.compressed())
        n = len(x)
        if n == 0:
            return masked_array(numpy.empty(len(p), dtype=float_), mask=True)
        elif n == 1:
            return masked_array(numpy.resize(x, p.shape), mask=nomask)
        aleph = (n*p + m)
        k = numpy.floor(aleph.clip(1, n-1)).astype(int_)
        gamma = (aleph-k).clip(0,1)
        return (1.-gamma)*x[(k-1).tolist()] + gamma*x[k.tolist()]

    # Initialization & checks ---------
    data = masked_array(data, copy=False)
    p = narray(prob, copy=False, ndmin=1)
    m = alphap + p*(1.-alphap-betap)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _quantiles1D(data, m, p)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        return apply_along_axis(_quantiles1D, axis, data, m, p)


def plotting_positions(data, alpha=0.4, beta=0.4):
    """Returns the plotting positions (or empirical percentile points) for the
    data.
    Plotting positions are defined as (i-alpha)/(n-alpha-beta), where:
        - i is the rank order statistics
        - n is the number of unmasked values along the given axis
        - alpha and beta are two parameters.

    Typical values for alpha and beta are:
        - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)
        - (.5,.5)  : *p(k) = (k-1/2.)/n* : piecewise linear function (R, type 5)
        - (0,0)    : *p(k) = k/(n+1)* : Weibull (R type 6)
        - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].
          That's R default (R type 7)
        - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8)
        - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*. Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM
    """
    data = masked_array(data, copy=False).reshape(1,-1)
    n = data.count()
    plpos = numpy.empty(data.size, dtype=float_)
    plpos[n:] = 0
    plpos[data.argsort()[:n]] = (numpy.arange(1,n+1) - alpha)/(n+1-alpha-beta)
    return masked_array(plpos, mask=data._mask)

meppf = plotting_positions


def mmedian(data, axis=None):
    """Returns the median of data along the given axis. Missing data are discarded."""
    def _median1D(data):
        x = numpy.sort(data.compressed())
        if x.size == 0:
            return masked
        return numpy.median(x)
    data = masked_array(data, subok=True, copy=True)
    if axis is None:
        return _median1D(data)
    else:
        return apply_along_axis(_median1D, axis, data)


def cov(x, y=None, rowvar=True, bias=False, strict=False):
    """Estimates the covariance matrix.


Normalization is by (N-1) where N is the number of observations (unbiased
estimate).  If bias is True then normalization is by N.

*Parameters*:
    x : {ndarray}
        Input data. If x is a 1D array, returns the variance. If x is a 2D array,
        returns the covariance matrix.
    y : {ndarray}, optional
        Optional set of variables.
    rowvar : {boolean}
        If rowvar is true, then each row is a variable with obersvations in columns.
        If rowvar is False, each column is a variable and the observations are in
        the rows.
    bias : {boolean}
        Whether to use a biased or unbiased estimate of the covariance.
        If bias is True, then the normalization is by N, the number of observations.
        Otherwise, the normalization is by (N-1)
    strict : {boolean}
        If strict is True, masked values are propagated: if a masked value appears in
        a row or column, the whole row or column is considered masked.
    """
    X = narray(x, ndmin=2, subok=True, dtype=float)
    if X.shape[0] == 1:
        rowvar = True
    if rowvar:
        axis = 0
        tup = (slice(None),None)
    else:
        axis = 1
        tup = (None, slice(None))
    #
    if y is not None:
        y = narray(y, copy=False, ndmin=2, subok=True, dtype=float)
        X = concatenate((X,y),axis)
    #
    X -= X.mean(axis=1-axis)[tup]
    n = X.count(1-axis)
    #
    if bias:
        fact = n*1.0
    else:
        fact = n-1.0
    #
    if not rowvar:
        return (dot(X.T, X.conj(), strict=False) / fact).squeeze()
    else:
        return (dot(X, X.T.conj(), strict=False) / fact).squeeze()


def idealfourths(data, axis=None):
    """Returns an estimate of the interquartile range of the data along the given
axis, as computed with the ideal fourths.
    """
    def _idf(data):
        x = numpy.sort(data.compressed())
        n = len(x)
        (j,h) = divmod(n/4. + 5/12.,1)
        qlo = (1-h)*x[j] + h*x[j+1]
        k = n - j
        qup = (1-h)*x[k] + h*x[k-1]
        return qup - qlo
    data = masked_array(data, copy=False)
    if (axis is None):
        return _idf(data)
    else:
        return apply_along_axis(_idf, axis, data)


def rsh(data, points=None):
    """Evalutates Rosenblatt's shifted histogram estimators for each point
on the dataset 'data'.

*Parameters* :
    data : {sequence}
        Input data. Masked values are ignored.
    points : {sequence}
        Sequence of points where to evaluate Rosenblatt shifted histogram.
        If None, use the data.
    """
    data = masked_array(data, copy=False)
    if points is None:
        points = data
    else:
        points = numpy.array(points, copy=False, ndmin=1)
    if data.ndim != 1:
        raise AttributeError("The input array should be 1D only !")
    n = data.count()
    h = 1.2 * idealfourths(data) / n**(1./5)
    nhi = (data[:,None] <= points[None,:] + h).sum(0)
    nlo = (data[:,None] < points[None,:] - h).sum(0)
    return (nhi-nlo) / (2.*n*h)

################################################################################
if __name__ == '__main__':
    from numpy.ma.testutils import assert_almost_equal
    if 1:
        a = numpy.ma.arange(1,101)
        a[1::2] = masked
        b = numpy.ma.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25., 50., 75.])
        assert_almost_equal(mquantiles(b, axis=0), numpy.ma.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1),
                            numpy.ma.resize([24.9, 50., 75.1], (100,3)))
