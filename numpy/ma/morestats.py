"""
Generic statistics functions, with support to MA.

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $
:version: $Id: morestats.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'


import numpy
from numpy import bool_, float_, int_, ndarray, \
    sqrt,\
    arange, empty,\
    r_
from numpy import array as narray
import numpy.core.numeric as numeric
from numpy.core.numeric import concatenate

import maskedarray as MA
from maskedarray.core import masked, nomask, MaskedArray, masked_array
from maskedarray.extras import apply_along_axis, dot
from maskedarray.mstats import trim_both, trimmed_stde, mquantiles, mmedian, stde_median

from scipy.stats.distributions import norm, beta, t, binom
from scipy.stats.morestats import find_repeats

__all__ = ['hdquantiles', 'hdmedian', 'hdquantiles_sd',
           'trimmed_mean_ci', 'mjci', 'rank_data']


#####--------------------------------------------------------------------------
#---- --- Quantiles ---
#####--------------------------------------------------------------------------
def hdquantiles(data, prob=list([.25,.5,.75]), axis=None, var=False,):
    """Computes quantile estimates with the Harrell-Davis method, where the estimates
are calculated as a weighted linear combination of order statistics.

*Parameters* :
    data: {ndarray}
        Data array.
    prob: {sequence}
        Sequence of quantiles to compute.
    axis : {integer}
        Axis along which to compute the quantiles. If None, use a flattened array.
    var : {boolean}
        Whether to return the variance of the estimate.

*Returns*
    A (p,) array of quantiles (if ``var`` is False), or a (2,p) array of quantiles
    and variances (if ``var`` is True), where ``p`` is the number of quantiles.

:Note:
    The function is restricted to 2D arrays.
    """
    def _hd_1D(data,prob,var):
        "Computes the HD quantiles for a 1D array. Returns nan for invalid data."
        xsorted = numpy.squeeze(numpy.sort(data.compressed().view(ndarray)))
        # Don't use length here, in case we have a numpy scalar
        n = xsorted.size
        #.........
        hd = empty((2,len(prob)), float_)
        if n < 2:
            hd.flat = numpy.nan
            if var:
                return hd
            return hd[0]
        #.........
        v = arange(n+1) / float(n)
        betacdf = beta.cdf
        for (i,p) in enumerate(prob):
            _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
            w = _w[1:] - _w[:-1]
            hd_mean = dot(w, xsorted)
            hd[0,i] = hd_mean
            #
            hd[1,i] = dot(w, (xsorted-hd_mean)**2)
            #
        hd[0, prob == 0] = xsorted[0]
        hd[0, prob == 1] = xsorted[-1]
        if var:
            hd[1, prob == 0] = hd[1, prob == 1] = numpy.nan
            return hd
        return hd[0]
    # Initialization & checks ---------
    data = masked_array(data, copy=False, dtype=float_)
    p = numpy.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None) or (data.ndim == 1):
        result = _hd_1D(data, p, var)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        result = apply_along_axis(_hd_1D, axis, data, p, var)
    #
    return masked_array(result, mask=numpy.isnan(result))

#..............................................................................
def hdmedian(data, axis=-1, var=False):
    """Returns the Harrell-Davis estimate of the median along the given axis.

*Parameters* :
    data: {ndarray}
        Data array.
    axis : {integer}
        Axis along which to compute the quantiles. If None, use a flattened array.
    var : {boolean}
        Whether to return the variance of the estimate.
    """
    result = hdquantiles(data,[0.5], axis=axis, var=var)
    return result.squeeze()


#..............................................................................
def hdquantiles_sd(data, prob=list([.25,.5,.75]), axis=None):
    """Computes the standard error of the Harrell-Davis quantile estimates by jackknife.


*Parameters* :
    data: {ndarray}
        Data array.
    prob: {sequence}
        Sequence of quantiles to compute.
    axis : {integer}
        Axis along which to compute the quantiles. If None, use a flattened array.

*Note*:
    The function is restricted to 2D arrays.
    """
    def _hdsd_1D(data,prob):
        "Computes the std error for 1D arrays."
        xsorted = numpy.sort(data.compressed())
        n = len(xsorted)
        #.........
        hdsd = empty(len(prob), float_)
        if n < 2:
            hdsd.flat = numpy.nan
        #.........
        vv = arange(n) / float(n-1)
        betacdf = beta.cdf
        #
        for (i,p) in enumerate(prob):
            _w = betacdf(vv, (n+1)*p, (n+1)*(1-p))
            w = _w[1:] - _w[:-1]
            mx_ = numpy.fromiter([dot(w,xsorted[r_[range(0,k),
                                                   range(k+1,n)].astype(int_)])
                                  for k in range(n)], dtype=float_)
            mx_var = numpy.array(mx_.var(), copy=False, ndmin=1) * n / float(n-1)
            hdsd[i] = float(n-1) * sqrt(numpy.diag(mx_var).diagonal() / float(n))
        return hdsd
    # Initialization & checks ---------
    data = masked_array(data, copy=False, dtype=float_)
    p = numpy.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _hdsd_1D(data.compressed(), p)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        result = apply_along_axis(_hdsd_1D, axis, data, p)
    #
    return masked_array(result, mask=numpy.isnan(result)).ravel()


#####--------------------------------------------------------------------------
#---- --- Confidence intervals ---
#####--------------------------------------------------------------------------

def trimmed_mean_ci(data, proportiontocut=0.2, alpha=0.05, axis=None):
    """Returns the selected confidence interval of the trimmed mean along the
given axis.

*Parameters* :
    data : {sequence}
        Input data. The data is transformed to a masked array
    proportiontocut : {float}
        Proportion of the data to cut from each side of the data .
        As a result, (2*proportiontocut*n) values are actually trimmed.
    alpha : {float}
        Confidence level of the intervals.
    axis : {integer}
        Axis along which to cut. If None, uses a flattened version of the input.
    """
    data = masked_array(data, copy=False)
    trimmed = trim_both(data, proportiontocut=proportiontocut, axis=axis)
    tmean = trimmed.mean(axis)
    tstde = trimmed_stde(data, proportiontocut=proportiontocut, axis=axis)
    df = trimmed.count(axis) - 1
    tppf = t.ppf(1-alpha/2.,df)
    return numpy.array((tmean - tppf*tstde, tmean+tppf*tstde))

#..............................................................................
def mjci(data, prob=[0.25,0.5,0.75], axis=None):
    """Returns the Maritz-Jarrett estimators of the standard error of selected
experimental quantiles of the data.

*Parameters* :
    data: {ndarray}
        Data array.
    prob: {sequence}
        Sequence of quantiles to compute.
    axis : {integer}
        Axis along which to compute the quantiles. If None, use a flattened array.
    """
    def _mjci_1D(data, p):
        data = data.compressed()
        sorted = numpy.sort(data)
        n = data.size
        prob = (numpy.array(p) * n + 0.5).astype(int_)
        betacdf = beta.cdf
        #
        mj = empty(len(prob), float_)
        x = arange(1,n+1, dtype=float_) / n
        y = x - 1./n
        for (i,m) in enumerate(prob):
            (m1,m2) = (m-1, n-m)
            W = betacdf(x,m-1,n-m) - betacdf(y,m-1,n-m)
            C1 = numpy.dot(W,sorted)
            C2 = numpy.dot(W,sorted**2)
            mj[i] = sqrt(C2 - C1**2)
        return mj
    #
    data = masked_array(data, copy=False)
    assert data.ndim <= 2, "Array should be 2D at most !"
    p = numpy.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _mjci_1D(data, p)
    else:
        return apply_along_axis(_mjci_1D, axis, data, p)

#..............................................................................
def mquantiles_cimj(data, prob=[0.25,0.50,0.75], alpha=0.05, axis=None):
    """Computes the alpha confidence interval for the selected quantiles of the
data, with Maritz-Jarrett estimators.

*Parameters* :
    data: {ndarray}
        Data array.
    prob: {sequence}
        Sequence of quantiles to compute.
    alpha : {float}
        Confidence level of the intervals.
    axis : {integer}
        Axis along which to compute the quantiles. If None, use a flattened array.
    """
    alpha = min(alpha, 1-alpha)
    z = norm.ppf(1-alpha/2.)
    xq = mquantiles(data, prob, alphap=0, betap=0, axis=axis)
    smj = mjci(data, prob, axis=axis)
    return (xq - z * smj, xq + z * smj)


#.............................................................................
def median_cihs(data, alpha=0.05, axis=None):
    """Computes the alpha-level confidence interval for the median of the data,
following the Hettmasperger-Sheather method.

*Parameters* :
    data : {sequence}
        Input data. Masked values are discarded. The input should be 1D only, or
        axis should be set to None.
    alpha : {float}
        Confidence level of the intervals.
    axis : {integer}
        Axis along which to compute the quantiles. If None, use a flattened array.
    """
    def _cihs_1D(data, alpha):
        data = numpy.sort(data.compressed())
        n = len(data)
        alpha = min(alpha, 1-alpha)
        k = int(binom._ppf(alpha/2., n, 0.5))
        gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        if gk < 1-alpha:
            k -= 1
            gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        gkk = binom.cdf(n-k-1,n,0.5) - binom.cdf(k,n,0.5)
        I = (gk - 1 + alpha)/(gk - gkk)
        lambd = (n-k) * I / float(k + (n-2*k)*I)
        lims = (lambd*data[k] + (1-lambd)*data[k-1],
                lambd*data[n-k-1] + (1-lambd)*data[n-k])
        return lims
    data = masked_array(data, copy=False)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _cihs_1D(data.compressed(), p, var)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        result = apply_along_axis(_cihs_1D, axis, data, alpha)
    #
    return result

#..............................................................................
def compare_medians_ms(group_1, group_2, axis=None):
    """Compares the medians from two independent groups along the given axis.

The comparison is performed using the McKean-Schrader estimate of the standard
error of the medians.

*Parameters* :
    group_1 : {sequence}
        First dataset.
    group_2 : {sequence}
        Second dataset.
    axis : {integer}
        Axis along which the medians are estimated. If None, the arrays are flattened.

*Returns* :
    A (p,) array of comparison values.

    """
    (med_1, med_2) = (mmedian(group_1, axis=axis), mmedian(group_2, axis=axis))
    (std_1, std_2) = (stde_median(group_1, axis=axis),
                      stde_median(group_2, axis=axis))
    W = abs(med_1 - med_2) / sqrt(std_1**2 + std_2**2)
    return 1 - norm.cdf(W)


#####--------------------------------------------------------------------------
#---- --- Ranking ---
#####--------------------------------------------------------------------------

#..............................................................................
def rank_data(data, axis=None, use_missing=False):
    """Returns the rank (also known as order statistics) of each data point
along the given axis.

If some values are tied, their rank is averaged.
If some values are masked, their rank is set to 0 if use_missing is False, or
set to the average rank of the unmasked values if use_missing is True.

*Parameters* :
    data : {sequence}
        Input data. The data is transformed to a masked array
    axis : {integer}
        Axis along which to perform the ranking. If None, the array is first
        flattened. An exception is raised if the axis is specified for arrays
        with a dimension larger than 2
    use_missing : {boolean}
        Whether the masked values have a rank of 0 (False) or equal to the
        average rank of the unmasked values (True).
    """
    #
    def _rank1d(data, use_missing=False):
        n = data.count()
        rk = numpy.empty(data.size, dtype=float_)
        idx = data.argsort()
        rk[idx[:n]] = numpy.arange(1,n+1)
        #
        if use_missing:
            rk[idx[n:]] = (n+1)/2.
        else:
            rk[idx[n:]] = 0
        #
        repeats = find_repeats(data)
        for r in repeats[0]:
            condition = (data==r).filled(False)
            rk[condition] = rk[condition].mean()
        return rk
    #
    data = masked_array(data, copy=False)
    if axis is None:
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        return apply_along_axis(_rank1d, axis, data, use_missing)

###############################################################################
if __name__ == '__main__':

    if 0:
        from maskedarray.testutils import assert_almost_equal
        data = [0.706560797,0.727229578,0.990399276,0.927065621,0.158953014,
            0.887764025,0.239407086,0.349638551,0.972791145,0.149789972,
            0.936947700,0.132359948,0.046041972,0.641675031,0.945530547,
            0.224218684,0.771450991,0.820257774,0.336458052,0.589113496,
            0.509736129,0.696838829,0.491323573,0.622767425,0.775189248,
            0.641461450,0.118455200,0.773029450,0.319280007,0.752229111,
            0.047841438,0.466295911,0.583850781,0.840581845,0.550086491,
            0.466470062,0.504765074,0.226855960,0.362641207,0.891620942,
            0.127898691,0.490094097,0.044882048,0.041441695,0.317976349,
            0.504135618,0.567353033,0.434617473,0.636243375,0.231803616,
            0.230154113,0.160011327,0.819464108,0.854706985,0.438809221,
            0.487427267,0.786907310,0.408367937,0.405534192,0.250444460,
            0.995309248,0.144389588,0.739947527,0.953543606,0.680051621,
            0.388382017,0.863530727,0.006514031,0.118007779,0.924024803,
            0.384236354,0.893687694,0.626534881,0.473051932,0.750134705,
            0.241843555,0.432947602,0.689538104,0.136934797,0.150206859,
            0.474335206,0.907775349,0.525869295,0.189184225,0.854284286,
            0.831089744,0.251637345,0.587038213,0.254475554,0.237781276,
            0.827928620,0.480283781,0.594514455,0.213641488,0.024194386,
            0.536668589,0.699497811,0.892804071,0.093835427,0.731107772]
        #
        assert_almost_equal(hdquantiles(data,[0., 1.]),
                            [0.006514031, 0.995309248])
        hdq = hdquantiles(data,[0.25, 0.5, 0.75])
        assert_almost_equal(hdq, [0.253210762, 0.512847491, 0.762232442,])
        hdq = hdquantiles_sd(data,[0.25, 0.5, 0.75])
        assert_almost_equal(hdq, [0.03786954, 0.03805389, 0.03800152,], 4)
        #
        data = numpy.array(data).reshape(10,10)
        hdq = hdquantiles(data,[0.25,0.5,0.75],axis=0)
