"""
Histogram-related functions
"""
from __future__ import division, absolute_import, print_function

import operator

import numpy as np
from numpy.compat.py3k import basestring

__all__ = ['histogram', 'histogramdd']


def _hist_bin_sqrt(x):
    """
    Square root histogram bin estimator.

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return x.ptp() / np.sqrt(x.size)


def _hist_bin_sturges(x):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return x.ptp() / (np.log2(x.size) + 1.0)


def _hist_bin_rice(x):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return x.ptp() / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x):
    """
    Scott histogram bin estimator.

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return (24.0 * np.pi**0.5 / x.size)**(1.0 / 3.0) * np.std(x)


def _hist_bin_doane(x):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            # These three operations add up to
            # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
            # but use only one temp array instead of three
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return x.ptp() / (1.0 + np.log2(x.size) +
                                    np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0


def _hist_bin_fd(x):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 1 for the number of bins.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(x):
    """
    Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x`. The Sturges estimator
    is quite good for small (<1000) datasets and is the default in the R
    language. This method gives good off the shelf behaviour.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    # There is no need to check for zero here. If ptp is, so is IQR and
    # vice versa. Either both are zero or neither one is.
    return min(_hist_bin_fd(x), _hist_bin_sturges(x))


# Private dict initialized at module load time
_hist_bin_selectors = {'auto': _hist_bin_auto,
                       'doane': _hist_bin_doane,
                       'fd': _hist_bin_fd,
                       'rice': _hist_bin_rice,
                       'scott': _hist_bin_scott,
                       'sqrt': _hist_bin_sqrt,
                       'sturges': _hist_bin_sturges}


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """
    a = np.asarray(a)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()

    if first_edge > last_edge:
        raise ValueError(
            'max must be larger than min in range parameter.')
    if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
        raise ValueError(
            'range parameter must be finite.')

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(a, bins, range, weights):
    """
    Computes the bins used internally by `histogram`.

    Parameters
    ==========
    a : ndarray
        Ravelled data array
    bins, range
        Forwarded arguments from `histogram`.
    weights : ndarray, optional
        Ravelled weights array, or None

    Returns
    =======
    bin_edges : ndarray
        Array of bin edges
    uniform_bins : (Number, Number, int):
        The upper bound, lowerbound, and number of bins, used in the optimized
        implementation of `histogram` that works on uniform bins.
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, basestring):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ValueError(
                "{!r} is not a valid estimator for `bins`".format(bin_name))
        if weights is not None:
            raise TypeError("Automated estimation of the number of "
                            "bins is not supported for weighted data")

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not np.logical_and.reduce(keep):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a)
            if width:
                n_equal_bins = int(np.ceil((last_edge - first_edge) / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError:
            raise TypeError(
                '`bins` must be an integer, a string, or an array')
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        first_edge, last_edge = _get_outer_edges(a, range)

    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError(
                '`bins` must increase monotonically, when an array')

    else:
        raise ValueError('`bins` must be 1d, when an array')

    if n_equal_bins is not None:
        # bin edges must be computed
        bin_edges = np.linspace(
            first_edge, last_edge, n_equal_bins + 1, endpoint=True)
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None


def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.

    In the context of a histogram, this makes the last bin edge inclusive
    """
    return np.concatenate((
        a.searchsorted(v[:-1], 'left'),
        a.searchsorted(v[-1:], 'right')
    ))


def histogram(a, bins=10, range=None, normed=False, weights=None,
              density=None):
    r"""
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

        .. versionadded:: 1.11.0

        If `bins` is a string from the list below, `histogram` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins (see `Notes` for more detail on
        the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good
            all around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that that takes into account data
            variability and data size.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.
    normed : bool, optional
        This keyword is deprecated in NumPy 1.6.0 due to confusing/buggy
        behavior. It will be removed in NumPy 2.0.0. Use the ``density``
        keyword instead. If ``False``, the result will contain the
        number of samples in each bin. If ``True``, the result is the
        value of the probability *density* function at the bin,
        normalized such that the *integral* over the range is 1. Note
        that this latter behavior is known to be buggy with unequal bin
        widths; use ``density`` instead.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

        Overrides the ``normed`` keyword if given.

    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words,
    if `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
    the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
    *includes* 4.

    .. versionadded:: 1.11.0

    The methods to estimate the optimal number of bins are well founded
    in literature, and are inspired by the choices R provides for
    histogram visualisation. Note that having the number of bins
    proportional to :math:`n^{1/3}` is asymptotically optimal, which is
    why it appears in most estimators. These are simply plug-in methods
    that give good starting points for number of bins. In the equations
    below, :math:`h` is the binwidth and :math:`n_h` is the number of
    bins. All estimators that compute bin counts are recast to bin width
    using the `ptp` of the data. The final bin count is obtained from
    ``np.round(np.ceil(range / h))`.

    'Auto' (maximum of the 'Sturges' and 'FD' estimators)
        A compromise to get a good value. For small datasets the Sturges
        value will usually be chosen, while larger datasets will usually
        default to FD.  Avoids the overly conservative behaviour of FD
        and Sturges for small and large datasets respectively.
        Switchover point is usually :math:`a.size \approx 1000`.

    'FD' (Freedman Diaconis Estimator)
        .. math:: h = 2 \frac{IQR}{n^{1/3}}

        The binwidth is proportional to the interquartile range (IQR)
        and inversely proportional to cube root of a.size. Can be too
        conservative for small datasets, but is quite good for large
        datasets. The IQR is very robust to outliers.

    'Scott'
        .. math:: h = \sigma \sqrt[3]{\frac{24 * \sqrt{\pi}}{n}}

        The binwidth is proportional to the standard deviation of the
        data and inversely proportional to cube root of ``x.size``. Can
        be too conservative for small datasets, but is quite good for
        large datasets. The standard deviation is not very robust to
        outliers. Values are very similar to the Freedman-Diaconis
        estimator in the absence of outliers.

    'Rice'
        .. math:: n_h = 2n^{1/3}

        The number of bins is only proportional to cube root of
        ``a.size``. It tends to overestimate the number of bins and it
        does not take into account data variability.

    'Sturges'
        .. math:: n_h = \log _{2}n+1

        The number of bins is the base 2 log of ``a.size``.  This
        estimator assumes normality of data and is too conservative for
        larger, non-normal datasets. This is the default method in R's
        ``hist`` method.

    'Doane'
        .. math:: n_h = 1 + \log_{2}(n) +
                        \log_{2}(1 + \frac{|g_1|}{\sigma_{g_1}})

            g_1 = mean[(\frac{x - \mu}{\sigma})^3]

            \sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}

        An improved version of Sturges' formula that produces better
        estimates for non-normal datasets. This estimator attempts to
        account for the skew of the data.

    'Sqrt'
        .. math:: n_h = \sqrt n
        The simplest and fastest estimator. Only takes into account the
        data size.

    Examples
    --------
    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([ 0.25,  0.25,  0.25,  0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([ 0.5,  0. ,  0.5,  0. ,  0. ,  0.5,  0. ,  0.5,  0. ,  0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> np.sum(hist * np.diff(bin_edges))
    1.0

    .. versionadded:: 1.11.0

    Automated Bin Selection Methods example, using 2 peak random data
    with 2000 points:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.RandomState(10)  # deterministic random data
    >>> a = np.hstack((rng.normal(size=1000),
    ...                rng.normal(loc=5, scale=2, size=1000)))
    >>> plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    >>> plt.title("Histogram with 'auto' bins")
    >>> plt.show()

    """
    a, weights = _ravel_and_check_weights(a, weights)

    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = np.dtype(np.intp)
    else:
        ntype = weights.dtype

    # We set a block size, as this allows us to iterate over chunks when
    # computing histograms, to minimize memory usage.
    BLOCK = 65536

    # The fast path uses bincount, but that only works for certain types
    # of weight
    simple_weights = (
        weights is None or
        np.can_cast(weights.dtype, np.double) or
        np.can_cast(weights.dtype, complex)
    )

    if uniform_bins is not None and simple_weights:
        # Fast algorithm for equal bins
        # We now convert values of a to bin indices, under the assumption of
        # equal bin widths (which is valid here).
        first_edge, last_edge, n_equal_bins = uniform_bins

        # Initialize empty histogram
        n = np.zeros(n_equal_bins, ntype)

        # Pre-compute histogram scaling factor
        norm = n_equal_bins / (last_edge - first_edge)

        # We iterate over blocks here for two reasons: the first is that for
        # large arrays, it is actually faster (for example for a 10^8 array it
        # is 2x as fast) and it results in a memory footprint 3x lower in the
        # limit of large arrays.
        for i in np.arange(0, len(a), BLOCK):
            tmp_a = a[i:i+BLOCK]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i:i + BLOCK]

            # Only include values in the right range
            keep = (tmp_a >= first_edge)
            keep &= (tmp_a <= last_edge)
            if not np.logical_and.reduce(keep):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]
            tmp_a_data = tmp_a.astype(float)
            tmp_a = tmp_a_data - first_edge
            tmp_a *= norm

            # Compute the bin indices, and for values that lie exactly on
            # last_edge we need to subtract one
            indices = tmp_a.astype(np.intp)
            indices[indices == n_equal_bins] -= 1

            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = tmp_a_data < bin_edges[indices]
            indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = ((tmp_a_data >= bin_edges[indices + 1])
                         & (indices != n_equal_bins - 1))
            indices[increment] += 1

            # We now compute the histogram using bincount
            if ntype.kind == 'c':
                n.real += np.bincount(indices, weights=tmp_w.real,
                                      minlength=n_equal_bins)
                n.imag += np.bincount(indices, weights=tmp_w.imag,
                                      minlength=n_equal_bins)
            else:
                n += np.bincount(indices, weights=tmp_w,
                                 minlength=n_equal_bins).astype(ntype)
    else:
        # Compute via cumulative histogram
        cum_n = np.zeros(bin_edges.shape, ntype)
        if weights is None:
            for i in np.arange(0, len(a), BLOCK):
                sa = np.sort(a[i:i+BLOCK])
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = np.zeros(1, dtype=ntype)
            for i in np.arange(0, len(a), BLOCK):
                tmp_a = a[i:i+BLOCK]
                tmp_w = weights[i:i+BLOCK]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = np.concatenate((zero, sw.cumsum()))
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]

        n = np.diff(cum_n)

    # density overrides the normed keyword
    if density is not None:
        normed = False

    if density:
        db = np.array(np.diff(bin_edges), float)
        return n/db/n.sum(), bin_edges
    elif normed:
        # deprecated, buggy behavior. Remove for NumPy 2.0.0
        db = np.array(np.diff(bin_edges), float)
        return n/(n*db).sum(), bin_edges
    else:
        return n, bin_edges


def histogramdd(sample, bins=10, range=None, normed=False, weights=None):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : array_like
        The data to be histogrammed. It must be an (N,D) array or data
        that can be converted to such. The rows of the resulting array
        are the coordinates of points in a D dimensional polytope.
    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True,
        returns the bin density ``bin_count / sample_count / bin_volume``.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if normed is True. If normed is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See normed and weights
        for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram

    Examples
    --------
    >>> r = np.random.randn(100,3)
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        # Handle empty input. Range can't be determined in that case, use 0-1.
        if N == 0:
            smin = np.zeros(D)
            smax = np.ones(D)
        else:
            smin = np.atleast_1d(np.array(sample.min(0), float))
            smax = np.atleast_1d(np.array(sample.max(0), float))
    else:
        if not np.all(np.isfinite(range)):
            raise ValueError(
                'range parameter must be finite.')
        smin = np.zeros(D)
        smax = np.zeros(D)
        for i in np.arange(D):
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in np.arange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # avoid rounding issues for comparisons when dealing with inexact types
    if np.issubdtype(sample.dtype, np.inexact):
        edge_dt = sample.dtype
    else:
        edge_dt = float
    # Create edge arrays
    for i in np.arange(D):
        if np.isscalar(bins[i]):
            if bins[i] < 1:
                raise ValueError(
                    "Element at index %s in `bins` should be a positive "
                    "integer." % i)
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i]-1, dtype=edge_dt)
        else:
            edges[i] = np.asarray(bins[i], edge_dt)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])
        if np.any(np.asarray(dedges[i]) <= 0):
            raise ValueError(
                "Found bin edge of size <= 0. Did you specify `bins` with"
                "non-monotonic sequence?")

    nbin = np.asarray(nbin)

    # Handle empty input.
    if N == 0:
        return np.zeros(nbin-2), edges

    # Compute the bin number each sample falls into.
    Ncount = {}
    for i in np.arange(D):
        Ncount[i] = np.digitize(sample[:, i], edges[i])

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in np.arange(D):
        # Rounding precision
        mindiff = dedges[i].min()
        if not np.isinf(mindiff):
            decimal = int(-np.log10(mindiff)) + 6
            # Find which points are on the rightmost edge.
            not_smaller_than_edge = (sample[:, i] >= edges[i][-1])
            on_edge = (np.around(sample[:, i], decimal) ==
                       np.around(edges[i][-1], decimal))
            # Shift these points one bin to the left.
            Ncount[i][np.nonzero(on_edge & not_smaller_than_edge)[0]] -= 1

    # Flattened histogram matrix (1D)
    # Reshape is used so that overlarge arrays
    # will raise an error.
    hist = np.zeros(nbin, float).reshape(-1)

    # Compute the sample indices in the flattened histogram matrix.
    ni = nbin.argsort()
    xy = np.zeros(N, int)
    for i in np.arange(0, D-1):
        xy += Ncount[ni[i]] * nbin[ni[i+1:]].prod()
    xy += Ncount[ni[-1]]

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    if len(xy) == 0:
        return np.zeros(nbin-2, int), edges

    flatcount = np.bincount(xy, weights)
    a = np.arange(len(flatcount))
    hist[a] = flatcount

    # Shape into a proper matrix
    hist = hist.reshape(np.sort(nbin))
    for i in np.arange(nbin.size):
        j = ni.argsort()[i]
        hist = hist.swapaxes(i, j)
        ni[i], ni[j] = ni[j], ni[i]

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D*[slice(1, -1)]
    hist = hist[core]

    # Normalize if normed is True
    if normed:
        s = hist.sum()
        for i in np.arange(D):
            shape = np.ones(D, int)
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist, edges
