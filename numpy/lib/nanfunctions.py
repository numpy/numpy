"""
Functions that ignore NaN.

Functions
---------

- `nanmin` -- minimum non-NaN value
- `nanmax` -- maximum non-NaN value
- `nanargmin` -- index of minimum non-NaN value
- `nanargmax` -- index of maximum non-NaN value
- `nansum` -- sum of non-NaN values
- `nanmean` -- mean of non-NaN values
- `nanvar` -- variance of non-NaN values
- `nanstd` -- standard deviation of non-NaN values

Classes
-------
- `NanWarning` -- Warning raised by nanfunctions

"""
from __future__ import division, absolute_import, print_function

import warnings
import numpy as np

__all__ = [
    'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
    'nanvar', 'nanstd', 'NanWarning'
    ]

class NanWarning(RuntimeWarning): pass


def _replace_nan(a, val):
    """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.

    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.

    """
    is_new = not isinstance(a, np.ndarray)
    if is_new:
        a = np.array(a)
    if not issubclass(a.dtype.type, np.inexact):
        return a, None
    if not is_new:
        # need copy
        a = np.array(a, subok=True)

    mask = np.isnan(a)
    np.copyto(a, val, where=mask)
    return a, mask


def _copyto(a, val, mask):
    """
    Replace values in `a` with NaN where `mask` is True.  This differs from
    copyto in that it will deal with the case where `a` is a numpy scalar.

    Parameters
    ----------
    a : ndarray or numpy scalar
        Array or numpy scalar some of whose values are to be replaced
        by val.
    val : numpy scalar
        Value used a replacement.
    mask : ndarray, scalar
        Boolean array. Where True the corresponding element of `a` is
        replaced by `val`. Broadcasts.

    Returns
    -------
    res : ndarray, scalar
        Array with elements replaced or scalar `val`.

    """
    if isinstance(a, np.ndarray):
        np.copyto(a, val, where=mask, casting='unsafe')
    else:
        a = a.dtype.type(val)
    return a


def _divide_by_count(a, b, out=None):
    """
    Compute a/b ignoring invalid results. If `a` is an array the division
    is done in place. If `a` is a scalar, then its type is preserved in the
    output. If out is None, then then a is used instead so that the
    division is in place.

    Parameters
    ----------
    a : {ndarray, numpy scalar}
        Numerator. Expected to be of inexact type but not checked.
    b : {ndarray, numpy scalar}
        Denominator.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.

    Returns
    -------
    ret : {ndarray, numpy scalar}
        The return value is a/b. If `a` was an ndarray the division is done
        in place. If `a` is a numpy scalar, the division preserves its type.

    """
    with np.errstate(invalid='ignore'):
        if isinstance(a, np.ndarray):
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                return a.dtype.type(a / b)
            else:
                # This is questionable, but currently a numpy scalar can
                # be output to a zero dimensional array.
                return np.divide(a, b, out=out, casting='unsafe')


def nanmin(a, axis=None, out=None, keepdims=False):
    """
    Return the minimum of an array or minimum along an axis, ignoring any
    NaNs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If `a` is not
        an array, a conversion is attempted.
    axis : int, optional
        Axis along which the minimum is computed. The default is to compute
        the minimum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.

        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        .. versionadded:: 1.8.0

    Returns
    -------
    nanmin : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an ndarray scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amax, fmax, maximum

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative infinity
    is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.min.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmin(a)
    1.0
    >>> np.nanmin(a, axis=0)
    array([ 1.,  2.])
    >>> np.nanmin(a, axis=1)
    array([ 1.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmin([1, 2, np.nan, np.inf])
    1.0
    >>> np.nanmin([1, 2, np.nan, np.NINF])
    -inf

    """
    return np.fmin.reduce(a, axis=axis, out=out, keepdims=keepdims)


def nanmax(a, axis=None, out=None, keepdims=False):
    """
    Return the maximum of an array or maximum along an axis, ignoring any NaNs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If `a` is not
        an array, a conversion is attempted.
    axis : int, optional
        Axis along which the maximum is computed. The default is to compute
        the maximum of the flattened array.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.

        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        .. versionadded:: 1.8.0

    Returns
    -------
    nanmax : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, an ndarray scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amin, fmin, minimum

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative infinity
    is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.max.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmax(a)
    3.0
    >>> np.nanmax(a, axis=0)
    array([ 3.,  2.])
    >>> np.nanmax(a, axis=1)
    array([ 2.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmax([1, 2, np.nan, np.NINF])
    2.0
    >>> np.nanmax([1, 2, np.nan, np.inf])
    inf

    """
    return np.fmax.reduce(a, axis=axis, out=out, keepdims=keepdims)


def nanargmin(a, axis=None):
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slices, the negative number ``np.iinfo('intp').min``
    is returned. It is platform dependent. Warning: the results cannot be
    trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmin, nanargmax

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmin(a)
    0
    >>> np.nanargmin(a)
    2
    >>> np.nanargmin(a, axis=0)
    array([1, 1])
    >>> np.nanargmin(a, axis=1)
    array([1, 0])

    """
    a, mask = _replace_nan(a, np.inf)
    if mask is None:
        return np.argmin(a, axis)
    # May later want to do something special for all nan slices.
    mask = mask.all(axis=axis)
    ind = np.argmin(a, axis)
    if mask.any():
        warnings.warn("All NaN axis detected.", NanWarning)
        ind =_copyto(ind, np.iinfo(np.intp).min, mask)
    return ind


def nanargmax(a, axis=None):
    """
    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For  all-NaN slices, the negative number ``np.iinfo('intp').min``
    is returned. It is platform dependent. Warning: the results cannot be
    trusted if a slice contains only NaNs and -Infs.


    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    See Also
    --------
    argmax, nanargmin

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmax(a)
    0
    >>> np.nanargmax(a)
    1
    >>> np.nanargmax(a, axis=0)
    array([1, 0])
    >>> np.nanargmax(a, axis=1)
    array([1, 1])

    """
    a, mask = _replace_nan(a, -np.inf)
    if mask is None:
        return np.argmax(a, axis)
    # May later want to do something special for all nan slices.
    mask = mask.all(axis=axis)
    ind = np.argmax(a, axis)
    if mask.any():
        warnings.warn("All NaN axis detected.", NanWarning)
        ind = _copyto(ind, np.iinfo(np.intp).min, mask)
    return ind


def nansum(a, axis=None, dtype=None, out=None, keepdims=0):
    """
    Return the sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero.

    FutureWarning: In Numpy versions <= 1.8 Nan is returned for slices that
    are all-NaN or empty. In later versions zero will be returned.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the sum is computed. The default is to compute
        the sum of the flattened array.
    dtype : data-type, optional
        Type to use in computing the sum.  For integer inputs, the default
        is the same as `int64`. For inexact inputs, it must be inexact.

        .. versionadded:: 1.8.0
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details. The casting of NaN to integer can
        yield unexpected results.

        .. versionadded:: 1.8.0
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will
        broadcast correctly against the original `arr`.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray or numpy scalar

    See Also
    --------
    numpy.sum : Sum across array propagating NaNs.
    isnan : Show which elements are NaN.
    isfinite: Show which elements are not NaN or +/-inf.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Arithmetic is modular when using integer types (all elements of `a` must
    be finite i.e. no elements that are NaNs, positive infinity and negative
    infinity because NaNs are floating point types), and no error is raised
    on overflow.


    Examples
    --------
    >>> np.nansum(1)
    1
    >>> np.nansum([1])
    1
    >>> np.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> np.nansum(a)
    3.0
    >>> np.nansum(a, axis=0)
    array([ 2.,  1.])
    >>> np.nansum([1, np.nan, np.inf])
    inf
    >>> np.nansum([1, np.nan, np.NINF])
    -inf
    >>> np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present
    nan

    """
    a, mask = _replace_nan(a, 0)
    # In version 1.9 uncomment the following line and delete the rest.
    #return a.sum(axis, dtype, out, keepdims)
    warnings.warn("In Numpy 1.9 the sum along empty slices will be zero.",
            FutureWarning)

    if mask is None:
        return a.sum(axis, dtype, out, keepdims)
    mask = mask.all(axis, keepdims=keepdims)
    tot = np.add.reduce(a, axis, dtype, out, keepdims)
    if mask.any():
        tot = _copyto(tot, np.nan, mask)
    return tot


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `NanWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the
    same precision the input has.  Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32`.
    Specifying a higher-precision accumulator using the `dtype` keyword
    can alleviate this issue.

    Examples
    --------
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanmean(a)
    2.6666666666666665
    >>> np.nanmean(a, axis=0)
    array([ 2.,  4.])
    >>> np.nanmean(a, axis=1)
    array([ 1.,  3.5])

    """
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis, dtype=dtype, out=out, keepdims=keepdims)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    # The warning context speeds things up.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cnt = np.add.reduce(~mask, axis, dtype=np.intp, keepdims=keepdims)
        tot = np.add.reduce(arr, axis, dtype=dtype, out=out, keepdims=keepdims)
        avg = _divide_by_count(tot, cnt, out=out)

    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", NanWarning)
        # NaN is the only possible bad value, so no further
        # action is needed to handle bad results.
    return avg


def nanvar(a, axis=None, dtype=None, out=None, ddof=0,
           keepdims=False):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    For all-NaN slices, NaN is returned and a `NanWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float32`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If `out` is None, return a new array containing the variance,
        otherwise return a reference to the output array. If ddof is >= the
        number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd, nanmean
    numpy.doc.ufuncs : Section "Output arguments"

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    Examples
    --------
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.var(a)
    1.5555555555555554
    >>> np.nanvar(a, axis=0)
    array([ 1.,  0.])
    >>> np.nanvar(a, axis=1)
    array([ 0.,  0.25])

    """
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.var(arr, axis, dtype=dtype, out=out, keepdims=keepdims)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Compute mean
        cnt = np.add.reduce(~mask, axis, dtype=np.intp, keepdims=True)
        tot = np.add.reduce(arr, axis, dtype=dtype, keepdims=True)
        avg = np.divide(tot, cnt, out=tot)

        # Compute squared deviation from mean.
        x = arr - avg
        np.copyto(x, 0, where=mask)
        if issubclass(arr.dtype.type, np.complexfloating):
            sqr = np.multiply(x, x.conj(), out=x).real
        else:
            sqr = np.multiply(x, x, out=x)

        # adjust cnt.
        if not keepdims:
            cnt = cnt.squeeze(axis)
        cnt -= ddof

        # Compute variance.
        var = np.add.reduce(sqr, axis, dtype=dtype, out=out, keepdims=keepdims)
        var = _divide_by_count(var, cnt)

    isbad = (cnt <= 0)
    if isbad.any():
        warnings.warn("Degrees of freedom <= 0 for slice.", NanWarning)
        # NaN, inf, or negative numbers are all possible bad
        # values, so explicitly replace them with NaN.
        var = _copyto(var, np.nan, isbad)
    return var


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """
    Compute the standard deviation along the specified axis, while
    ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the non-NaN array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    For all-NaN slices, NaN is returned and a `NanWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of the non-NaN values.
    axis : int, optional
        Axis along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non-NaN
        elements.  By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard
        deviation, otherwise return a reference to the output array. If
        ddof is >= the number of non-NaN elements in a slice or the slice
        contains only NaNs, then the result for that slice is NaN.

    See Also
    --------
    var, mean, std
    nanvar, nanmean
    numpy.doc.ufuncs : Section "Output arguments"

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Note that, for complex numbers, `std` takes the absolute
    value before squaring, so that the result is always real and nonnegative.

    For floating-point input, the *std* is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example below).
    Specifying a higher-accuracy accumulator using the `dtype` keyword can
    alleviate this issue.

    Examples
    --------
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanstd(a)
    1.247219128924647
    >>> np.nanstd(a, axis=0)
    array([ 1.,  0.])
    >>> np.nanstd(a, axis=1)
    array([ 0.,  0.5])

    """
    var = nanvar(a, axis, dtype, out, ddof, keepdims)
    if isinstance(var, np.ndarray):
        std = np.sqrt(var, out=var)
    else:
        std = var.dtype.type(np.sqrt(var))
    return std
