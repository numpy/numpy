"""Functions that ignore nan.

"""
from __future__ import division, absolute_import, print_function

import numpy as np

__all__ = [
    'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
    'nanvar', 'nanstd'
    ]


def _nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    # Using array() instead of asanyarray() because the former always
    # makes a copy, which is important due to the copyto() action later
    arr = np.array(a, subok=True)
    mask = np.isnan(arr)

    # Cast bool, unsigned int, and int to float64
    if np.dtype is None and issubclass(arr.dtype.type, (np.integer, np.bool_)):
        ret = np.add.reduce(arr, axis=axis, dtype='f8',
                            out=out, keepdims=keepdims)
    else:
        np.copyto(arr, 0.0, where=mask)
        ret = np.add.reduce(arr, axis=axis, dtype=dtype,
                            out=out, keepdims=keepdims)
    rcount = (~mask).sum(axis=axis)
    if isinstance(ret, np.ndarray):
        ret = np.true_divide(ret, rcount, out=ret, casting='unsafe',
                             subok=False)
    else:
        ret = ret / rcount
    return ret


def _nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    # Using array() instead of asanyarray() because the former always
    # makes a copy, which is important due to the copyto() action later
    arr = np.array(a, subok=True)
    mask = np.isnan(arr)

    # First compute the mean, saving 'rcount' for reuse later
    if dtype is None and issubclass(arr.dtype.type, (np.integer, np.bool_)):
        arrmean = np.add.reduce(arr, axis=axis, dtype='f8', keepdims=True)
    else:
        np.copyto(arr, 0.0, where=mask)
        arrmean = np.add.reduce(arr, axis=axis, dtype=dtype, keepdims=True)
    rcount = (~mask).sum(axis=axis, keepdims=True)
    if isinstance(arrmean, np.ndarray):
        arrmean = np.true_divide(arrmean, rcount,
                            out=arrmean, casting='unsafe', subok=False)
    else:
        arrmean = arrmean / rcount

    # arr - arrmean
    x = arr - arrmean
    np.copyto(x, 0.0, where=mask)

    # (arr - arrmean) ** 2
    if issubclass(arr.dtype.type, np.complex_):
        x = np.multiply(x, np.conjugate(x), out=x).real
    else:
        x = np.multiply(x, x, out=x)

    # add.reduce((arr - arrmean) ** 2, axis)
    ret = np.add.reduce(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # add.reduce((arr - arrmean) ** 2, axis) / (n - ddof)
    if not keepdims and isinstance(rcount, np.ndarray):
        rcount = rcount.squeeze(axis=axis)
    rcount -= ddof
    if isinstance(ret, np.ndarray):
        ret = np.true_divide(ret, rcount, out=ret, casting='unsafe', subok=False)
    else:
        ret = ret / rcount

    return ret


def _nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = _nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                  keepdims=keepdims)

    if isinstance(ret, np.ndarray):
        ret = np.sqrt(ret, out=ret)
    else:
        ret = np.sqrt(ret)

    return ret


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
    y = np.array(a, subok=True)

    # We only need to take care of NaN's in floating point arrays
    dt = y.dtype
    if np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.bool_):
        return op(y, axis=axis)

    mask = np.isnan(a)
    # y[mask] = fill
    # We can't use fancy indexing here as it'll mess w/ MaskedArrays
    # Instead, let's fill the array directly...
    np.copyto(y, fill, where=mask)
    res = op(y, axis=axis)
    mask_all_along_axis = mask.all(axis=axis)

    # Along some axes, only nan's were encountered.  As such, any values
    # calculated along that axis should be set to nan.
    if mask_all_along_axis.any():
        if np.isscalar(res):
            res = np.nan
        else:
            res[mask_all_along_axis] = np.nan

    return res


def nansum(a, axis=None):
    """
    Return the sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the sum is computed. The default is to compute
        the sum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as a, with the specified axis removed.
        If a is a 0-d array, or if axis is None, a scalar is returned with
        the same dtype as `a`.

    See Also
    --------
    numpy.sum : Sum across array including Not a Numbers.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite: Shows which elements are not: Not a Number, positive and
             negative infinity

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

    When positive infinity and negative infinity are present

    >>> np.nansum([1, np.nan, np.inf])
    inf
    >>> np.nansum([1, np.nan, np.NINF])
    -inf
    >>> np.nansum([1, np.nan, np.inf, np.NINF])
    nan

    """
    return _nanop(np.sum, 0, a, axis)


def nanmin(a, axis=None):
    """
    Return the minimum of an array or minimum along an axis, ignoring any NaNs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If `a` is not
        an array, a conversion is attempted.
    axis : int, optional
        Axis along which the minimum is computed. The default is to compute
        the minimum of the flattened array.

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
    a = np.asanyarray(a)
    if axis is not None:
        return np.fmin.reduce(a, axis)
    else:
        return np.fmin.reduce(a.flat)


def nanargmin(a, axis=None):
    """
    Return indices of the minimum values over an axis, ignoring NaNs.

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
    return _nanop(np.argmin, np.inf, a, axis)


def nanmax(a, axis=None):
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
    a = np.asanyarray(a)
    if axis is not None:
        return np.fmax.reduce(a, axis)
    else:
        return np.fmax.reduce(a.flat)


def nanargmax(a, axis=None):
    """
    Return indices of the maximum values over an axis, ignoring NaNs.

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
    return _nanop(np.argmax, -np.inf, a, axis)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

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
        is `float64`; for floating point inputs, it is the same as the
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
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-nan elements along the axis
    divided by the number of non-nan elements.

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
    if not (type(a) is np.ndarray):
        try:
            mean = a.nanmean
            return mean(axis=axis, dtype=dtype, out=out)
        except AttributeError:
            pass

    return _nanmean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """
    Compute the standard deviation along the specified axis, while
    ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the non-NaN array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

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
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

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

    if not (type(a) is np.ndarray):
        try:
            nanstd = a.nanstd
            return nanstd(axis=axis, dtype=dtype, out=out, ddof=ddof)
        except AttributeError:
            pass

    return _nanstd(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                   keepdims=keepdims)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0,
           keepdims=False):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

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
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

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
    if not (type(a) is np.ndarray):
        try:
            nanvar = a.nanvar
            return nanvar(axis=axis, dtype=dtype, out=out, ddof=ddof)
        except AttributeError:
            pass

    return _nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                            keepdims=keepdims)
