"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.
"""
__all__ = ['fix', 'isneginf', 'isposinf', 'log2']

import numpy.core.numeric as nx

def fix(x, y=None):
    """
    Round to nearest integer towards zero.

    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters
    ----------
    x : array_like
        An array of floats to be rounded
    y : ndarray, optional
        Output array

    Returns
    -------
    out : ndarray of floats
        The array of rounded numbers

    See Also
    --------
    floor : Round downwards
    around : Round to given number of decimals

    Examples
    --------
    >>> np.fix(3.14)
    3.0
    >>> np.fix(3)
    3.0
    >>> np.fix([2.1, 2.9, -2.1, -2.9])
    array([ 2.,  2., -2., -2.])

    """
    x = nx.asanyarray(x)
    if y is None:
        y = nx.zeros_like(x)
    y1 = nx.floor(x)
    y2 = nx.ceil(x)
    y[...] = nx.where(x >= 0, y1, y2)
    return y

def isposinf(x, y=None):
    """
    Shows which elements of the input are positive infinity.

    Returns a numpy array resulting from an element-wise test for positive
    infinity.

    Parameters
    ----------
    x : array_like
      The input array.
    y : array_like
      A boolean array with the same shape as `x` to store the result.

    Returns
    -------
    y : ndarray
      A numpy boolean array with the same dimensions as the input.
      If second argument is not supplied then a numpy boolean array is returned
      with values True where the corresponding element of the input is positive
      infinity and values False where the element of the input is not positive
      infinity.

      If second argument is supplied then an numpy integer array is returned
      with values 1 where the corresponding element of the input is positive
      positive infinity.

    See Also
    --------
    isinf : Shows which elements are negative or positive infinity.
    isneginf : Shows which elements are negative infinity.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite: Shows which elements are not: Not a number, positive and
             negative infinity

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    Errors result if second argument is also supplied with scalar input or
    if first and second arguments have different shapes.

    Numpy's definitions for positive infinity (PINF) and negative infinity
    (NINF) may be change in the future versions.


    Examples
    --------
    >>> np.isposinf(np.PINF)
    array(True, dtype=bool)
    >>> np.isposinf(np.inf)
    array(True, dtype=bool)
    >>> np.isposinf(np.NINF)
    array(False, dtype=bool)
    >>> np.isposinf([-np.inf, 0., np.inf])
    array([False, False,  True], dtype=bool)
    >>> x=np.array([-np.inf, 0., np.inf])
    >>> y=np.array([2,2,2])
    >>> np.isposinf(x,y)
    array([1, 0, 0])
    >>> y
    array([1, 0, 0])

    """
    if y is None:
        x = nx.asarray(x)
        y = nx.empty(x.shape, dtype=nx.bool_)
    nx.logical_and(nx.isinf(x), ~nx.signbit(x), y)
    return y

def isneginf(x, y=None):
    """
    Return True where x is -infinity, and False otherwise.

    Parameters
    ----------
    x : array_like
      The input array.
    y : array_like
      A boolean array with the same shape as `x` to store the result.

    Returns
    -------
    y : ndarray
      A boolean array where y[i] = True only if x[i] = -Inf.

    See Also
    --------
    isposinf, isfinite

    Examples
    --------
    >>> np.isneginf([-np.inf, 0., np.inf])
    array([ True, False, False], dtype=bool)

    """
    if y is None:
        x = nx.asarray(x)
        y = nx.empty(x.shape, dtype=nx.bool_)
    nx.logical_and(nx.isinf(x), nx.signbit(x), y)
    return y

_log2 = nx.log(2)
def log2(x, y=None):
    """
    Return the base 2 logarithm.

    Parameters
    ----------
    x : array_like
      Input array.
    y : array_like
      Optional output array with the same shape as `x`.

    Returns
    -------
    y : ndarray
      The logarithm to the base 2 of `x` elementwise.
      NaNs are returned where `x` is negative.

    See Also
    --------
    log, log1p, log10

    Examples
    --------
    >>> np.log2([-1,2,4])
    array([ NaN,   1.,   2.])

    """
    x = nx.asanyarray(x)
    if y is None:
        y = nx.log(x)
    else:
        nx.log(x, y)
    y /= _log2
    return y
