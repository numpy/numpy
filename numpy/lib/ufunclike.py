"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.
"""
__all__ = ['fix', 'isneginf', 'isposinf', 'log2']

import numpy.core.numeric as nx

def fix(x, y=None):
    """ Round x to nearest integer towards zero.
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
    Return True where x is +infinity, and False otherwise.

    Parameters
    ----------
    x : array_like
      The input array.
    y : array_like
      A boolean array with the same shape as `x` to store the result.

    Returns
    -------
    y : ndarray
      A boolean array where y[i] = True only if x[i] = +Inf.

    See Also
    --------
    isneginf, isfinite

    Examples
    --------
    >>> np.isposinf([-np.inf, 0., np.inf])
    array([ False, False, True], dtype=bool)

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
    y : {ndarray, scalar}
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
