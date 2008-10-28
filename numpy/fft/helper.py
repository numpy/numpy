"""
Discrete Fourier Transforms - helper.py
"""
# Created by Pearu Peterson, September 2002

__all__ = ['fftshift','ifftshift','fftfreq']

from numpy.core import asarray, concatenate, arange, take, \
    integer, empty
import types

def fftshift(x,axes=None):
    """
    Shift zero-frequency component to center of spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    If len(x) is even then the Nyquist component is y[0].

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None which shifts all axes.

    See Also
    --------
    ifftshift

    """
    tmp = asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = range(ndim)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = (n+1)/2
        mylist = concatenate((arange(p2,n),arange(p2)))
        y = take(y,mylist,k)
    return y


def ifftshift(x,axes=None):
    """
    Inverse of fftshift.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None which is over all axes.

    See Also
    --------
    fftshift

    """
    tmp = asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = range(ndim)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = n-(n+1)/2
        mylist = concatenate((arange(p2,n),arange(p2)))
        y = take(y,mylist,k)
    return y

def fftfreq(n,d=1.0):
    """
    Discrete Fourier Transform sample frequencies.

    The returned float array contains the frequency bins in
    cycles/unit (with zero at the start) given a window length `n` and a
    sample spacing `d`.
    ::

      f = [0,1,...,n/2-1,-n/2,...,-1]/(d*n)         if n is even
      f = [0,1,...,(n-1)/2,-(n-1)/2,...,-1]/(d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar
        Sample spacing.

    Returns
    -------
    out : ndarray, shape(`n`,)
        Sample frequencies.

    Examples
    --------
    >>> signal = np.array([-2.,  8., -6.,  4.,  1., 0.,  3.,  5.])
    >>> fourier = np.fft.fft(signal)
    >>> n = len(signal)
    >>> timestep = 0.1
    >>> freq = np.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])

    """
    assert isinstance(n,types.IntType) or isinstance(n, integer)
    val = 1.0/(n*d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0,N,dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2),0,dtype=int)
    results[N:] = p2
    return results * val
    #return hstack((arange(0,(n-1)/2 + 1), arange(-(n/2),0))) / (n*d)
