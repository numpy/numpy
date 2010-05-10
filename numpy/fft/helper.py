"""
Discrete Fourier Transforms - helper.py
"""
# Created by Pearu Peterson, September 2002

__all__ = ['fftshift','ifftshift','fftfreq']

from numpy.core import asarray, concatenate, arange, take, \
    integer, empty
import numpy.core.numerictypes as nt
import types

def fftshift(x,axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.fftshift(freqs, axes=(1,))
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """
    tmp = asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = range(ndim)
    elif isinstance(axes, (int, nt.integer)):
        axes = (axes,)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = (n+1)//2
        mylist = concatenate((arange(p2,n),arange(p2)))
        y = take(y,mylist,k)
    return y


def ifftshift(x,axes=None):
    """
    The inverse of fftshift.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.ifftshift(np.fft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """
    tmp = asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = range(ndim)
    elif isinstance(axes, (int, nt.integer)):
        axes = (axes,)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = n-(n+1)//2
        mylist = concatenate((arange(p2,n),arange(p2)))
        y = take(y,mylist,k)
    return y

def fftfreq(n,d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array contains the frequency bins in
    cycles/unit (with zero at the start) given a window length `n` and a
    sample spacing `d`::

      f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)         if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar
        Sample spacing.

    Returns
    -------
    out : ndarray
        The array of length `n`, containing the sample frequencies.

    Examples
    --------
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = np.fft.fft(signal)
    >>> n = signal.size
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
