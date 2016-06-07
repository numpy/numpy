"""
Discrete Fourier Transforms - helper.py

"""
from __future__ import division, absolute_import, print_function

from collections import OrderedDict

from numpy.compat import integer_types
from numpy.core import (
        asarray, concatenate, arange, take, integer, empty
        )

# Created by Pearu Peterson, September 2002

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']

integer_types = integer_types + (integer,)


def fftshift(x, axes=None):
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
        axes = list(range(ndim))
    elif isinstance(axes, integer_types):
        axes = (axes,)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = (n+1)//2
        mylist = concatenate((arange(p2, n), arange(p2)))
        y = take(y, mylist, k)
    return y


def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

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
        axes = list(range(ndim))
    elif isinstance(axes, integer_types):
        axes = (axes,)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = n-(n+1)//2
        mylist = concatenate((arange(p2, n), arange(p2)))
        y = take(y, mylist, k)
    return y


def fftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length `n` containing the sample frequencies.

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
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val
    #return hstack((arange(0,(n-1)/2 + 1), arange(-(n/2),0))) / (n*d)


def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = np.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0/(n*d)
    N = n//2 + 1
    results = arange(0, N, dtype=int)
    return results * val


class _FFTCache(object):
    """
    Cache for the FFT init functions as an LRU (least recently used) cache.

    Parameters
    ----------
    max_size_in_mb : int
        Maximum memory usage of the cache before items are being evicted.
    max_item_count : int
        Maximum item count of the cache before items are being evicted.

    Notes
    -----
    Items will be evicted if either limit has been reached upon getting and
    setting. The maximum memory usages is not strictly the given
    ``max_size_in_mb`` but rather
    ``max(max_size_in_mb, 1.5 * size_of_largest_item)``. Thus the cache will
    never be completely cleared - at least one item will remain and a single
    large item can cause the cache to retain several smaller items even if the
    given maximum cache size has been exceeded.
    """
    def __init__(self, max_size_in_mb, max_item_count):
        self._max_size_in_bytes = max_size_in_mb * 1024 ** 2
        self._max_item_count = max_item_count
        # Much simpler than inheriting from it and having to work around
        # recursive behaviour.
        self._dict = OrderedDict()

    def setdefault(self, key, value):
        return self._dict.setdefault(key, value)

    def __getitem__(self, key):
        # pop + add to move it to the end.
        value = self._dict.pop(key)
        self._dict[key] = value
        self._prune_dict()
        return value

    def __setitem__(self, key, value):
        # Just setting is it not enough to move it to the end if it already
        # exists.
        try:
            del self._dict[key]
        except:
            pass
        self._dict[key] = value
        self._prune_dict()

    def _prune_dict(self):
        # Always keep at least one item.
        while len(self._dict) > 1 and (
                len(self._dict) > self._max_item_count or self._check_size()):
            self._dict.popitem(last=False)

    def _check_size(self):
        item_sizes = [sum(_j.nbytes for _j in _i)
                      for _i in self._dict.values() if _i]
        max_size = max(self._max_size_in_bytes, 1.5 * max(item_sizes))
        return sum(item_sizes) > max_size
