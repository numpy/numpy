"""
Discrete Fourier Transforms - FFT.py

The underlying code for these functions is an f2c translated and modified
version of the FFTPACK routines.

fft(a, n=None, axis=-1)
ifft(a, n=None, axis=-1)
rfft(a, n=None, axis=-1)
irfft(a, n=None, axis=-1)
hfft(a, n=None, axis=-1)
ihfft(a, n=None, axis=-1)
fftn(a, s=None, axes=None)
ifftn(a, s=None, axes=None)
rfftn(a, s=None, axes=None)
irfftn(a, s=None, axes=None)
fft2(a, s=None, axes=(-2,-1))
ifft2(a, s=None, axes=(-2, -1))
rfft2(a, s=None, axes=(-2,-1))
irfft2(a, s=None, axes=(-2, -1))
"""
__all__ = ['fft','ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn',
           'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'refft', 'irefft','refftn','irefftn', 'refft2', 'irefft2']

from numpy.core import asarray, zeros, swapaxes, shape, conjugate, \
     take
import fftpack_lite as fftpack
from helper import *

_fft_cache = {}
_real_fft_cache = {}

def _raw_fft(a, n=None, axis=-1, init_function=fftpack.cffti,
             work_function=fftpack.cfftf, fft_cache = _fft_cache ):
    a = asarray(a)

    if n is None:
        n = a.shape[axis]

    if n < 1:
        raise ValueError("Invalid number of FFT data points (%d) specified." % n)

    try:
        wsave = fft_cache[n]
    except(KeyError):
        wsave = init_function(n)
        fft_cache[n] = wsave

    if a.shape[axis] != n:
        s = list(a.shape)
        if s[axis] > n:
            index = [slice(None)]*len(s)
            index[axis] = slice(0,n)
            a = a[index]
        else:
            index = [slice(None)]*len(s)
            index[axis] = slice(0,s[axis])
            s[axis] = n
            z = zeros(s, a.dtype.char)
            z[index] = a
            a = z

    if axis != -1:
        a = swapaxes(a, axis, -1)
    r = work_function(a, wsave)
    if axis != -1:
        r = swapaxes(r, axis, -1)
    return r


def fft(a, n=None, axis=-1):
    """
    Compute the one dimensional fft on a given axis.

    Return the n point discrete Fourier transform of a. n defaults to the
    length of a. If n is larger than the length of a, then a will be
    zero-padded to make up the difference.  If n is smaller than the length of
    a, only the first n items in a will be used.

    Parameters
    ----------

    a : array
        input array
    n : int
        length of the fft
    axis : int
        axis over which to compute the fft

    Notes
    -----

    The packing of the result is "standard": If A = fft(a, n), then A[0]
    contains the zero-frequency term, A[1:n/2+1] contains the
    positive-frequency terms, and A[n/2+1:] contains the negative-frequency
    terms, in order of decreasingly negative frequency. So for an 8-point
    transform, the frequencies of the result are [ 0, 1, 2, 3, 4, -3, -2, -1].

    This is most efficient for n a power of two. This also stores a cache of
    working memory for different sizes of fft's, so you could theoretically
    run into memory problems if you call this too many times with too many
    different n's.

    """

    return _raw_fft(a, n, axis, fftpack.cffti, fftpack.cfftf, _fft_cache)


def ifft(a, n=None, axis=-1):
    """
    Compute the one-dimensonal inverse fft along an axis.

    Return the `n` point inverse discrete Fourier transform of `a`.  The length
    `n` defaults to the length of `a`. If `n` is larger than the length of `a`,
    then `a` will be zero-padded to make up the difference. If `n` is smaller
    than the length of `a`, then `a` will be truncated to reduce its size.

    Parameters
    ----------

    a : array_like
        Input array.
    n : int, optional
        Length of the fft.
    axis : int, optional
        Axis over which to compute the inverse fft.

    See Also
    --------
    fft

    Notes
    -----
    The input array is expected to be packed the same way as the output of
    fft, as discussed in the fft documentation.

    This is the inverse of fft: ifft(fft(a)) == a within numerical
    accuracy.

    This is most efficient for `n` a power of two. This also stores a cache of
    working memory for different sizes of fft's, so you could theoretically
    run into memory problems if you call this too many times with too many
    different `n` values.

    """

    a = asarray(a).astype(complex)
    if n is None:
        n = shape(a)[axis]
    return _raw_fft(a, n, axis, fftpack.cffti, fftpack.cfftb, _fft_cache) / n


def rfft(a, n=None, axis=-1):
    """
    Compute the one-dimensional fft for real input.

    Return the n point discrete Fourier transform of the real valued
    array a. n defaults to the length of a. n is the length of the
    input, not the output.

    Parameters
    ----------

    a : array
        input array with real data type
    n : int
        length of the fft
    axis : int
        axis over which to compute the fft

    Notes
    -----

    The returned array will be the nonnegative frequency terms of the
    Hermite-symmetric, complex transform of the real array. So for an 8-point
    transform, the frequencies in the result are [ 0, 1, 2, 3, 4]. The first
    term will be real, as will the last if n is even. The negative frequency
    terms are not needed because they are the complex conjugates of the
    positive frequency terms. (This is what I mean when I say
    Hermite-symmetric.)

    This is most efficient for n a power of two.

    """

    a = asarray(a).astype(float)
    return _raw_fft(a, n, axis, fftpack.rffti, fftpack.rfftf, _real_fft_cache)


def irfft(a, n=None, axis=-1):
    """
    Compute the one-dimensional inverse fft for real input.

    Parameters
    ----------
    a : array
        Input array with real data type.
    n : int
        Length of the fft.
    axis : int
        Axis over which to compute the fft.

    See Also
    --------
    rfft

    Notes
    -----
    Return the real valued `n` point inverse discrete Fourier transform
    of `a`, where `a` contains the nonnegative frequency terms of a
    Hermite-symmetric sequence. `n` is the length of the result, not the
    input. If `n` is not supplied, the default is 2*(len(`a`)-1). If you
    want the length of the result to be odd, you have to say so.

    If you specify an `n` such that `a` must be zero-padded or truncated, the
    extra/removed values will be added/removed at high frequencies. One can
    thus resample a series to m points via Fourier interpolation by: a_resamp
    = irfft(rfft(a), m).

    Within numerical accuracy ``irfft`` is the inverse of ``rfft``::

     irfft(rfft(a), len(a)) == a

    """

    a = asarray(a).astype(complex)
    if n is None:
        n = (shape(a)[axis] - 1) * 2
    return _raw_fft(a, n, axis, fftpack.rffti, fftpack.rfftb,
                    _real_fft_cache) / n


def hfft(a, n=None, axis=-1):
    """
    Compute the fft of a signal which spectrum has Hermitian symmetry.

    Parameters
    ----------
    a : array
        input array
    n : int
        length of the hfft
    axis : int
        axis over which to compute the hfft

    See also
    --------
    rfft
    ihfft

    Notes
    -----
    These are a pair analogous to rfft/irfft, but for the
    opposite case: here the signal is real in the frequency domain and has
    Hermite symmetry in the time domain. So here it's hermite_fft for which
    you must supply the length of the result if it is to be odd.

    ihfft(hfft(a), len(a)) == a
    within numerical accuracy.

    """

    a = asarray(a).astype(complex)
    if n is None:
        n = (shape(a)[axis] - 1) * 2
    return irfft(conjugate(a), n, axis) * n


def ihfft(a, n=None, axis=-1):
    """
    Compute the inverse fft of a signal whose spectrum has Hermitian symmetry.

    Parameters
    ----------
    a : array_like
        Input array.
    n : int, optional
        Length of the ihfft.
    axis : int, optional
        Axis over which to compute the ihfft.

    See also
    --------
    rfft, hfft

    Notes
    -----
    These are a pair analogous to rfft/irfft, but for the
    opposite case: here the signal is real in the frequency domain and has
    Hermite symmetry in the time domain. So here it's hermite_fft for which
    you must supply the length of the result if it is to be odd.

    ihfft(hfft(a), len(a)) == a
    within numerical accuracy.

    """

    a = asarray(a).astype(float)
    if n is None:
        n = shape(a)[axis]
    return conjugate(rfft(a, n, axis))/n


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = range(-len(s), 0)
    if len(s) != len(axes):
        raise ValueError, "Shape and axes have different lengths."
    if invreal and shapeless:
        s[axes[-1]] = (s[axes[-1]] - 1) * 2
    return s, axes


def _raw_fftnd(a, s=None, axes=None, function=fft):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    itl = range(len(axes))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii])
    return a


def fftn(a, s=None, axes=None):
    """
    Compute the N-dimensional Fast Fourier Transform.

    Parameters
    ----------
    a : array_like
        Input array.
    s : sequence of ints
        Shape of each axis of the input (s[0] refers to axis 0, s[1] to
        axis 1, etc.).  This corresponds to `n` for `fft(x, n)`.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
    axes : tuple of int
        Axes over which to compute the FFT.

    Notes
    -----
    Analogously to `fft`, the term for zero frequency in all axes is in the
    low-order corner, while the term for the Nyquist frequency in all axes is
    in the middle.

    If neither `s` nor `axes` is specified, the transform is taken along all
    axes. If `s` is specified and `axes` is not, the last ``len(s)`` axes are
    used.  If `axes` is specified and `s` is not, the input shape along the
    specified axes is used. If `s` and `axes` are both specified and are not
    the same length, an exception is raised.

    """

    return _raw_fftnd(a,s,axes,fft)

def ifftn(a, s=None, axes=None):
    """
    Compute the inverse of fftn.

    Parameters
    ----------
    a : array
        input array
    s : sequence (int)
        shape of the ifft
    axis : int
        axis over which to compute the ifft

    Notes
    -----
    The n-dimensional ifft of a. s is a sequence giving the shape of the input
    an result along the transformed axes, as n for fft. Results are packed
    analogously to fft: the term for zero frequency in all axes is in the
    low-order corner, while the term for the Nyquist frequency in all axes is
    in the middle.

    If neither s nor axes is specified, the transform is taken along all
    axes. If s is specified and axes is not, the last len(s) axes are used.
    If axes are specified and s is not, the input shape along the specified
    axes is used. If s and axes are both specified and are not the same
    length, an exception is raised.

    """

    return _raw_fftnd(a, s, axes, ifft)


def fft2(a, s=None, axes=(-2,-1)):
    """
    Compute the 2-D FFT of an array.

    Parameters
    ----------
    a : array_like
        Input array.  The rank (dimensions) of `a` must be 2 or greater.
    s : shape tuple
        Shape of the FFT.
    axes : sequence of 2 ints
        The 2 axes over which to compute the FFT.  The default is the last two
        axes (-2, -1).

    Notes
    -----
    This is really just ``fftn`` with different default behavior.

    """

    return _raw_fftnd(a,s,axes,fft)


def ifft2(a, s=None, axes=(-2,-1)):
    """
    Compute the inverse 2d fft of an array.

    Parameters
    ----------
    a : array
        input array
    s : sequence (int)
        shape of the ifft
    axis : int
        axis over which to compute the ifft

    Notes
    -----
    This is really just ifftn with different default behavior.

    """

    return _raw_fftnd(a, s, axes, ifft)


def rfftn(a, s=None, axes=None):
    """
    Compute the n-dimensional fft of a real array.

    Parameters
    ----------
    a : array (real)
        input array
    s : sequence (int)
        shape of the fft
    axis : int
        axis over which to compute the fft

    Notes
    -----
    A real transform as rfft is performed along the axis specified by the last
    element of axes, then complex transforms as fft are performed along the
    other axes.

    """

    a = asarray(a).astype(float)
    s, axes = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1])
    for ii in range(len(axes)-1):
        a = fft(a, s[ii], axes[ii])
    return a

def rfft2(a, s=None, axes=(-2,-1)):
    """
    Compute the 2-dimensional fft of a real array.

    Parameters
    ----------
    a : array (real)
        input array
    s : sequence (int)
        shape of the fft
    axis : int
        axis over which to compute the fft

    Notes
    -----
    The 2-D fft of the real valued array a. This is really just rfftn with
    different default behavior.

    """

    return rfftn(a, s, axes)

def irfftn(a, s=None, axes=None):
    """
    Compute the n-dimensional inverse fft of a real array.

    Parameters
    ----------
    a : array (real)
        input array
    s : sequence (int)
        shape of the inverse fft
    axis : int
        axis over which to compute the inverse fft

    Notes
    -----
    The transform implemented in ifftn is applied along
    all axes but the last, then the transform implemented in irfft is performed
    along the last axis. As with irfft, the length of the result along that
    axis must be specified if it is to be odd.

    """

    a = asarray(a).astype(complex)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes)-1):
        a = ifft(a, s[ii], axes[ii])
    a = irfft(a, s[-1], axes[-1])
    return a

def irfft2(a, s=None, axes=(-2,-1)):
    """
    Compute the 2-dimensional inverse fft of a real array.

    Parameters
    ----------
    a : array (real)
        input array
    s : sequence (int)
        shape of the inverse fft
    axis : int
        axis over which to compute the inverse fft

    Notes
    -----
    This is really irfftn with different default.

    """

    return irfftn(a, s, axes)

# Deprecated names
from numpy import deprecate
refft = deprecate(rfft, 'refft', 'rfft')
irefft = deprecate(irfft, 'irefft', 'irfft')
refft2 = deprecate(rfft2, 'refft2', 'rfft2')
irefft2 = deprecate(irfft2, 'irefft2', 'irfft2')
refftn = deprecate(rfftn, 'refftn', 'rfftn')
irefftn = deprecate(irfftn, 'irefftn', 'irfftn')
