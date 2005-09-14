"""
Discrete Fourier Transforms - FFT.py 

The underlying code for these functions is an f2c translated and modified
version of the FFTPACK routines.

fft(a, n=None, axis=-1) 
inverse_fft(a, n=None, axis=-1) 
real_fft(a, n=None, axis=-1) 
inverse_real_fft(a, n=None, axis=-1)
hermite_fft(a, n=None, axis=-1)
inverse_hermite_fft(a, n=None, axis=-1)
fftnd(a, s=None, axes=None)
inverse_fftnd(a, s=None, axes=None)
real_fftnd(a, s=None, axes=None)
inverse_real_fftnd(a, s=None, axes=None)
fft2d(a, s=None, axes=(-2,-1)) 
inverse_fft2d(a, s=None, axes=(-2, -1))
real_fft2d(a, s=None, axes=(-2,-1)) 
inverse_real_fft2d(a, s=None, axes=(-2, -1))
"""
import Numeric, fftpack, copy

_fft_cache = {}
_real_fft_cache = {}

def _raw_fft(a, n=None, axis=-1, init_function=fftpack.cffti, 
             work_function=fftpack.cfftf, fft_cache = _fft_cache ):
    a = Numeric.asarray(a)

    if n == None: n = a.shape[axis]

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
            z = Numeric.zeros(s, a.typecode())
            z[index] = a
            a = z

    if axis != -1:
        a = Numeric.swapaxes(a, axis, -1)
    r = work_function(a, wsave)
    if axis != -1:
        r = Numeric.swapaxes(r, axis, -1)
    return r


def fft(a, n=None, axis=-1):
    """fft(a, n=None, axis=-1) 

    Will return the n point discrete Fourier transform of a. n defaults to the
    length of a. If n is larger than a, then a will be zero-padded to make up
    the difference. If n is smaller than a, the first n items in a will be
    used.

    The packing of the result is "standard": If A = fft(a, n), then A[0]
    contains the zero-frequency term, A[1:n/2+1] contains the
    positive-frequency terms, and A[n/2+1:] contains the negative-frequency
    terms, in order of decreasingly negative frequency. So for an 8-point
    transform, the frequencies of the result are [ 0, 1, 2, 3, 4, -3, -2, -1].

    This is most efficient for n a power of two. This also stores a cache of
    working memory for different sizes of fft's, so you could theoretically
    run into memory problems if you call this too many times with too many
    different n's."""

    return _raw_fft(a, n, axis, fftpack.cffti, fftpack.cfftf, _fft_cache)


def inverse_fft(a, n=None, axis=-1):
    """inverse_fft(a, n=None, axis=-1) 

    Will return the n point inverse discrete Fourier transform of a.  n
    defaults to the length of a. If n is larger than a, then a will be
    zero-padded to make up the difference. If n is smaller than a, then a will
    be truncated to reduce its size.

    The input array is expected to be packed the same way as the output of
    fft, as discussed in it's documentation.

    This is the inverse of fft: inverse_fft(fft(a)) == a within numerical
    accuracy.

    This is most efficient for n a power of two. This also stores a cache of
    working memory for different sizes of fft's, so you could theoretically
    run into memory problems if you call this too many times with too many
    different n's."""

    a = Numeric.asarray(a).astype(Numeric.Complex)
    if n == None:
        n = Numeric.shape(a)[axis]
    return _raw_fft(a, n, axis, fftpack.cffti, fftpack.cfftb, _fft_cache) / n


def real_fft(a, n=None, axis=-1):
    """real_fft(a, n=None, axis=-1) 

    Will return the n point discrete Fourier transform of the real valued
    array a. n defaults to the length of a. n is the length of the input, not
    the output.

    The returned array will be the nonnegative frequency terms of the
    Hermite-symmetric, complex transform of the real array. So for an 8-point
    transform, the frequencies in the result are [ 0, 1, 2, 3, 4]. The first
    term will be real, as will the last if n is even. The negative frequency
    terms are not needed because they are the complex conjugates of the
    positive frequency terms. (This is what I mean when I say
    Hermite-symmetric.)

    This is most efficient for n a power of two."""

    a = Numeric.asarray(a).astype(Numeric.Float)
    return _raw_fft(a, n, axis, fftpack.rffti, fftpack.rfftf, _real_fft_cache)


def inverse_real_fft(a, n=None, axis=-1):
    """inverse_real_fft(a, n=None, axis=-1)
    
    Will return the real valued n point inverse discrete Fourier transform of
    a, where a contains the nonnegative frequency terms of a Hermite-symmetric
    sequence. n is the length of the result, not the input. If n is not
    supplied, the default is 2*(len(a)-1). If you want the length of the
    result to be odd, you have to say so.

    If you specify an n such that a must be zero-padded or truncated, the
    extra/removed values will be added/removed at high frequencies. One can
    thus resample a series to m points via Fourier interpolation by: a_resamp
    = inverse_real_fft(real_fft(a), m).

    This is the inverse of real_fft:
    inverse_real_fft(real_fft(a), len(a)) == a
    within numerical accuracy."""

    a = Numeric.asarray(a).astype(Numeric.Complex)
    if n == None:
        n = (Numeric.shape(a)[axis] - 1) * 2
    return _raw_fft(a, n, axis, fftpack.rffti, fftpack.rfftb,
                    _real_fft_cache) / n


def hermite_fft(a, n=None, axis=-1):
    """hermite_fft(a, n=None, axis=-1)
    inverse_hermite_fft(a, n=None, axis=-1)

    These are a pair analogous to real_fft/inverse_real_fft, but for the
    opposite case: here the signal is real in the frequency domain and has
    Hermite symmetry in the time domain. So here it's hermite_fft for which
    you must supply the length of the result if it is to be odd.

    inverse_hermite_fft(hermite_fft(a), len(a)) == a
    within numerical accuracy."""

    a = Numeric.asarray(a).astype(Numeric.Complex)
    if n == None:
        n = (Numeric.shape(a)[axis] - 1) * 2
    return inverse_real_fft(Numeric.conjugate(a), n, axis) * n


def inverse_hermite_fft(a, n=None, axis=-1):
    """hermite_fft(a, n=None, axis=-1)
    inverse_hermite_fft(a, n=None, axis=-1)

    These are a pair analogous to real_fft/inverse_real_fft, but for the
    opposite case: here the signal is real in the frequency domain and has
    Hermite symmetry in the time domain. So here it's hermite_fft for which
    you must supply the length of the result if it is to be odd.

    inverse_hermite_fft(hermite_fft(a), len(a)) == a
    within numerical accuracy."""
    
    a = Numeric.asarray(a).astype(Numeric.Float)
    if n == None:
        n = Numeric.shape(a)[axis]
    return Numeric.conjugate(real_fft(a, n, axis))/n


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes == None:
            s = list(a.shape)
        else:
            s = Numeric.take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes == None:
        axes = range(-len(s), 0)
    if len(s) != len(axes):
        raise ValueError, "Shape and axes have different lengths."
    if invreal and shapeless:
        s[axes[-1]] = (s[axes[-1]] - 1) * 2
    return s, axes


def _raw_fftnd(a, s=None, axes=None, function=fft):
    a = Numeric.asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    itl = range(len(axes))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii])
    return a


def fftnd(a, s=None, axes=None):
    """fftnd(a, s=None, axes=None)

    The n-dimensional fft of a. s is a sequence giving the shape of the input
    an result along the transformed axes, as n for fft. Results are packed
    analogously to fft: the term for zero frequency in all axes is in the
    low-order corner, while the term for the Nyquist frequency in all axes is
    in the middle.

    If neither s nor axes is specified, the transform is taken along all
    axes. If s is specified and axes is not, the last len(s) axes are used.
    If axes are specified and s is not, the input shape along the specified
    axes is used. If s and axes are both specified and are not the same
    length, an exception is raised."""

    return _raw_fftnd(a,s,axes,fft)


def inverse_fftnd(a, s=None, axes=None):
    """inverse_fftnd(a, s=None, axes=None)
    
    The inverse of fftnd."""
    
    return _raw_fftnd(a, s, axes, inverse_fft)


def fft2d(a, s=None, axes=(-2,-1)):
    """fft2d(a, s=None, axes=(-2,-1)) 
    
    The 2d fft of a. This is really just fftnd with different default
    behavior."""

    return _raw_fftnd(a,s,axes,fft)


def inverse_fft2d(a, s=None, axes=(-2,-1)):
    """inverse_fft2d(a, s=None, axes=(-2, -1))

    The inverse of fft2d. This is really just inverse_fftnd with different
    default behavior."""

    return _raw_fftnd(a, s, axes, inverse_fft)


def real_fftnd(a, s=None, axes=None):
    """real_fftnd(a, s=None, axes=None)

    The n-dimensional discrete Fourier transform of a real array a. A real
    transform as real_fft is performed along the axis specified by the last
    element of axes, then complex transforms as fft are performed along the
    other axes."""
    
    a = Numeric.asarray(a).astype(Numeric.Float)
    s, axes = _cook_nd_args(a, s, axes)
    a = real_fft(a, s[-1], axes[-1])
    for ii in range(len(axes)-1):
        a = fft(a, s[ii], axes[ii])
    return a


def real_fft2d(a, s=None, axes=(-2,-1)):
    """real_fft2d(a, s=None, axes=(-2,-1)) 

    The 2d fft of the real valued array a. This is really just real_fftnd with
    different default behavior."""
    
    return real_fftnd(a, s, axes)


def inverse_real_fftnd(a, s=None, axes=None):
    """inverse_real_fftnd(a, s=None, axes=None)

    The inverse of real_fftnd. The transform implemented in inverse_fft is
    applied along all axes but the last, then the transform implemented in
    inverse_real_fft is performed along the last axis. As with
    inverse_real_fft, the length of the result along that axis must be
    specified if it is to be odd."""
    
    a = Numeric.asarray(a).astype(Numeric.Complex)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes)-1):
        a = inverse_fft(a, s[ii], axes[ii])
    a = inverse_real_fft(a, s[-1], axes[-1])
    return a


def inverse_real_fft2d(a, s=None, axes=(-2,-1)):
    """inverse_real_fft2d(a, s=None, axes=(-2, -1))

    The inverse of real_fft2d. This is really just inverse_real_fftnd with
    different default behavior."""
    
    return inverse_real_fftnd(a, s, axes)


