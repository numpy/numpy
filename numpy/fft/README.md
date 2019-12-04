PocketFFT
---------

This is a heavily modified implementation of FFTPack [1,2], with the following
advantages:

- strictly C99 compliant
- more accurate twiddle factor computation
- very fast plan generation
- worst case complexity for transform sizes with large prime factors is
  `N*log(N)`, because Bluestein's algorithm [3] is used for these cases.


Some code details
-----------------

Twiddle factor computation:

- making use of symmetries to reduce number of sin/cos evaluations
- all angles are reduced to the range `[0; pi/4]` for higher accuracy
- an adapted implementation of `sincospi()` is used, which actually computes
  `sin(x)` and `(cos(x)-1)`.
- if `n` sin/cos pairs are required, the adjusted `sincospi()` is only called
  `2*sqrt(n)` times; the remaining values are obtained by evaluating the
  angle addition theorems in a numerically accurate way.

Parallel invocation:

- Plans only contain read-only data; all temporary arrays are allocated and
  deallocated during an individual FFT execution. This means that a single plan
  can be used in several threads at the same time.

Efficient codelets are available for the factors:

- 2, 3, 4, 5, 7, 11 for complex-valued FFTs
- 2, 3, 4, 5 for real-valued FFTs

Larger prime factors are handled by somewhat less efficient, generic routines.

For lengths with very large prime factors, Bluestein's algorithm is used, and
instead of an FFT of length `n`, a convolution of length `n2 >= 2*n-1`
is performed, where `n2` is chosen to be highly composite.


[1] Swarztrauber, P. 1982, Vectorizing the Fast Fourier Transforms
    (New York: Academic Press), 51
[2] https://www.netlib.org/fftpack/
[3] https://en.wikipedia.org/wiki/Chirp_Z-transform
