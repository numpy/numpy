"""
Functions to operate on polynomials.
"""

__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
           'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
           'polyfit', 'RankWarning']

import re
import warnings
import numpy.core.numeric as NX

from numpy.core import isscalar, abs
from numpy.lib.getlimits import finfo
from numpy.lib.twodim_base import diag, vander
from numpy.lib.shape_base import hstack, atleast_1d
from numpy.lib.function_base import trim_zeros, sort_complex
from numpy.linalg import eigvals, lstsq

class RankWarning(UserWarning):
    """Issued by polyfit when Vandermonde matrix is rank deficient.
    """
    pass

def poly(seq_of_zeros):
    """
    Return polynomial coefficients given a sequence of roots.

    Calculate the coefficients of a polynomial given the zeros
    of the polynomial.

    If a square matrix is given, then the coefficients for
    characteristic equation of the matrix, defined by
    :math:`\\mathrm{det}(\\mathbf{A} - \\lambda \\mathbf{I})`,
    are returned.

    Parameters
    ----------
    seq_of_zeros : ndarray
        A sequence of polynomial roots or a square matrix.

    Returns
    -------
    coefs : ndarray
        A sequence of polynomial coefficients representing the polynomial

        :math:`\\mathrm{coefs}[0] x^{n-1} + \\mathrm{coefs}[1] x^{n-2} +
                      ... + \\mathrm{coefs}[2] x + \\mathrm{coefs}[n]`

    See Also
    --------
    numpy.poly1d : A one-dimensional polynomial class.
    numpy.roots : Return the roots of the polynomial coefficients in p
    numpy.polyfit : Least squares polynomial fit

    Examples
    --------
    Given a sequence of polynomial zeros,

    >>> b = np.roots([1, 3, 1, 5, 6])
    >>> np.poly(b)
    array([ 1.,  3.,  1.,  5.,  6.])

    Given a square matrix,

    >>> P = np.array([[19, 3], [-2, 26]])
    >>> np.poly(P)
    array([   1.,  -45.,  500.])

    """
    seq_of_zeros = atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape
    if len(sh) == 2 and sh[0] == sh[1]:
        seq_of_zeros = eigvals(seq_of_zeros)
    elif len(sh) ==1:
        pass
    else:
        raise ValueError, "input must be 1d or square 2d array."

    if len(seq_of_zeros) == 0:
        return 1.0

    a = [1]
    for k in range(len(seq_of_zeros)):
        a = NX.convolve(a, [1, -seq_of_zeros[k]], mode='full')

    if issubclass(a.dtype.type, NX.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = NX.asarray(seq_of_zeros, complex)
        pos_roots = sort_complex(NX.compress(roots.imag > 0, roots))
        neg_roots = NX.conjugate(sort_complex(
                                        NX.compress(roots.imag < 0,roots)))
        if (len(pos_roots) == len(neg_roots) and
            NX.alltrue(neg_roots == pos_roots)):
            a = a.real.copy()

    return a

def roots(p):
    """
    Return the roots of a polynomial with coefficients given in p.

    The values in the rank-1 array `p` are coefficients of a polynomial.
    If the length of `p` is n+1 then the polynomial is described by
    p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    Parameters
    ----------
    p : array_like of shape(M,)
        Rank-1 array of polynomial co-efficients.

    Returns
    -------
    out : ndarray
        An array containing the complex roots of the polynomial.

    Raises
    ------
    ValueError:
        When `p` cannot be converted to a rank-1 array.

    Examples
    --------

    >>> coeff = [3.2, 2, 1]
    >>> print np.roots(coeff)
    [-0.3125+0.46351241j -0.3125-0.46351241j]

    """
    # If input is scalar, this makes it an array
    p = atleast_1d(p)
    if len(p.shape) != 1:
        raise ValueError,"Input must be a rank-1 array."

    # find non-zero array entries
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return NX.array([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = diag(NX.ones((N-2,), p.dtype), -1)
        A[0, :] = -p[1:] / p[0]
        roots = eigvals(A)
    else:
        roots = NX.array([])

    # tack any zeros onto the back of the array
    roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots

def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.

    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part

    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}

    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.

    Parameters
    ----------
    p : {array_like, poly1d}
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : {None, list of `m` scalars, scalar}, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.

        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.

    See Also
    --------
    polyder : derivative of a polynomial
    poly1d.integ : equivalent method

    Examples
    --------
    The defining property of the antiderivative:

    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ])
    >>> np.polyder(P) == p
    True

    The integration constants default to zero, but can be specified:

    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6,5,3])
    >>> P
    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3.,  5.,  3. ])

    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:

    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0

    """
    m = int(m)
    if m < 0:
        raise ValueError, "Order of integral must be positive (see polyder)"
    if k is None:
        k = NX.zeros(m, float)
    k = atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0]*NX.ones(m, float)
    if len(k) < m:
        raise ValueError, \
              "k must be a scalar or a rank-1 array of length 1 or >m."

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p) + 0.0
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        y = NX.zeros(len(p) + 1, p.dtype)
        y[:-1] = p*1.0/NX.arange(len(p), 0, -1)
        y[-1] = k[0]
        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return poly1d(val)
        return val

def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.

    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([ 0.])

    """
    m = int(m)
    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    n = len(p)-1
    y = p[:-1] * NX.arange(n, 0, -1)
    if m < 0:
        raise ValueError, "Order of derivative must be positive (see polyint)"
    if m == 0:
        return p
    else:
        val = polyder(y, m-1)
        if truepoly:
            val = poly1d(val)
        return val

def polyfit(x, y, deg, rcond=None, full=False):
    """
    Least squares polynomial fit.

    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than this
        relative to the largest singular value will be ignored. The default
        value is len(x)*eps, where eps is the relative precision of the float
        type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is
        False (the default) just the coefficients are returned, when True
        diagnostic information from the singular value decomposition is also
        returned.

    Returns
    -------
    p : ndarray, shape (M,) or (M, K)
        Polynomial coefficients, highest power first.
        If `y` was 2-D, the coefficients for `k`-th data set are in ``p[:,k]``.

    residuals, rank, singular_values, rcond : present only if `full` = True
        Residuals of the least-squares fit, the effective rank of the scaled
        Vandermonde coefficient matrix, its singular values, and the specified
        value of `rcond`. For more details, see `linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.

        The warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)

    See Also
    --------
    polyval : Computes polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution minimizes the squared error

    .. math ::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2

    in the equations::

        x[0]**n * p[n] + ... + x[0] * p[1] + p[0] = y[0]
        x[1]**n * p[n] + ... + x[1] * p[1] + p[0] = y[1]
        ...
        x[k]**n * p[n] + ... + x[k] * p[1] + p[0] = y[k]

    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.

    `polyfit` issues a `RankWarning` when the least-squares fit is badly
    conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.

    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           http://en.wikipedia.org/wiki/Polynomial_interpolation

    Examples
    --------
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])

    It is convenient to use `poly1d` objects for dealing with polynomials:

    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179
    >>> p(3.5)
    -0.34732142857143039
    >>> p(10)
    22.579365079365115

    High-order polynomials may oscillate wildly:

    >>> p30 = np.poly1d(np.polyfit(x, y, 30))
    /... RankWarning: Polyfit may be poorly conditioned...
    >>> p30(4)
    -0.80000000000000204
    >>> p30(5)
    -0.99999999999999445
    >>> p30(4.5)
    -0.10547061179440398

    Illustration:

    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    >>> plt.ylim(-2,2)
    >>> plt.show()

    """
    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError, "expected deg >= 0"
    if x.ndim != 1:
        raise TypeError, "expected 1D vector for x"
    if x.size == 0:
        raise TypeError, "expected non-empty vector for x"
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError, "expected 1D or 2D array for y"
    if x.shape[0] != y.shape[0] :
        raise TypeError, "expected x and y to have same length"

    # set rcond
    if rcond is None :
        rcond = len(x)*finfo(x.dtype).eps

    # scale x to improve condition number
    scale = abs(x).max()
    if scale != 0 :
        x /= scale

    # solve least squares equation for powers of x
    v = vander(x, order)
    c, resids, rank, s = lstsq(v, y, rcond)

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    # scale returned coefficients
    if scale != 0 :
        if c.ndim == 1 :
            c /= vander([scale], order)[0]
        else :
            c /= vander([scale], order).T

    if full :
        return c, resids, rank, s, rcond
    else :
        return c



def polyval(p, x):
    """
    Evaluate a polynomial at specific values.

    If p is of length N, this function returns the value:

        p[0]*(x**N-1) + p[1]*(x**N-2) + ... + p[N-2]*x + p[N-1]

    If x is a sequence then p(x) will be returned for all elements of x.
    If x is another polynomial then the composite polynomial p(x) will
    be returned.

    Parameters
    ----------
    p : {array_like, poly1d}
       1D array of polynomial coefficients from highest degree to zero or an
       instance of poly1d.
    x : {array_like, poly1d}
       A number, a 1D array of numbers, or an instance of poly1d.

    Returns
    -------
    values : {ndarray, poly1d}
       If either p or x is an instance of poly1d, then an instance of poly1d
       is returned, otherwise a 1D array is returned. In the case where x is
       a poly1d, the result is the composition of the two polynomials, i.e.,
       substitution is used.

    See Also
    --------
    poly1d: A polynomial class.

    Notes
    -----
    Horner's method is used to evaluate the polynomial. Even so, for
    polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.


    Examples
    --------
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76

    """
    p = NX.asarray(p)
    if isinstance(x, poly1d):
        y = 0
    else:
        x = NX.asarray(x)
        y = NX.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[i]
    return y

def polyadd(a1, a2):
    """
    Returns sum of two polynomials.

    Returns sum of polynomials; `a1` + `a2`.  Input polynomials are
    represented as an array_like sequence of terms or a poly1d object.

    Parameters
    ----------
    a1 : {array_like, poly1d}
        Polynomial as sequence of terms.
    a2 : {array_like, poly1d}
        Polynomial as sequence of terms.

    Returns
    -------
    out : {ndarray, poly1d}
        Array representing the polynomial terms.

    See Also
    --------
    polyval, polydiv, polymul, polyadd

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 + a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) + a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 + NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val

def polysub(a1, a2):
    """
    Returns difference from subtraction of two polynomials input as sequences.

    Returns difference of polynomials; `a1` - `a2`.  Input polynomials are
    represented as an array_like sequence of terms or a poly1d object.

    Parameters
    ----------
    a1 : {array_like, poly1d}
        Minuend polynomial as sequence of terms.
    a2 : {array_like, poly1d}
        Subtrahend polynomial as sequence of terms.

    Returns
    -------
    out : {ndarray, poly1d}
        Array representing the polynomial terms.

    See Also
    --------
    polyval, polydiv, polymul, polyadd

    Examples
    --------
    .. math:: (2 x^2 + 10 x - 2) - (3 x^2 + 10 x -4) = (-x^2 + 2)

    >>> np.polysub([2, 10, -2], [3, 10, -4])
    array([-1,  0,  2])

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 - a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) - a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 - NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val


def polymul(a1, a2):
    """
    Returns product of two polynomials represented as sequences.

    The input arrays specify the polynomial terms in turn with a length equal
    to the polynomial degree plus 1.

    Parameters
    ----------
    a1 : {array_like, poly1d}
        First multiplier polynomial.
    a2 : {array_like, poly1d}
        Second multiplier polynomial.

    Returns
    -------
    out : {ndarray, poly1d}
        Product of inputs.

    See Also
    --------
    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub,
    polyval

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1,a2 = poly1d(a1),poly1d(a2)
    val = NX.convolve(a1, a2)
    if truepoly:
        val = poly1d(val)
    return val

def polydiv(u, v):
    """
    Returns the quotient and remainder of polynomial division.

    The input arrays specify the polynomial terms in turn with a length equal
    to the polynomial degree plus 1.

    Parameters
    ----------
    u : {array_like, poly1d}
        Dividend polynomial.
    v : {array_like, poly1d}
        Divisor polynomial.

    Returns
    -------
    q : ndarray
        Polynomial terms of quotient.
    r : ndarray
        Remainder of polynomial division.

    See Also
    --------
    poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub,
    polyval

    Examples
    --------
    .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25

    >>> x = np.array([3.0, 5.0, 2.0])
    >>> y = np.array([2.0, 1.0])
    >>> np.polydiv(x, y)
    >>> (array([ 1.5 ,  1.75]), array([ 0.25]))

    """
    truepoly = (isinstance(u, poly1d) or isinstance(u, poly1d))
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.copy()
    for k in range(0, m-n+1):
        d = scale * r[k]
        q[k] = d
        r[k:k+n+1] -= d*v
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r

_poly_mat = re.compile(r"[*][*]([0-9]*)")
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while 1:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' '*(len(power)-1)
        toadd1 = ' '*(len(partstr)-1) + power
        if ((len(line2)+len(toadd2) > wrap) or \
            (len(line1)+len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' '*(len(power)-1)
            line1 += ' '*(len(partstr)-1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]


class poly1d(object):
    """
    A one-dimensional polynomial class.

    Parameters
    ----------
    c_or_r : array_like
        Polynomial coefficients, in decreasing powers.  For example,
        ``(1, 2, 3)`` implies :math:`x^2 + 2x + 3`.  If `r` is set
        to True, these coefficients specify the polynomial roots
        (values where the polynomial evaluate to 0) instead.
    r : bool, optional
        If True, `c_or_r` gives the polynomial roots.  Default is False.

    Examples
    --------
    Construct the polynomial :math:`x^2 + 2x + 3`:

    >>> p = np.poly1d([1, 2, 3])
    >>> print np.poly1d(p)
       2
    1 x + 2 x + 3

    Evaluate the polynomial:

    >>> p(0.5)
    4.25

    Find the roots:

    >>> p.r
    array([-1.+1.41421356j, -1.-1.41421356j])

    Show the coefficients:

    >>> p.c
    array([1, 2, 3])

    Display the order (the leading zero-coefficients are removed):

    >>> p.order
    2

    Show the coefficient of the k-th power in the polynomial
    (which is equivalent to ``p.c[-(i+1)]``):

    >>> p[1]
    2

    Polynomials can be added, substracted, multplied and divided
    (returns quotient and remainder):

    >>> p * p
    poly1d([ 1,  4, 10, 12,  9])

    >>> (p**3 + 4) / p
    (poly1d([  1.,   4.,  10.,  12.,   9.]), poly1d([4]))

    ``asarray(p)`` gives the coefficient array, so polynomials can be
    used in all functions that accept arrays:

    >>> p**2 # square of polynomial
    poly1d([ 1,  4, 10, 12,  9])

    >>> np.square(p) # square of individual coefficients
    array([1, 4, 9])

    The variable used in the string representation of `p` can be modified,
    using the `variable` parameter:

    >>> p = np.poly1d([1,2,3], variable='z')
    >>> print p
       2
    1 z + 2 z + 3

    Construct a polynomial from its roots:

    >>> np.poly1d([1, 2], True)
    poly1d([ 1, -3,  2])

    This is the same polynomial as obtained by:

    >>> np.poly1d([1, -1]) * np.poly1d([1, -2])
    poly1d([ 1, -3,  2])

    """
    coeffs = None
    order = None
    variable = None
    def __init__(self, c_or_r, r=0, variable=None):
        if isinstance(c_or_r, poly1d):
            for key in c_or_r.__dict__.keys():
                self.__dict__[key] = c_or_r.__dict__[key]
            if variable is not None:
                self.__dict__['variable'] = variable
            return
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = atleast_1d(c_or_r)
        if len(c_or_r.shape) > 1:
            raise ValueError, "Polynomial must be 1d only."
        c_or_r = trim_zeros(c_or_r, trim='f')
        if len(c_or_r) == 0:
            c_or_r = NX.array([0.])
        self.__dict__['coeffs'] = c_or_r
        self.__dict__['order'] = len(c_or_r) - 1
        if variable is None:
            variable = 'x'
        self.__dict__['variable'] = variable

    def __array__(self, t=None):
        if t:
            return NX.asarray(self.coeffs, t)
        else:
            return NX.asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        thestr = "0"
        var = self.variable

        # Remove leading zeros
        coeffs = self.coeffs[NX.logical_or.accumulate(self.coeffs != 0)]
        N = len(coeffs)-1

        for k in range(len(coeffs)):
            coefstr ='%.4g' % abs(coeffs[k])
            if coefstr[-4:] == '0000':
                coefstr = coefstr[:-5]
            power = (N-k)
            if power == 0:
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    if k == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = '%s %s' % (coefstr, var)
            else:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = '%s**%d' % (var, power,)
                else:
                    newstr = '%s %s**%d' % (coefstr, var, power)

            if k > 0:
                if newstr != '':
                    if coeffs[k] < 0:
                        thestr = "%s - %s" % (thestr, newstr)
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            elif (k == 0) and (newstr != '') and (coeffs[k] < 0):
                thestr = "-%s" % (newstr,)
            else:
                thestr = newstr
        return _raise_power(thestr)


    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError, "Power to non-negative integers only."
        res = [1]
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __div__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs/other)
        else:
            other = poly1d(other)
            return polydiv(self, other)

    def __rdiv__(self, other):
        if isscalar(other):
            return poly1d(other/self.coeffs)
        else:
            other = poly1d(other)
            return polydiv(other, self)

    def __eq__(self, other):
        return NX.alltrue(self.coeffs == other.coeffs)

    def __ne__(self, other):
        return NX.any(self.coeffs != other.coeffs)

    def __setattr__(self, key, val):
        raise ValueError, "Attributes cannot be changed this way."

    def __getattr__(self, key):
        if key in ['r', 'roots']:
            return roots(self.coeffs)
        elif key in ['c','coef','coefficients']:
            return self.coeffs
        elif key in ['o']:
            return self.order
        else:
            try:
                return self.__dict__[key]
            except KeyError:
                raise AttributeError("'%s' has no attribute '%s'" % (self.__class__, key))

    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError, "Does not support negative powers."
        if key > self.order:
            zr = NX.zeros(key-self.order, self.coeffs.dtype)
            self.__dict__['coeffs'] = NX.concatenate((zr, self.coeffs))
            self.__dict__['order'] = key
            ind = 0
        self.__dict__['coeffs'][ind] = val
        return

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        """
        Return an antiderivative (indefinite integral) of this polynomial.

        Refer to `polyint` for full documentation.

        See Also
        --------
        polyint : equivalent function

        """
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `polyder` for full documentation.

        See Also
        --------
        polyder : equivalent function

        """
        return poly1d(polyder(self.coeffs, m=m))

# Stuff to do on module import

warnings.simplefilter('always',RankWarning)
