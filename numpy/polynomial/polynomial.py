"""Functions for dealing with polynomials.

This module provides a number of functions that are useful in dealing with
polynomials as well as a ``Polynomial`` class that encapsuletes the usual
arithmetic operations. All arrays of polynomial coefficients are assumed to
be ordered from low to high degree, thus `array([1,2,3])` will be treated
as the polynomial ``1 + 2*x + 3*x**2``

Constants
---------
- polydomain -- Polynomial default domain
- polyzero -- Polynomial that evaluates to 0.
- polyone -- Polynomial that evaluates to 1.
- polyx -- Polynomial of the identity map (x).

Arithmetic
----------
- polyadd -- add a polynomial to another.
- polysub -- subtract a polynomial from another.
- polymul -- multiply a polynomial by another
- polydiv -- divide one polynomial by another.
- polyval -- evaluate a polynomial at given points.

Calculus
--------
- polyder -- differentiate a polynomial.
- polyint -- integrate a polynomial.

Misc Functions
--------------
- polyfromroots -- create a polynomial with specified roots.
- polyroots -- find the roots of a polynomial.
- polyvander -- Vandermode like matrix for powers.
- polyfit -- least squares fit returning a polynomial.
- polytrim -- trim leading coefficients from a polynomial.
- polyline -- Polynomial of given straight line

Classes
-------
- Polynomial -- polynomial class.

"""
from __future__ import division

__all__ = ['polyzero', 'polyone', 'polyx', 'polydomain',
        'polyline','polyadd', 'polysub', 'polymul', 'polydiv', 'polyval',
        'polyder', 'polyint', 'polyfromroots', 'polyvander', 'polyfit',
        'polytrim', 'polyroots', 'Polynomial']

import numpy as np
import numpy.linalg as la
import polyutils as pu
import warnings
from polytemplate import polytemplate

polytrim = pu.trimcoef

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Polynomial default domain.
polydomain = np.array([-1,1])

# Polynomial coefficients representing zero.
polyzero = np.array([0])

# Polynomial coefficients representing one.
polyone = np.array([1])

# Polynomial coefficients representing the identity x.
polyx = np.array([0,1])

#
# Polynomial series functions
#

def polyline(off, scl) :
    """Polynomial whose graph is a straight line.

    The line has the formula ``off + scl*x``

    Parameters:
    -----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns:
    --------
    series : 1d ndarray
        The polynomial equal to ``off + scl*x``.

    """
    if scl != 0 :
        return np.array([off,scl])
    else :
        return np.array([off])

def polyfromroots(roots) :
    """Generate a polynomial with given roots.

    Generate a polynomial whose roots are given by `roots`. The resulting
    series is the produet `(x - roots[0])*(x - roots[1])*...`

    Inputs
    ------
    roots : array_like
        1-d array containing the roots in sorted order.

    Returns
    -------
    series : ndarray
        1-d array containing the coefficients of the Chebeshev series
        ordered from low to high.

    See Also
    --------
    polyroots

    """
    if len(roots) == 0 :
        return np.ones(1)
    else :
        [roots] = pu.as_series([roots], trim=False)
        prd = np.zeros(len(roots) + 1, dtype=roots.dtype)
        prd[-1] = 1
        for i in range(len(roots)) :
            prd[-(i+2):-1] -= roots[i]*prd[-(i+1):]
        return prd


def polyadd(c1, c2):
    """Add one polynomial to another.

    Returns the sum of two polynomials `c1` + `c2`. The arguments are
    sequences of coefficients ordered from low to high, i.e., [1,2,3] is
    the polynomial ``1 + 2*x + 3*x**2"``.

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        polynomial of the sum.

    See Also
    --------
    polysub, polymul, polydiv, polypow

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if len(c1) > len(c2) :
        c1[:c2.size] += c2
        ret = c1
    else :
        c2[:c1.size] += c1
        ret = c2
    return pu.trimseq(ret)


def polysub(c1, c2):
    """Subtract one polynomial from another.

    Returns the difference of two polynomials `c1` - `c2`. The arguments
    are sequences of coefficients ordered from low to high, i.e., [1,2,3]
    is the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        polynomial of the difference.

    See Also
    --------
    polyadd, polymul, polydiv, polypow

    Examples
    --------

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if len(c1) > len(c2) :
        c1[:c2.size] -= c2
        ret = c1
    else :
        c2 = -c2
        c2[:c1.size] += c1
        ret = c2
    return pu.trimseq(ret)


def polymul(c1, c2):
    """Multiply one polynomial by another.

    Returns the product of two polynomials `c1` * `c2`. The arguments
    are sequences of coefficients ordered from low to high, i.e., [1,2,3]
    is the polynomial  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of polyyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        polynomial of the product.

    See Also
    --------
    polyadd, polysub, polydiv, polypow

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    ret = np.convolve(c1, c2)
    return pu.trimseq(ret)


def polydiv(c1, c2):
    """Divide one polynomial by another.

    Returns the quotient of two polynomials `c1` / `c2`. The arguments are
    sequences of coefficients ordered from low to high, i.e., [1,2,3] is
    the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarray
        polynomial of the quotient and remainder.

    See Also
    --------
    polyadd, polysub, polymul, polypow

    Examples
    --------

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0 :
        raise ZeroDivisionError()

    len1 = len(c1)
    len2 = len(c2)
    if len2 == 1 :
        return c1/c2[-1], c1[:1]*0
    elif len1 < len2 :
        return c1[:1]*0, c1
    else :
        dlen = len1 - len2
        scl = c2[-1]
        c2  = c2[:-1]/scl
        i = dlen
        j = len1 - 1
        while i >= 0 :
            c1[i:j] -= c2*c1[j]
            i -= 1
            j -= 1
        return c1[j+1:]/scl, pu.trimseq(c1[:j+1])

def polypow(cs, pow, maxpower=None) :
    """Raise a polynomial to a power.

    Returns the polynomial `cs` raised to the power `pow`. The argument
    `cs` is a sequence of coefficients ordered from low to high. i.e.,
    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    cs : array_like
        1d array of chebyshev series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to umanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Chebyshev series of power.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    power = int(pow)
    if power != pow or power < 0 :
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower :
        raise ValueError("Power is too large")
    elif power == 0 :
        return np.array([1], dtype=cs.dtype)
    elif power == 1 :
        return cs
    else :
        # This can be made more efficient by using powers of two
        # in the usual way.
        prd = cs
        for i in range(2, power + 1) :
            prd = np.convolve(prd, cs)
        return prd

def polyder(cs, m=1, scl=1) :
    """Differentiate a polynomial.

    Returns the polynomial `cs` differentiated `m` times. The argument `cs`
    is a sequence of coefficients ordered from low to high. i.e., [1,2,3]
    is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    cs: array_like
        1d array of chebyshev series coefficients ordered from low to high.
    m : int, optional
        Order of differentiation, must be non-negative. (default: 1)
    scl : scalar, optional
        The result of each derivation is multiplied by `scl`. The end
        result is multiplication by `scl`**`m`. This is for use in a linear
        change of variable. (default: 1)

    Returns
    -------
    der : ndarray
        polynomial of the derivative.

    See Also
    --------
    polyint

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if m < 0 :
        raise ValueError, "The order of derivation must be positive"
    if not np.isscalar(scl) :
        raise ValueError, "The scl parameter must be a scalar"

    if m == 0 :
        return cs
    elif m >= len(cs) :
        return cs[:1]*0
    else :
        n = len(cs)
        d = np.arange(n)*scl
        for i in range(m) :
            cs[i:] *= d[:n-i]
        return cs[i+1:].copy()

def polyint(cs, m=1, k=[], lbnd=0, scl=1) :
    """Integrate a polynomial.

    Returns the polynomial `cs` integrated from `lbnd` to x `m` times. At
    each iteration the resulting series is multiplied by `scl` and an
    integration constant specified by `k` is added. The scaling factor is
    for use in a linear change of variable. The argument `cs` is a sequence
    of coefficients ordered from low to high. i.e., [1,2,3] is the
    polynomial ``1 + 2*x + 3*x**2``.


    Parameters
    ----------
    cs : array_like
        1d array of chebyshev series coefficients ordered from low to high.
    m : int, optional
        Order of integration, must be positeve. (default: 1)
    k : {[], list, scalar}, optional
        Integration constants. The value of the first integral at zero is
        the first value in the list, the value of the second integral at
        zero is the second value in the list, and so on. If ``[]``
        (default), all constants are set zero.  If `m = 1`, a single scalar
        can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (default: 0)
    scl : scalar, optional
        Following each integration the result is multiplied by `scl` before
        the integration constant is added. (default: 1)

    Returns
    -------
    der : ndarray
        polynomial of the integral.

    Raises
    ------
    ValueError

    See Also
    --------
    polyder

    Examples
    --------

    """
    if np.isscalar(k) :
        k = [k]
    if m < 1 :
        raise ValueError, "The order of integration must be positive"
    if len(k) > m :
        raise ValueError, "Too many integration constants"
    if not np.isscalar(lbnd) :
        raise ValueError, "The lbnd parameter must be a scalar"
    if not np.isscalar(scl) :
        raise ValueError, "The scl parameter must be a scalar"

    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    k = list(k) + [0]*(m - len(k))
    fac = np.arange(1, len(cs) + m)/scl
    ret = np.zeros(len(cs) + m, dtype=cs.dtype)
    ret[m:] = cs
    for i in range(m) :
        ret[m - i:] /= fac[:len(cs) + i]
        ret[m - i - 1] += k[i] - polyval(lbnd, ret[m - i - 1:])
    return ret

def polyval(x, cs):
    """Evaluate a polynomial.

    If `cs` is of length `n`, this function returns :

    ``p(x) = cs[0] + cs[1]*x + ... + cs[n-1]*x**(n-1)``

    If x is a sequence or array then p(x) will have the same shape as x.
    If r is a ring_like object that supports multiplication and addition
    by the values in `cs`, then an object of the same type is returned.

    Parameters
    ----------
    x : array_like, ring_like
        If x is a list or tuple, it is converted to an ndarray. Otherwise
        it must support addition and multiplication with itself and the
        elements of `cs`.
    cs : array_like
        1-d array of Chebyshev coefficients ordered from low to high.

    Returns
    -------
    values : ndarray
        The return array has the same shape as `x`.

    See Also
    --------
    polyfit

    Examples
    --------

    Notes
    -----
    The evaluation uses Horner's method.

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if isinstance(x, tuple) or isinstance(x, list) :
        x = np.asarray(x)

    c0 = cs[-1] + x*0
    for i in range(2, len(cs) + 1) :
        c0 = cs[-i] + c0*x
    return c0

def polyvander(x, deg) :
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points `x`.
    This isn't a true Vandermonde matrix because `x` can be an arbitrary
    ndarray. If ``V`` is the returned matrix and `x` is a 2d array, then
    the elements of ``V`` are ``V[i,j,k] = x[i,j]**k``

    Parameters
    ----------
    x : array_like
        Array of points. The values are converted to double or complex doubles.
    deg : integer
        Degree of the resulting matrix.

    Returns
    -------
    vander : Vandermonde matrix.
        The shape of the returned matrix is ``x.shape + (deg+1,)``. The last
        index is the degree.

    """
    x = np.asarray(x) + 0.0
    order = int(deg) + 1
    v = np.ones(x.shape + (order,), dtype=x.dtype)
    if order > 1 :
        v[...,1] = x
        for i in range(2, order) :
            v[...,i] = x*v[...,i-1]
    return v

def polyfit(x, y, deg, rcond=None, full=False):
    """Least squares fit of polynomial to data.

    Fit a polynomial ``p(x) = p[0] * T_{deq}(x) + ... + p[deg] *
    T_{0}(x)`` of degree `deg` to points `(x, y)`. Returns a vector of
    coefficients `p` that minimises the squared error.

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
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Polynomial coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

    [residuals, rank, singular_values, rcond] : present when `full` = True
        Residuals of the least-squares fit, the effective rank of the
        scaled Vandermonde matrix and its singular values, and the
        specified value of `rcond`. For more details, see `linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', RankWarning)

    See Also
    --------
    polyval : Evaluates a polynomial.
    polyvander : Vandermonde matrix for powers.
    chebfit : least squares fit using Chebyshev series.
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution are the coefficients ``c[i]`` of the polynomial ``P(x)``
    that minimizes the squared error

    ``E = \sum_j |y_j - P(x_j)|^2``.

    This problem is solved by setting up as the overdetermined matrix
    equation

    ``V(x)*c = y``,

    where ``V`` is the Vandermonde matrix of `x`, the elements of ``c`` are
    the coefficients to be solved for, and the elements of `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of ``V``.

    If some of the singular values of ``V`` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coeficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using double precision and polynomials tend to fail at about
    degree 20. Fits using Chebyshev series are generally better
    conditioned, but much can still depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

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
        rcond = len(x)*np.finfo(x.dtype).eps

    # set up the design matrix and solve the least squares equation
    A = polyvander(x, deg)
    scl = np.sqrt((A*A).sum(0))
    c, resids, rank, s = la.lstsq(A/scl, y, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning)

    if full :
        return c, [resids, rank, s, rcond]
    else :
        return c


def polyroots(cs):
    """Roots of a polynomial.

    Compute the roots of the Chebyshev series `cs`. The argument `cs` is a
    sequence of coefficients ordered from low to high. i.e., [1,2,3] is the
    polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    cs : array_like of shape(M,)
        1D array of polynomial coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        An array containing the complex roots of the polynomial series.

    Examples
    --------

    """
    # cs is a trimmed copy
    [cs] = pu.as_series([cs])
    if len(cs) <= 1 :
        return np.array([], dtype=cs.dtype)
    if len(cs) == 2 :
        return np.array([-cs[0]/cs[1]])
    n = len(cs) - 1
    cmat = np.zeros((n,n), dtype=cs.dtype)
    cmat.flat[n::n+1] = 1
    cmat[:,-1] -= cs[:-1]/cs[-1]
    roots = la.eigvals(cmat)
    roots.sort()
    return roots


#
# polynomial class
#

exec polytemplate.substitute(name='Polynomial', nick='poly', domain='[-1,1]')

