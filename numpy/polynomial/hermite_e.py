"""
Objects for dealing with Hermite_e series.

This module provides a number of objects (mostly functions) useful for
dealing with Hermite_e series, including a `HermiteE` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Constants
---------
- `hermedomain` -- Hermite_e series default domain, [-1,1].
- `hermezero` -- Hermite_e series that evaluates identically to 0.
- `hermeone` -- Hermite_e series that evaluates identically to 1.
- `hermex` -- Hermite_e series for the identity map, ``f(x) = x``.

Arithmetic
----------
- `hermemulx` -- multiply a Hermite_e series in ``P_i(x)`` by ``x``.
- `hermeadd` -- add two Hermite_e series.
- `hermesub` -- subtract one Hermite_e series from another.
- `hermemul` -- multiply two Hermite_e series.
- `hermediv` -- divide one Hermite_e series by another.
- `hermeval` -- evaluate a Hermite_e series at given points.
- `hermeval2d` -- evaluate a 2D Hermite_e series at given points.
- `hermeval3d` -- evaluate a 3D Hermite_e series at given points.
- `hermegrid2d` -- evaluate a 2D Hermite_e series on a Cartesian product.
- `hermegrid3d` -- evaluate a 3D Hermite_e series on a Cartesian product.

Calculus
--------
- `hermeder` -- differentiate a Hermite_e series.
- `hermeint` -- integrate a Hermite_e series.

Misc Functions
--------------
- `hermefromroots` -- create a Hermite_e series with specified roots.
- `hermeroots` -- find the roots of a Hermite_e series.
- `hermevander` -- Vandermonde-like matrix for Hermite_e polynomials.
- `hermevander2d` -- Vandermonde-like matrix for 2D power series.
- `hermevander3d` -- Vandermonde-like matrix for 3D power series.
- `hermegauss` -- Gauss-Hermite_e quadrature, points and weights.
- `hermeweight` -- Hermite_e weight function.
- `hermecompanion` -- symmetrized companion matrix in Hermite_e form.
- `hermefit` -- least-squares fit returning a Hermite_e series.
- `hermetrim` -- trim leading coefficients from a Hermite_e series.
- `hermeline` -- Hermite_e series of given straight line.
- `herme2poly` -- convert a Hermite_e series to a polynomial.
- `poly2herme` -- convert a polynomial to a Hermite_e series.

Classes
-------
- `HermiteE` -- A Hermite_e series class.

See also
--------
`numpy.polynomial`

"""
from __future__ import division

import numpy as np
import numpy.linalg as la
import polyutils as pu
import warnings
from polytemplate import polytemplate

__all__ = ['hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline',
    'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv', 'hermpow',
    'hermeval',
    'hermeder', 'hermeint', 'herme2poly', 'poly2herme', 'hermefromroots',
    'hermevander', 'hermefit', 'hermetrim', 'hermeroots', 'HermiteE',
    'hermeval2d', 'hermeval3d', 'hermegrid2d', 'hermegrid3d', 'hermevander2d',
    'hermevander3d', 'hermecompanion', 'hermegauss', 'hermeweight']

hermetrim = pu.trimcoef


def poly2herme(pol) :
    """
    poly2herme(pol)

    Convert a polynomial to a Hermite series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Hermite series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-d array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-d array containing the coefficients of the equivalent Hermite
        series.

    See Also
    --------
    herme2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import poly2herme
    >>> poly2herme(np.arange(4))
    array([  2.,  10.,   2.,   3.])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1) :
        res = hermeadd(hermemulx(res), pol[i])
    return res


def herme2poly(c) :
    """
    Convert a Hermite series to a polynomial.

    Convert an array representing the coefficients of a Hermite series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-d array containing the Hermite series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-d array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2herme

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import herme2poly
    >>> herme2poly([  2.,  10.,   2.,   3.])
    array([ 0.,  1.,  2.,  3.])

    """
    from polynomial import polyadd, polysub, polymulx

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1) :
            tmp = c0
            c0 = polysub(c[i - 2], c1*(i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        return polyadd(c0, polymulx(c1))

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Hermite
hermedomain = np.array([-1,1])

# Hermite coefficients representing zero.
hermezero = np.array([0])

# Hermite coefficients representing one.
hermeone = np.array([1])

# Hermite coefficients representing the identity x.
hermex = np.array([0, 1])


def hermeline(off, scl) :
    """
    Hermite series whose graph is a straight line.



    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Hermite series for
        ``off + scl*x``.

    See Also
    --------
    polyline, chebline

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeline
    >>> from numpy.polynomial.hermite_e import hermeline, hermeval
    >>> hermeval(0,hermeline(3, 2))
    3.0
    >>> hermeval(1,hermeline(3, 2))
    5.0

    """
    if scl != 0 :
        return np.array([off,scl])
    else :
        return np.array([off])


def hermefromroots(roots) :
    """
    Generate a HermiteE series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in HermiteE form, where the `r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * He_1(x) + ... +  c_n * He_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in HermiteE form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    polyfromroots, legfromroots, lagfromroots, hermfromroots,
    chebfromroots.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermefromroots, hermeval
    >>> coef = hermefromroots((-1, 0, 1))
    >>> hermeval((-1, 0, 1), coef)
    array([ 0.,  0.,  0.])
    >>> coef = hermefromroots((-1j, 1j))
    >>> hermeval((-1j, 1j), coef)
    array([ 0.+0.j,  0.+0.j])

    """
    if len(roots) == 0 :
        return np.ones(1)
    else :
        [roots] = pu.as_series([roots], trim=False)
        prd = np.array([1], dtype=roots.dtype)
        for r in roots:
            prd = hermesub(hermemulx(prd), r*prd)
        return prd


def hermeadd(c1, c2):
    """
    Add one Hermite series to another.

    Returns the sum of two Hermite series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Hermite series of their sum.

    See Also
    --------
    hermesub, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Hermite series
    is a Hermite series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeadd
    >>> hermeadd([1, 2, 3], [1, 2, 3, 4])
    array([ 2.,  4.,  6.,  4.])

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


def hermesub(c1, c2):
    """
    Subtract one Hermite series from another.

    Returns the difference of two Hermite series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their difference.

    See Also
    --------
    hermeadd, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Hermite
    series is a Hermite series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermesub
    >>> hermesub([1, 2, 3, 4], [1, 2, 3])
    array([ 0.,  0.,  0.,  4.])

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


def hermemulx(c):
    """Multiply a Hermite series by x.

    Multiply the Hermite series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-d array of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    Notes
    -----
    The multiplication uses the recursion relationship for Hermite
    polynomials in the form

    .. math::

    xP_i(x) = (P_{i + 1}(x) + iP_{i - 1}(x)))

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemulx
    >>> hermemulx([1, 2, 3])
    array([ 2.,  7.,  2.,  3.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1] = c[0]
    for i in range(1, len(c)):
        prd[i + 1] = c[i]
        prd[i - 1] += c[i]*i
    return prd


def hermemul(c1, c2):
    """
    Multiply one Hermite series by another.

    Returns the product of two Hermite series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their product.

    See Also
    --------
    hermeadd, hermesub, hermediv, hermepow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "re-project" the
    product onto said basis set, which may produce "un-intuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemul
    >>> hermemul([1, 2, 3], [0, 1, 2])
    array([ 14.,  15.,  28.,   7.,   6.])

    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0]*xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]*xs
        c1 = c[1]*xs
    else :
        nd = len(c)
        c0 = c[-2]*xs
        c1 = c[-1]*xs
        for i in range(3, len(c) + 1) :
            tmp = c0
            nd =  nd - 1
            c0 = hermesub(c[-i]*xs, c1*(nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1))
    return hermeadd(c0, hermemulx(c1))


def hermediv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-d arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermeadd, hermesub, hermemul, hermepow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "re-project" the results onto the Hermite
    basis set, which may produce "un-intuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermediv
    >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 0.]))
    >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 1.,  2.]))

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0 :
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2 :
        return c1[:1]*0, c1
    elif lc2 == 1 :
        return c1/c2[-1], c1[:1]*0
    else :
        quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        for i in range(lc1 - lc2, - 1, -1):
            p = hermemul([0]*i + [1], c2)
            q = rem[-1]/p[-1]
            rem = rem[:-1] - q*p[:-1]
            quo[i] = q
        return quo, pu.trimseq(rem)


def hermepow(c, pow, maxpower=16) :
    """Raise a Hermite series to a power.

    Returns the Hermite series `c` raised to the power `pow`. The
    arguement `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1d array of Hermite series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to umanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Hermite series of power.

    See Also
    --------
    hermeadd, hermesub, hermemul, hermediv

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermepow
    >>> hermepow([1, 2, 3], 2)
    array([ 23.,  28.,  46.,  12.,   9.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    power = int(pow)
    if power != pow or power < 0 :
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower :
        raise ValueError("Power is too large")
    elif power == 0 :
        return np.array([1], dtype=c.dtype)
    elif power == 1 :
        return c
    else :
        # This can be made more efficient by using powers of two
        # in the usual way.
        prd = c
        for i in range(2, power + 1) :
            prd = hermemul(prd, c)
        return prd


def hermeder(c, m=1, scl=1, axis=0) :
    """
    Differentiate a Hermite_e series.

    Returns the series coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*He_0 + 2*He_1 + 3*He_2``
    while [[1,2],[1,2]] represents ``1*He_0(x)*He_0(y) + 1*He_1(x)*He_0(y)
    + 2*He_0(x)*He_1(y) + 2*He_1(x)*He_1(y)`` if axis=0 is ``x`` and axis=1
    is ``y``.

    Parameters
    ----------
    c: array_like
        Array of Hermite_e series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Hermite series of the derivative.

    See Also
    --------
    hermeint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "un-intuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeder
    >>> hermeder([ 1.,  1.,  1.,  1.])
    array([ 1.,  2.,  3.])
    >>> hermeder([-0.25,  1.,  1./2.,  1./3.,  1./4 ], m=2)
    array([ 1.,  2.,  3.])

    """
    c = np.array(c, ndmin=1, copy=1)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt, iaxis = [int(t) for t in [m, axis]]

    if cnt != m:
        raise ValueError("The order of derivation must be integer")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    if iaxis != axis:
        raise ValueError("The axis must be integer")
    if not -c.ndim <= iaxis < c.ndim:
        raise ValueError("The axis is out of range")
    if iaxis < 0:
        iaxis += c.ndim

    if cnt == 0:
        return c

    c = np.rollaxis(c, iaxis)
    n = len(c)
    if cnt >= n:
        return c[:1]*0
    else :
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = j*c[j]
            c = der
    c = np.rollaxis(c, 0, iaxis + 1)
    return c


def hermeint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite_e series.

    Returns the Hermite_e series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axix, e.g., [1,2,3]
    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Hermite_e series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
        ``np.isscalar(scl) == False``.

    See Also
    --------
    hermeder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    .. math::`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "re-projected" onto the C-series basis set.  Thus, typically,
    the result of this function is "un-intuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeint
    >>> hermeint([1, 2, 3]) # integrate once, value 0 at 0.
    array([ 1.,  1.,  1.,  1.])
    >>> hermeint([1, 2, 3], m=2) # integrate twice, value & deriv 0 at 0
    array([-0.25      ,  1.        ,  0.5       ,  0.33333333,  0.25      ])
    >>> hermeint([1, 2, 3], k=1) # integrate once, value 1 at 0.
    array([ 2.,  1.,  1.,  1.])
    >>> hermeint([1, 2, 3], lbnd=-1) # integrate once, value 0 at -1
    array([-1.,  1.,  1.,  1.])
    >>> hermeint([1, 2, 3], m=2, k=[1, 2], lbnd=-1)
    array([ 1.83333333,  0.        ,  0.5       ,  0.33333333,  0.25      ])

    """
    c = np.array(c, ndmin=1, copy=1)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt, iaxis = [int(t) for t in [m, axis]]

    if cnt != m:
        raise ValueError("The order of integration must be integer")
    if cnt < 0 :
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt :
        raise ValueError("Too many integration constants")
    if iaxis != axis:
        raise ValueError("The axis must be integer")
    if not -c.ndim <= iaxis < c.ndim:
        raise ValueError("The axis is out of range")
    if iaxis < 0:
        iaxis += c.ndim

    if cnt == 0:
        return c

    c = np.rollaxis(c, iaxis)
    k = list(k) + [0]*(cnt - len(k))
    for i in range(cnt) :
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j]/(j + 1)
            tmp[0] += k[i] - hermeval(lbnd, tmp)
            c = tmp
    c = np.rollaxis(c, 0, iaxis + 1)
    return c


def hermeval(x, c, tensor=True):
    """
    Evaluate an HermiteE series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * He_0(x) + c_1 * He_1(x) + ... + c_n * He_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimesional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

        .. versionadded:: 1.7.0

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    hermeval2d, hermegrid2d, hermeval3d, hermegrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeval
    >>> coef = [1,2,3]
    >>> hermeval(1, coef)
    3.0
    >>> hermeval([[1,2],[3,4]], coef)
    array([[  3.,  14.],
           [ 31.,  54.]])

    """
    c = np.array(c, ndmin=1, copy=0)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
       c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1 :
        c0 = c[0]
        c1 = 0
    elif len(c) == 2 :
        c0 = c[0]
        c1 = c[1]
    else :
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1) :
            tmp = c0
            nd =  nd - 1
            c0 = c[-i] - c1*(nd - 1)
            c1 = tmp + c1*x
    return c0 + c1*x


def hermeval2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * He_i(x) * He_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    .. versionadded::1.7.0

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points `(x, y)`,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    hermeval, hermegrid2d, hermeval3d, hermegrid3d

    """
    try:
        x, y = np.array((x, y), copy=0)
    except:
        raise ValueError('x, y are incompatible')

    c = hermeval(x, c)
    c = hermeval(y, c, tensor=False)
    return c


def hermegrid2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series on the Cartesion product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \sum_{i,j} c_{i,j} * H_i(a) * H_j(b)

    where the points `(a, b)` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesion
        product of `x` and `y`.

    See Also
    --------
    hermeval, hermeval2d, hermeval3d, hermegrid3d

    """
    c = hermeval(x, c)
    c = hermeval(y, c)
    return c


def hermeval3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite_e series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * He_i(x) * He_j(y) * He_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    .. versionadded::1.7.0

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimension polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    hermeval, hermeval2d, hermegrid2d, hermegrid3d

    """
    try:
        x, y, z = np.array((x, y, z), copy=0)
    except:
        raise ValueError('x, y, z are incompatible')

    c = hermeval(x, c)
    c = hermeval(y, c, tensor=False)
    c = hermeval(z, c, tensor=False)
    return c


def hermegrid3d(x, y, z, c):
    """
    Evaluate a 3-D HermiteE series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * He_i(a) * He_j(b) * He_k(c)

    where the points `(a, b, c)` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + yshape + z.shape.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesion
        product of `x` and `y`.

    See Also
    --------
    hermeval, hermeval2d, hermegrid2d, hermeval3d

    """
    c = hermeval(x, c)
    c = hermeval(y, c)
    c = hermeval(z, c)
    return c


def hermevander(x, deg) :
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points `x`.
    This isn't a true Vandermonde matrix because `x` can be an arbitrary
    ndarray and the Hermite polynomials aren't powers. If ``V`` is the
    returned matrix and `x` is a 2d array, then the elements of ``V`` are
    ``V[i,j,k] = P_k(x[i,j])``, where ``P_k`` is the Hermite polynomial
    of degree ``k``.

    Parameters
    ----------
    x : array_like
        Array of points. The values are converted to double or complex
        doubles. If x is scalar it is converted to a 1D array.
    deg : integer
        Degree of the resulting matrix.

    Returns
    -------
    vander : Vandermonde matrix.
        The shape of the returned matrix is ``x.shape + (deg+1,)``. The last
        index is the degree.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermevander
    >>> x = np.array([-1, 0, 1])
    >>> hermevander(x, 3)
    array([[ 1., -1.,  0.,  2.],
           [ 1.,  0., -1., -0.],
           [ 1.,  1.,  0., -2.]])

    """
    ideg = int(deg)
    if ideg != deg:
        raise ValueError("deg must be integer")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=0, ndmin=1) + 0.0
    v = np.empty((ideg + 1,) + x.shape, dtype=x.dtype)
    v[0] = x*0 + 1
    if ideg > 0 :
        v[1] = x
        for i in range(2, ideg + 1) :
            v[i] = (v[i-1]*x - v[i-2]*(i - 1))
    return np.rollaxis(v, 0, v.ndim)


def hermevander2d(x, y, deg) :
    """Pseudo Vandermonde matrix of given degree.

    Returns the pseudo Vandermonde matrix for 2D Hermite_e series in `x`
    and `y`. The sample point coordinates must all have the same shape
    after conversion to arrays and the dtype will be converted to either
    float64 or complex128 depending on whether any of `x` or 'y' are
    complex.  The maximum degrees of the 2D Hermite_e series in each
    variable are specified in the list `deg` in the form ``[xdeg, ydeg]``.
    The return array has the shape ``x.shape + (order,)`` if `x`, and `y`
    are arrays or ``(1, order) if they are scalars. Here order is the
    number of elements in a flattened coefficient array of original shape
    ``(xdeg + 1, ydeg + 1)``.  The flattening is done so that the resulting
    pseudo Vandermonde array can be easily used in least squares fits.

    Parameters
    ----------
    x,y : array_like
        Arrays of point coordinates, each of the same shape.
    deg : list
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is described above.

    See Also
    --------
    hermevander, hermevander3d. hermeval2d, hermeval3d

    """
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [1, 1]:
        raise ValueError("degrees must be non-negative integers")
    degx, degy = ideg
    x, y = np.array((x, y), copy=0) + 0.0

    vx = hermevander(x, degx)
    vy = hermevander(y, degy)
    v = vx[..., None]*vy[..., None, :]
    return v.reshape(v.shape[:-2] + (-1,))


def hermevander3d(x, y, z, deg) :
    """Pseudo Vandermonde matrix of given degree.

    Returns the pseudo Vandermonde matrix for 3D Hermite_e series in `x`,
    `y`, or `z`. The sample point coordinates must all have the same shape
    after conversion to arrays and the dtype will be converted to either
    float64 or complex128 depending on whether any of `x`, `y`, or 'z' are
    complex.  The maximum degrees of the 3D Hermite_e series in each
    variable are specified in the list `deg` in the form ``[xdeg, ydeg,
    zdeg]``. The return array has the shape ``x.shape + (order,)`` if `x`,
    `y`, and `z` are arrays or ``(1, order) if they are scalars. Here order
    is the number of elements in a flattened coefficient array of original
    shape ``(xdeg + 1, ydeg + 1, zdeg + 1)``.  The flattening is done so
    that the resulting pseudo Vandermonde array can be easily used in least
    squares fits.

    Parameters
    ----------
    x,y,z : array_like
        Arrays of point coordinates, each of the same shape.
    deg : list
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is described above.

    See Also
    --------
    hermevander, hermevander3d. hermeval2d, hermeval3d

    """
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [1, 1, 1]:
        raise ValueError("degrees must be non-negative integers")
    degx, degy, degz = ideg
    x, y, z = np.array((x, y, z), copy=0) + 0.0

    vx = hermevander(x, degx)
    vy = hermevander(y, degy)
    vz = hermevander(z, degz)
    v = vx[..., None, None]*vy[..., None, :, None]*vz[..., None, None, :]
    return v.reshape(v.shape[:-3] + (-1,))


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Hermite series to data.

    Fit a Hermite series ``p(x) = p[0] * P_{0}(x) + ... + p[deg] *
    P_{deg}(x)`` of degree `deg` to points `(x, y)`. Returns a vector of
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
    w : array_like, shape (`M`,), optional
        Weights. If not None, the contribution of each point
        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
        weights are chosen so that the errors of the products ``w[i]*y[i]``
        all have the same variance.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Hermite coefficients ordered from low to high. If `y` was 2-D,
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
    hermeval : Evaluates a Hermite series.
    hermevander : Vandermonde matrix of Hermite series.
    polyfit : least squares fit using polynomials.
    chebfit : least squares fit using Chebyshev series.
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution are the coefficients ``c[i]`` of the Hermite series
    ``P(x)`` that minimizes the squared error

    ``E = \\sum_j |y_j - P(x_j)|^2``.

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

    Fits using Hermite series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermefit, hermeval
    >>> x = np.linspace(-10, 10)
    >>> err = np.random.randn(len(x))/10
    >>> y = hermeval(x, [1, 2, 3]) + err
    >>> hermefit(x, y, 2)
    array([ 1.01690445,  1.99951418,  2.99948696])

    """
    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    # set up the least squares matrices
    lhs = hermevander(x, deg)
    rhs = y
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")
        # apply weights
        if rhs.ndim == 2:
            lhs *= w[:, np.newaxis]
            rhs *= w[:, np.newaxis]
        else:
            lhs *= w[:, np.newaxis]
            rhs *= w

    # set rcond
    if rcond is None :
        rcond = len(x)*np.finfo(x.dtype).eps

    # scale the design matrix and solve the least squares equation
    scl = np.sqrt((lhs*lhs).sum(0))
    c, resids, rank, s = la.lstsq(lhs/scl, rhs, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning)

    if full :
        return c, [resids, rank, s, rcond]
    else :
        return c


def hermecompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` represents a single HermiteE polynomial. This
    provides better eigenvalue estimates than the unscaled case and in the
    single polynomial case the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-d array of Legendre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    """
    accprod = np.multiply.accumulate
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array(-c[0]/c[1])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = np.hstack((1., np.sqrt(np.arange(1,n))))
    scl = np.multiply.accumulate(scl)
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[...] = np.sqrt(np.arange(1,n))
    bot[...] = top
    mat[:,-1] -= (c[:-1]/c[-1])*(scl/scl[-1])
    return mat


def hermeroots(c):
    """
    Compute the roots of a HermiteE series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * He_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    polyroots, legroots, lagroots, hermroots, chebroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The HermiteE series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeroots, hermefromroots
    >>> coef = hermefromroots([-1, 0, 1])
    >>> coef
    array([ 0.,  2.,  0.,  1.])
    >>> hermeroots(coef)
    array([-1.,  0.,  1.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1 :
        return np.array([], dtype=c.dtype)
    if len(c) == 2 :
        return np.array([-c[0]/c[1]])

    m = hermecompanion(c)
    r = la.eigvals(m)
    r.sort()
    return r


def hermegauss(deg):
    """Gauss Hermite_e quadrature.

    Computes the sample points and weights for Gauss-Hermite_e quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree ``2*deg - 1`` or less over the interval ``[-inf, inf]`` with the
    weight function ``f(x) = exp(-.5*x**2)``.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100. Higher degrees may
    be problematic. The weights are determined by using the fact that

          w = c / (He'_n(x_k) * He_{n-1}(x_k))

    where ``c`` is a constant independent of ``k`` and ``x_k`` is the k'th
    root of ``He_n``, and then scaling the results to get the right value
    when integrating 1.

    """
    ideg = int(deg)
    if ideg != deg or ideg < 1:
        raise ValueError("deg must be a non-negative integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0]*deg + [1])
    m = hermecompanion(c)
    x = la.eigvals(m)
    x.sort()

    # improve roots by one application of Newton
    dy = hermeval(x, c)
    df = hermeval(x, hermeder(c))
    x -= dy/df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = hermeval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1/(fm * df)

    # for Hermite_e we can also symmetrize
    w = (w + w[::-1])/2
    x = (x - x[::-1])/2

    # scale w to get the right value
    w *= np.sqrt(2*np.pi) / w.sum()

    return x, w


def hermeweight(x):
    """Weight function of the Hermite_e polynomials.

    The weight function for which the Hermite_e polynomials are orthogonal.
    In this case the weight function is ``exp(-.5*x**2)``. Note that the
    Hermite_e polynomials are not normalized.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    """
    w = np.exp(-.5*x**2)
    return w


#
# HermiteE series class
#

exec polytemplate.substitute(name='HermiteE', nick='herme', domain='[-1,1]')
