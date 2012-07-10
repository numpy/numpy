"""
Objects for dealing with Hermite series.

This module provides a number of objects (mostly functions) useful for
dealing with Hermite series, including a `Hermite` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Constants
---------
- `hermdomain` -- Hermite series default domain, [-1,1].
- `hermzero` -- Hermite series that evaluates identically to 0.
- `hermone` -- Hermite series that evaluates identically to 1.
- `hermx` -- Hermite series for the identity map, ``f(x) = x``.

Arithmetic
----------
- `hermmulx` -- multiply a Hermite series in ``P_i(x)`` by ``x``.
- `hermadd` -- add two Hermite series.
- `hermsub` -- subtract one Hermite series from another.
- `hermmul` -- multiply two Hermite series.
- `hermdiv` -- divide one Hermite series by another.
- `hermval` -- evaluate a Hermite series at given points.
- `hermval2d` -- evaluate a 2D Hermite series at given points.
- `hermval3d` -- evaluate a 3D Hermite series at given points.
- `hermgrid2d` -- evaluate a 2D Hermite series on a Cartesian product.
- `hermgrid3d` -- evaluate a 3D Hermite series on a Cartesian product.

Calculus
--------
- `hermder` -- differentiate a Hermite series.
- `hermint` -- integrate a Hermite series.

Misc Functions
--------------
- `hermfromroots` -- create a Hermite series with specified roots.
- `hermroots` -- find the roots of a Hermite series.
- `hermvander` -- Vandermonde-like matrix for Hermite polynomials.
- `hermvander2d` -- Vandermonde-like matrix for 2D power series.
- `hermvander3d` -- Vandermonde-like matrix for 3D power series.
- `hermgauss` -- Gauss-Hermite quadrature, points and weights.
- `hermweight` -- Hermite weight function.
- `hermcompanion` -- symmetrized companion matrix in Hermite form.
- `hermfit` -- least-squares fit returning a Hermite series.
- `hermtrim` -- trim leading coefficients from a Hermite series.
- `hermline` -- Hermite series of given straight line.
- `herm2poly` -- convert a Hermite series to a polynomial.
- `poly2herm` -- convert a polynomial to a Hermite series.

Classes
-------
- `Hermite` -- A Hermite series class.

See also
--------
`numpy.polynomial`

"""
from __future__ import division

import numpy as np
import numpy.linalg as la
import polyutils as pu
import warnings

__all__ = ['hermzero', 'hermone', 'hermx', 'hermdomain', 'hermline',
    'hermadd', 'hermsub', 'hermmulx', 'hermmul', 'hermdiv', 'hermpow',
    'hermval', 'hermder', 'hermint', 'herm2poly', 'poly2herm',
    'hermfromroots', 'hermvander', 'hermfit', 'hermtrim', 'hermroots',
    'Hermite', 'hermval2d', 'hermval3d', 'hermgrid2d', 'hermgrid3d',
    'hermvander2d', 'hermvander3d', 'hermcompanion', 'hermgauss',
    'hermweight']

hermtrim = pu.trimcoef


def poly2herm(pol) :
    """
    poly2herm(pol)

    Convert a polynomial to a Hermite series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Hermite series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Hermite
        series.

    See Also
    --------
    herm2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import poly2herme
    >>> poly2herm(np.arange(4))
    array([ 1.   ,  2.75 ,  0.5  ,  0.375])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1) :
        res = hermadd(hermmulx(res), pol[i])
    return res


def herm2poly(c) :
    """
    Convert a Hermite series to a polynomial.

    Convert an array representing the coefficients of a Hermite series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Hermite series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2herm

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite import herm2poly
    >>> herm2poly([ 1.   ,  2.75 ,  0.5  ,  0.375])
    array([ 0.,  1.,  2.,  3.])

    """
    from polynomial import polyadd, polysub, polymulx

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        c[1] *= 2
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1) :
            tmp = c0
            c0 = polysub(c[i - 2], c1*(2*(i - 1)))
            c1 = polyadd(tmp, polymulx(c1)*2)
        return polyadd(c0, polymulx(c1)*2)

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Hermite
hermdomain = np.array([-1,1])

# Hermite coefficients representing zero.
hermzero = np.array([0])

# Hermite coefficients representing one.
hermone = np.array([1])

# Hermite coefficients representing the identity x.
hermx = np.array([0, 1/2])


def hermline(off, scl) :
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
    >>> from numpy.polynomial.hermite import hermline, hermval
    >>> hermval(0,hermline(3, 2))
    3.0
    >>> hermval(1,hermline(3, 2))
    5.0

    """
    if scl != 0 :
        return np.array([off,scl/2])
    else :
        return np.array([off])


def hermfromroots(roots) :
    """
    Generate a Hermite series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Hermite form, where the `r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * H_1(x) + ... +  c_n * H_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Hermite form.

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
    polyfromroots, legfromroots, lagfromroots, chebfromroots,
    hermefromroots.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermfromroots, hermval
    >>> coef = hermfromroots((-1, 0, 1))
    >>> hermval((-1, 0, 1), coef)
    array([ 0.,  0.,  0.])
    >>> coef = hermfromroots((-1j, 1j))
    >>> hermval((-1j, 1j), coef)
    array([ 0.+0.j,  0.+0.j])

    """
    if len(roots) == 0 :
        return np.ones(1)
    else :
        [roots] = pu.as_series([roots], trim=False)
        roots.sort()
        p = [hermline(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [hermmul(p[i], p[i+m]) for i in range(m)]
            if r:
                tmp[0] = hermmul(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]


def hermadd(c1, c2):
    """
    Add one Hermite series to another.

    Returns the sum of two Hermite series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Hermite series of their sum.

    See Also
    --------
    hermsub, hermmul, hermdiv, hermpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Hermite series
    is a Hermite series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermadd
    >>> hermadd([1, 2, 3], [1, 2, 3, 4])
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


def hermsub(c1, c2):
    """
    Subtract one Hermite series from another.

    Returns the difference of two Hermite series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their difference.

    See Also
    --------
    hermadd, hermmul, hermdiv, hermpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Hermite
    series is a Hermite series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermsub
    >>> hermsub([1, 2, 3, 4], [1, 2, 3])
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


def hermmulx(c):
    """Multiply a Hermite series by x.

    Multiply the Hermite series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
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

    xP_i(x) = (P_{i + 1}(x)/2 + i*P_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermmulx
    >>> hermmulx([1, 2, 3])
    array([ 2. ,  6.5,  1. ,  1.5])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1] = c[0]/2
    for i in range(1, len(c)):
        prd[i + 1] = c[i]/2
        prd[i - 1] += c[i]*i
    return prd


def hermmul(c1, c2):
    """
    Multiply one Hermite series by another.

    Returns the product of two Hermite series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their product.

    See Also
    --------
    hermadd, hermsub, hermdiv, hermpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermmul
    >>> hermmul([1, 2, 3], [0, 1, 2])
    array([ 52.,  29.,  52.,   7.,   6.])

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
            c0 = hermsub(c[-i]*xs, c1*(2*(nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1)*2)
    return hermadd(c0, hermmulx(c1)*2)


def hermdiv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermadd, hermsub, hermmul, hermpow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "reproject" the results onto the Hermite
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermdiv
    >>> hermdiv([ 52.,  29.,  52.,   7.,   6.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 0.]))
    >>> hermdiv([ 54.,  31.,  52.,   7.,   6.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 2.,  2.]))
    >>> hermdiv([ 53.,  30.,  52.,   7.,   6.], [0, 1, 2])
    (array([ 1.,  2.,  3.]), array([ 1.,  1.]))

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
            p = hermmul([0]*i + [1], c2)
            q = rem[-1]/p[-1]
            rem = rem[:-1] - q*p[:-1]
            quo[i] = q
        return quo, pu.trimseq(rem)


def hermpow(c, pow, maxpower=16) :
    """Raise a Hermite series to a power.

    Returns the Hermite series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Hermite series of power.

    See Also
    --------
    hermadd, hermsub, hermmul, hermdiv

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermpow
    >>> hermpow([1, 2, 3], 2)
    array([ 81.,  52.,  82.,  12.,   9.])

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
            prd = hermmul(prd, c)
        return prd


def hermder(c, m=1, scl=1, axis=0) :
    """
    Differentiate a Hermite series.

    Returns the Hermite series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*H_0 + 2*H_1 + 3*H_2``
    while [[1,2],[1,2]] represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) +
    2*H_0(x)*H_1(y) + 2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite series coefficients. If `c` is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    der : ndarray
        Hermite series of the derivative.

    See Also
    --------
    hermint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermder
    >>> hermder([ 1. ,  0.5,  0.5,  0.5])
    array([ 1.,  2.,  3.])
    >>> hermder([-0.5,  1./2.,  1./8.,  1./12.,  1./16.], m=2)
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
        c = c[:1]*0
    else :
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = (2*j)*c[j]
            c = der
    c = np.rollaxis(c, 0, iaxis + 1)
    return c


def hermint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite series.

    Returns the Hermite series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
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
        Axis over which the integral is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    S : ndarray
        Hermite series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
        ``np.isscalar(scl) == False``.

    See Also
    --------
    hermder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    .. math::`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermint
    >>> hermint([1,2,3]) # integrate once, value 0 at 0.
    array([ 1. ,  0.5,  0.5,  0.5])
    >>> hermint([1,2,3], m=2) # integrate twice, value & deriv 0 at 0
    array([-0.5       ,  0.5       ,  0.125     ,  0.08333333,  0.0625    ])
    >>> hermint([1,2,3], k=1) # integrate once, value 1 at 0.
    array([ 2. ,  0.5,  0.5,  0.5])
    >>> hermint([1,2,3], lbnd=-1) # integrate once, value 0 at -1
    array([-2. ,  0.5,  0.5,  0.5])
    >>> hermint([1,2,3], m=2, k=[1,2], lbnd=-1)
    array([ 1.66666667, -0.5       ,  0.125     ,  0.08333333,  0.0625    ])

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
            tmp[1] = c[0]/2
            for j in range(1, n):
                tmp[j + 1] = c[j]/(2*(j + 1))
            tmp[0] += k[i] - hermval(lbnd, tmp)
            c = tmp
    c = np.rollaxis(c, 0, iaxis + 1)
    return c


def hermval(x, c, tensor=True):
    """
    Evaluate an Hermite series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * H_0(x) + c_1 * H_1(x) + ... + c_n * H_n(x)

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
        degree n are contained in c[n]. If `c` is multidimensional the
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
    hermval2d, hermgrid2d, hermval3d, hermgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermval
    >>> coef = [1,2,3]
    >>> hermval(1, coef)
    11.0
    >>> hermval([[1,2],[3,4]], coef)
    array([[  11.,   51.],
           [ 115.,  203.]])

    """
    c = np.array(c, ndmin=1, copy=0)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    x2 = x*2
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
            c0 = c[-i] - c1*(2*(nd - 1))
            c1 = tmp + c1*x2
    return c0 + c1*x2


def hermval2d(x, y, c):
    """
    Evaluate a 2-D Hermite series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * H_i(x) * H_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

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
    hermval, hermgrid2d, hermval3d, hermgrid3d

    Notes
    -----

    .. versionadded::1.7.0

    """
    try:
        x, y = np.array((x, y), copy=0)
    except:
        raise ValueError('x, y are incompatible')

    c = hermval(x, c)
    c = hermval(y, c, tensor=False)
    return c


def hermgrid2d(x, y, c):
    """
    Evaluate a 2-D Hermite series on the Cartesian product of x and y.

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
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    hermval, hermval2d, hermval3d, hermgrid3d

    Notes
    -----

    .. versionadded::1.7.0

    """
    c = hermval(x, c)
    c = hermval(y, c)
    return c


def hermval3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * H_i(x) * H_j(y) * H_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

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
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    hermval, hermval2d, hermgrid2d, hermgrid3d

    Notes
    -----

    .. versionadded::1.7.0

    """
    try:
        x, y, z = np.array((x, y, z), copy=0)
    except:
        raise ValueError('x, y, z are incompatible')

    c = hermval(x, c)
    c = hermval(y, c, tensor=False)
    c = hermval(z, c, tensor=False)
    return c


def hermgrid3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * H_i(a) * H_j(b) * H_k(c)

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
    x.shape + y.shape + z.shape.

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
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    hermval, hermval2d, hermgrid2d, hermval3d

    Notes
    -----

    .. versionadded::1.7.0

    """
    c = hermval(x, c)
    c = hermval(y, c)
    c = hermval(z, c)
    return c


def hermvander(x, deg) :
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = H_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Hermite polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    array ``V = hermvander(x, n)``, then ``np.dot(V, c)`` and
    ``hermval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Hermite series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander: ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Hermite polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermvander
    >>> x = np.array([-1, 0, 1])
    >>> hermvander(x, 3)
    array([[ 1., -2.,  2.,  4.],
           [ 1.,  0., -2., -0.],
           [ 1.,  2.,  2., -4.]])

    """
    ideg = int(deg)
    if ideg != deg:
        raise ValueError("deg must be integer")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=0, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x*0 + 1
    if ideg > 0 :
        x2 = x*2
        v[1] = x2
        for i in range(2, ideg + 1) :
            v[i] = (v[i-1]*x2 - v[i-2]*(2*(i - 1)))
    return np.rollaxis(v, 0, v.ndim)


def hermvander2d(x, y, deg) :
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y)`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., deg[1]*i + j] = H_i(x) * H_j(y),

    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
    `V` index the points `(x, y)` and the last index encodes the degrees of
    the Hermite polynomials.

    If ``V = hermvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``hermval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Hermite
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    hermvander, hermvander3d. hermval2d, hermval3d

    Notes
    -----

    .. versionadded::1.7.0

    """
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [1, 1]:
        raise ValueError("degrees must be non-negative integers")
    degx, degy = ideg
    x, y = np.array((x, y), copy=0) + 0.0

    vx = hermvander(x, degx)
    vy = hermvander(y, degy)
    v = vx[..., None]*vy[..., None, :]
    return v.reshape(v.shape[:-2] + (-1,))


def hermvander3d(x, y, z, deg) :
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = H_i(x)*H_j(y)*H_k(z),

    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
    indices of `V` index the points `(x, y, z)` and the last index encodes
    the degrees of the Hermite polynomials.

    If ``V = hermvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``hermval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Hermite
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    hermvander, hermvander3d. hermval2d, hermval3d

    Notes
    -----

    .. versionadded::1.7.0

    """
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [1, 1, 1]:
        raise ValueError("degrees must be non-negative integers")
    degx, degy, degz = ideg
    x, y, z = np.array((x, y, z), copy=0) + 0.0

    vx = hermvander(x, degx)
    vy = hermvander(y, degy)
    vz = hermvander(z, degz)
    v = vx[..., None, None]*vy[..., None, :, None]*vz[..., None, None, :]
    return v.reshape(v.shape[:-3] + (-1,))


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Hermite series to data.

    Return the coefficients of a Hermite series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * H_1(x) + ... + c_n * H_n(x),

    where `n` is `deg`.

    Since numpy version 1.7.0, hermfit also supports NA. If any of the
    elements of `x`, `y`, or `w` are NA, then the corresponding rows of the
    linear least squares problem (see Notes) are set to 0. If `y` is 2-D,
    then an NA in any row of `y` invalidates that whole row.

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
    chebfit, legfit, lagfit, polyfit, hermefit
    hermval : Evaluates a Hermite series.
    hermvander : Vandermonde matrix of Hermite series.
    hermweight : Hermite weight function
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Hermite series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Hermite series are probably most useful when the data can be
    approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the Hermite
    weight. In that case the weight ``sqrt(w(x[i])`` should be used
    together with data values ``y[i]/sqrt(w(x[i])``. The weight function is
    available as `hermweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermfit, hermval
    >>> x = np.linspace(-10, 10)
    >>> err = np.random.randn(len(x))/10
    >>> y = hermval(x, [1, 2, 3]) + err
    >>> hermfit(x, y, 2)
    array([ 0.97902637,  1.99849131,  3.00006   ])

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

    # set up the least squares matrices in transposed form
    lhs = hermvander(x, deg).T
    rhs = y.T
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")
        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w
        rhs = rhs * w

    # set rcond
    if rcond is None :
        rcond = len(x)*np.finfo(x.dtype).eps

    # scale the design matrix and solve the least squares equation
    scl = np.sqrt((lhs*lhs).sum(1))
    c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning)

    if full :
        return c, [resids, rank, s, rcond]
    else :
        return c


def hermcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an Hermite basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    Notes
    -----

    .. versionadded::1.7.0

    """
    accprod = np.multiply.accumulate
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array(-.5*c[0]/c[1])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = np.hstack((1., np.sqrt(2.*np.arange(1,n))))
    scl = np.multiply.accumulate(scl)
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[...] = np.sqrt(.5*np.arange(1,n))
    bot[...] = top
    mat[:,-1] -= (c[:-1]/c[-1])*(scl/scl[-1])*.5
    return mat


def hermroots(c):
    """
    Compute the roots of a Hermite series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * H_i(x).

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
    polyroots, legroots, lagroots, chebroots, hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Hermite series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.hermite import hermroots, hermfromroots
    >>> coef = hermfromroots([-1, 0, 1])
    >>> coef
    array([ 0.   ,  0.25 ,  0.   ,  0.125])
    >>> hermroots(coef)
    array([ -1.00000000e+00,  -1.38777878e-17,   1.00000000e+00])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1 :
        return np.array([], dtype=c.dtype)
    if len(c) == 2 :
        return np.array([-.5*c[0]/c[1]])

    m = hermcompanion(c)
    r = la.eigvals(m)
    r.sort()
    return r


def hermgauss(deg):
    """
    Gauss-Hermite quadrature.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-\inf, \inf]`
    with the weight function :math:`f(x) = \exp(-x^2)`.

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

    .. versionadded::1.7.0

    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (H'_n(x_k) * H_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`H_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    ideg = int(deg)
    if ideg != deg or ideg < 1:
        raise ValueError("deg must be a non-negative integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0]*deg + [1])
    m = hermcompanion(c)
    x = la.eigvals(m)
    x.sort()

    # improve roots by one application of Newton
    dy = hermval(x, c)
    df = hermval(x, hermder(c))
    x -= dy/df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = hermval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1/(fm * df)

    # for Hermite we can also symmetrize
    w = (w + w[::-1])/2
    x = (x - x[::-1])/2

    # scale w to get the right value
    w *= np.sqrt(np.pi) / w.sum()

    return x, w


def hermweight(x):
    """
    Weight function of the Hermite polynomials.

    The weight function is :math:`\exp(-x^2)` and the interval of
    integration is :math:`[-\inf, \inf]`. the Hermite polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Notes
    -----

    .. versionadded::1.7.0

    """
    w = np.exp(-x**2)
    return w


#
# Hermite series class
#

# Text between 'REPLACE POLYTEMPLATE' and 'END REPLACE' is inserted by 'polytemplate.py'
#REPLACE POLYTEMPLATE name='Hermite', nick='herm', domain='[-1,1]'

if 1/2 == 0:
    raise ImportError("missing required 'from __future__ import division'")
import numpy as np
from numpy.polynomial import polyutils as pu

class Hermite(pu.PolyBase) :
    """A Hermite series class.

    Hermite instances provide the standard Python numerical methods '+',
    '-', '*', '//', '%', 'divmod', '**', and '()' as well as the listed
    methods.

    Parameters
    ----------
    coef : array_like
        Hermite coefficients, in increasing order.  For example,
        ``(1, 2, 3)`` implies ``P_0 + 2P_1 + 3P_2`` where the
        ``P_i`` are a graded polynomial basis.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped to
        the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1,1].
    window : (2,) array_like, optional
        Window, see ``domain`` for its use. The default value is [-1,1].
        .. versionadded:: 1.6.0

    Attributes
    ----------
    coef : (N,) ndarray
        Hermite coefficients, from low to high.
    domain : (2,) ndarray
        Domain that is mapped to ``window``.
    window : (2,) ndarray
        Window that ``domain`` is mapped to.

    Class Attributes
    ----------------
    maxpower : int
        Maximum power allowed, i.e., the largest number ``n`` such that
        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
    domain : (2,) ndarray
        Default domain of the class.
    window : (2,) ndarray
        Default window of the class.

    Notes
    -----
    It is important to specify the domain in many cases, for instance in
    fitting data, because many of the important properties of the
    polynomial basis only hold in a specified interval and consequently
    the data must be mapped into that interval in order to benefit.

    Examples
    --------

    """
    # Limit runaway size. T_n^m has degree n*2^m
    maxpower = 16
    # Default domain
    domain = np.array([-1,1])
    # Default window
    window = np.array([-1,1])
    # Don't let participate in array operations. Value doesn't matter.
    __array_priority__ = 1000

    def has_samecoef(self, other):
        """Check if coefficients match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``coef`` attribute.

        Returns
        -------
        bool : boolean
            True if the coefficients are the same, False otherwise.

        Notes
        -----
        .. versionadded:: 1.6.0

        """
        if len(self.coef) != len(other.coef):
            return False
        elif not np.all(self.coef == other.coef):
            return False
        else:
            return True

    def has_samedomain(self, other):
        """Check if domains match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``domain`` attribute.

        Returns
        -------
        bool : boolean
            True if the domains are the same, False otherwise.

        Notes
        -----
        .. versionadded:: 1.6.0

        """
        return np.all(self.domain == other.domain)

    def has_samewindow(self, other):
        """Check if windows match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``window`` attribute.

        Returns
        -------
        bool : boolean
            True if the windows are the same, False otherwise.

        Notes
        -----
        .. versionadded:: 1.6.0

        """
        return np.all(self.window == other.window)

    def has_sametype(self, other):
        """Check if types match.

        Parameters
        ----------
        other : object
            Class instance.

        Returns
        -------
        bool : boolean
            True if other is same class as self

        Notes
        -----
        .. versionadded:: 1.7.0

        """
        return isinstance(other, self.__class__)

    def __init__(self, coef, domain=[-1,1], window=[-1,1]) :
        [coef, dom, win] = pu.as_series([coef, domain, window], trim=False)
        if len(dom) != 2 :
            raise ValueError("Domain has wrong number of elements.")
        if len(win) != 2 :
            raise ValueError("Window has wrong number of elements.")
        self.coef = coef
        self.domain = dom
        self.window = win

    def __repr__(self):
        format = "%s(%s, %s, %s)"
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        window = repr(self.window)[6:-1]
        return format % ('Hermite', coef, domain, window)

    def __str__(self) :
        format = "%s(%s)"
        coef = str(self.coef)
        return format % ('herm', coef)

    # Pickle and copy

    def __getstate__(self) :
        ret = self.__dict__.copy()
        ret['coef'] = self.coef.copy()
        ret['domain'] = self.domain.copy()
        ret['window'] = self.window.copy()
        return ret

    def __setstate__(self, dict) :
        self.__dict__ = dict

    # Call

    def __call__(self, arg) :
        off, scl = pu.mapparms(self.domain, self.window)
        arg = off + scl*arg
        return hermval(arg, self.coef)

    def __iter__(self) :
        return iter(self.coef)

    def __len__(self) :
        return len(self.coef)

    # Numeric properties.

    def __neg__(self) :
        return self.__class__(-self.coef, self.domain, self.window)

    def __pos__(self) :
        return self

    def __add__(self, other) :
        """Returns sum"""
        if isinstance(other, pu.PolyBase):
            if not self.has_sametype(other):
                raise TypeError("Polynomial types differ")
            elif not self.has_samedomain(other):
                raise TypeError("Domains differ")
            elif not self.has_samewindow(other):
                raise TypeError("Windows differ")
            else:
                coef = hermadd(self.coef, other.coef)
        else :
            try :
                coef = hermadd(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(coef, self.domain, self.window)

    def __sub__(self, other) :
        """Returns difference"""
        if isinstance(other, pu.PolyBase):
            if not self.has_sametype(other):
                raise TypeError("Polynomial types differ")
            elif not self.has_samedomain(other):
                raise TypeError("Domains differ")
            elif not self.has_samewindow(other):
                raise TypeError("Windows differ")
            else:
                coef = hermsub(self.coef, other.coef)
        else :
            try :
                coef = hermsub(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(coef, self.domain, self.window)

    def __mul__(self, other) :
        """Returns product"""
        if isinstance(other, pu.PolyBase):
            if not self.has_sametype(other):
                raise TypeError("Polynomial types differ")
            elif not self.has_samedomain(other):
                raise TypeError("Domains differ")
            elif not self.has_samewindow(other):
                raise TypeError("Windows differ")
            else:
                coef = hermmul(self.coef, other.coef)
        else :
            try :
                coef = hermmul(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(coef, self.domain, self.window)

    def __div__(self, other):
        # set to __floordiv__,  /, for now.
        return self.__floordiv__(other)

    def __truediv__(self, other) :
        # there is no true divide if the rhs is not a scalar, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        if np.isscalar(other) :
            # this might be overly restrictive
            coef = self.coef/other
            return self.__class__(coef, self.domain, self.window)
        else :
            return NotImplemented

    def __floordiv__(self, other) :
        """Returns the quotient."""
        if isinstance(other, pu.PolyBase):
            if not self.has_sametype(other):
                raise TypeError("Polynomial types differ")
            elif not self.has_samedomain(other):
                raise TypeError("Domains differ")
            elif not self.has_samewindow(other):
                raise TypeError("Windows differ")
            else:
                quo, rem = hermdiv(self.coef, other.coef)
        else :
            try :
                quo, rem = hermdiv(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(quo, self.domain, self.window)

    def __mod__(self, other) :
        """Returns the remainder."""
        if isinstance(other, pu.PolyBase):
            if not self.has_sametype(other):
                raise TypeError("Polynomial types differ")
            elif not self.has_samedomain(other):
                raise TypeError("Domains differ")
            elif not self.has_samewindow(other):
                raise TypeError("Windows differ")
            else:
                quo, rem = hermdiv(self.coef, other.coef)
        else :
            try :
                quo, rem = hermdiv(self.coef, other)
            except :
                return NotImplemented
        return self.__class__(rem, self.domain, self.window)

    def __divmod__(self, other) :
        """Returns quo, remainder"""
        if isinstance(other, self.__class__) :
            if not self.has_samedomain(other):
                raise TypeError("Domains are not equal")
            elif not self.has_samewindow(other):
                raise TypeError("Windows are not equal")
            else:
                quo, rem = hermdiv(self.coef, other.coef)
        else :
            try :
                quo, rem = hermdiv(self.coef, other)
            except :
                return NotImplemented
        quo = self.__class__(quo, self.domain, self.window)
        rem = self.__class__(rem, self.domain, self.window)
        return quo, rem

    def __pow__(self, other) :
        try :
            coef = hermpow(self.coef, other, maxpower = self.maxpower)
        except :
            raise
        return self.__class__(coef, self.domain, self.window)

    def __radd__(self, other) :
        try :
            coef = hermadd(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(coef, self.domain, self.window)

    def __rsub__(self, other):
        try :
            coef = hermsub(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(coef, self.domain, self.window)

    def __rmul__(self, other) :
        try :
            coef = hermmul(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(coef, self.domain, self.window)

    def __rdiv__(self, other):
        # set to __floordiv__ /.
        return self.__rfloordiv__(other)

    def __rtruediv__(self, other) :
        # there is no true divide if the rhs is not a scalar, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        if len(self.coef) == 1 :
            try :
                quo, rem = hermdiv(other, self.coef[0])
            except :
                return NotImplemented
        return self.__class__(quo, self.domain, self.window)

    def __rfloordiv__(self, other) :
        try :
            quo, rem = hermdiv(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(quo, self.domain, self.window)

    def __rmod__(self, other) :
        try :
            quo, rem = hermdiv(other, self.coef)
        except :
            return NotImplemented
        return self.__class__(rem, self.domain, self.window)

    def __rdivmod__(self, other) :
        try :
            quo, rem = hermdiv(other, self.coef)
        except :
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window)
        rem = self.__class__(rem, self.domain, self.window)
        return quo, rem

    # Enhance me
    # some augmented arithmetic operations could be added here

    def __eq__(self, other) :
        res = isinstance(other, self.__class__)                 and self.has_samecoef(other)                 and self.has_samedomain(other)                 and self.has_samewindow(other)
        return res

    def __ne__(self, other) :
        return not self.__eq__(other)

    #
    # Extra methods.
    #

    def copy(self) :
        """Return a copy.

        Return a copy of the current Hermite instance.

        Returns
        -------
        new_instance : Hermite
            Copy of current instance.

        """
        return self.__class__(self.coef, self.domain, self.window)

    def degree(self) :
        """The degree of the series.

        Notes
        -----
        .. versionadded:: 1.5.0

        """
        return len(self) - 1

    def cutdeg(self, deg) :
        """Truncate series to the given degree.

        Reduce the degree of the Hermite series to `deg` by discarding the
        high order terms. If `deg` is greater than the current degree a
        copy of the current series is returned. This can be useful in least
        squares where the coefficients of the high degree terms may be very
        small.

        Parameters
        ----------
        deg : non-negative int
            The series is reduced to degree `deg` by discarding the high
            order terms. The value of `deg` must be a non-negative integer.

        Returns
        -------
        new_instance : Hermite
            New instance of Hermite with reduced degree.

        Notes
        -----
        .. versionadded:: 1.5.0

        """
        return self.truncate(deg + 1)

    def trim(self, tol=0) :
        """Remove small leading coefficients

        Remove leading coefficients until a coefficient is reached whose
        absolute value greater than `tol` or the beginning of the series is
        reached. If all the coefficients would be removed the series is set to
        ``[0]``. A new Hermite instance is returned with the new coefficients.
        The current instance remains unchanged.

        Parameters
        ----------
        tol : non-negative number.
            All trailing coefficients less than `tol` will be removed.

        Returns
        -------
        new_instance : Hermite
            Contains the new set of coefficients.

        """
        coef = pu.trimcoef(self.coef, tol)
        return self.__class__(coef, self.domain, self.window)

    def truncate(self, size) :
        """Truncate series to length `size`.

        Reduce the Hermite series to length `size` by discarding the high
        degree terms. The value of `size` must be a positive integer. This
        can be useful in least squares where the coefficients of the
        high degree terms may be very small.

        Parameters
        ----------
        size : positive int
            The series is reduced to length `size` by discarding the high
            degree terms. The value of `size` must be a positive integer.

        Returns
        -------
        new_instance : Hermite
            New instance of Hermite with truncated coefficients.

        """
        isize = int(size)
        if isize != size or isize < 1 :
            raise ValueError("size must be a positive integer")
        if isize >= len(self.coef) :
            coef = self.coef
        else :
            coef = self.coef[:isize]
        return self.__class__(coef, self.domain, self.window)

    def convert(self, domain=None, kind=None, window=None) :
        """Convert to different class and/or domain.

        Parameters
        ----------
        domain : array_like, optional
            The domain of the converted series. If the value is None,
            the default domain of `kind` is used.
        kind : class, optional
            The polynomial series type class to which the current instance
            should be converted. If kind is None, then the class of the
            current instance is used.
        window : array_like, optional
            The window of the converted series. If the value is None,
            the default window of `kind` is used.

        Returns
        -------
        new_series_instance : `kind`
            The returned class can be of different type than the current
            instance and/or have a different domain.

        Notes
        -----
        Conversion between domains and class types can result in
        numerically ill defined series.

        Examples
        --------

        """
        if kind is None:
            kind = Hermite
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        return self(kind.identity(domain, window=window))

    def mapparms(self) :
        """Return the mapping parameters.

        The returned values define a linear map ``off + scl*x`` that is
        applied to the input arguments before the series is evaluated. The
        map depends on the ``domain`` and ``window``; if the current
        ``domain`` is equal to the ``window`` the resulting map is the
        identity.  If the coefficients of the ``Hermite`` instance are to be
        used by themselves outside this class, then the linear function
        must be substituted for the ``x`` in the standard representation of
        the base polynomials.

        Returns
        -------
        off, scl : floats or complex
            The mapping function is defined by ``off + scl*x``.

        Notes
        -----
        If the current domain is the interval ``[l_1, r_1]`` and the window
        is ``[l_2, r_2]``, then the linear mapping function ``L`` is
        defined by the equations::

            L(l_1) = l_2
            L(r_1) = r_2

        """
        return pu.mapparms(self.domain, self.window)

    def integ(self, m=1, k=[], lbnd=None) :
        """Integrate.

        Return an instance of Hermite that is the definite integral of the
        current series. Refer to `hermint` for full documentation.

        Parameters
        ----------
        m : non-negative int
            The number of integrations to perform.
        k : array_like
            Integration constants. The first constant is applied to the
            first integration, the second to the second, and so on. The
            list of values must less than or equal to `m` in length and any
            missing values are set to zero.
        lbnd : Scalar
            The lower bound of the definite integral.

        Returns
        -------
        integral : Hermite
            The integral of the series using the same domain.

        See Also
        --------
        hermint : similar function.
        hermder : similar function for derivative.

        """
        off, scl = self.mapparms()
        if lbnd is None :
            lbnd = 0
        else :
            lbnd = off + scl*lbnd
        coef = hermint(self.coef, m, k, lbnd, 1./scl)
        return self.__class__(coef, self.domain, self.window)

    def deriv(self, m=1):
        """Differentiate.

        Return an instance of Hermite that is the derivative of the current
        series.  Refer to `hermder` for full documentation.

        Parameters
        ----------
        m : non-negative int
            The number of integrations to perform.

        Returns
        -------
        derivative : Hermite
            The derivative of the series using the same domain.

        See Also
        --------
        hermder : similar function.
        hermint : similar function for integration.

        """
        off, scl = self.mapparms()
        coef = hermder(self.coef, m, scl)
        return self.__class__(coef, self.domain, self.window)

    def roots(self) :
        """Return list of roots.

        Return ndarray of roots for this series. See `hermroots` for
        full documentation. Note that the accuracy of the roots is likely to
        decrease the further outside the domain they lie.

        See Also
        --------
        hermroots : similar function
        hermfromroots : function to go generate series from roots.

        """
        roots = hermroots(self.coef)
        return pu.mapdomain(roots, self.window, self.domain)

    def linspace(self, n=100, domain=None):
        """Return x,y values at equally spaced points in domain.

        Returns x, y values at `n` linearly spaced points across domain.
        Here y is the value of the polynomial at the points x. By default
        the domain is the same as that of the Hermite instance.  This method
        is intended mostly as a plotting aid.

        Parameters
        ----------
        n : int, optional
            Number of point pairs to return. The default value is 100.
        domain : {None, array_like}
            If not None, the specified domain is used instead of that of
            the calling instance. It should be of the form ``[beg,end]``.
            The default is None.

        Returns
        -------
        x, y : ndarrays
            ``x`` is equal to linspace(self.domain[0], self.domain[1], n)
            ``y`` is the polynomial evaluated at ``x``.

        .. versionadded:: 1.5.0

        """
        if domain is None:
            domain = self.domain
        x = np.linspace(domain[0], domain[1], n)
        y = self(x)
        return x, y



    @staticmethod
    def fit(x, y, deg, domain=None, rcond=None, full=False, w=None,
        window=[-1,1]):
        """Least squares fit to data.

        Return a `Hermite` instance that is the least squares fit to the data
        `y` sampled at `x`. Unlike `hermfit`, the domain of the returned
        instance can be specified and this will often result in a superior
        fit with less chance of ill conditioning. Support for NA was added
        in version 1.7.0. See `hermfit` for full documentation of the
        implementation.

        Parameters
        ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points ``(x[i], y[i])``.
        y : array_like, shape (M,) or (M, K)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per column.
        deg : int
            Degree of the fitting polynomial.
        domain : {None, [beg, end], []}, optional
            Domain to use for the returned Hermite instance. If ``None``,
            then a minimal domain that covers the points `x` is chosen.  If
            ``[]`` the default domain ``[-1,1]`` is used. The default
            value is [-1,1] in numpy 1.4.x and ``None`` in later versions.
            The ``'[]`` value was added in numpy 1.5.0.
        rcond : float, optional
            Relative condition number of the fit. Singular values smaller
            than this relative to the largest singular value will be
            ignored. The default value is len(x)*eps, where eps is the
            relative precision of the float type, about 2e-16 in most
            cases.
        full : bool, optional
            Switch determining nature of return value. When it is False
            (the default) just the coefficients are returned, when True
            diagnostic information from the singular value decomposition is
            also returned.
        w : array_like, shape (M,), optional
            Weights. If not None the contribution of each point
            ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
            weights are chosen so that the errors of the products
            ``w[i]*y[i]`` all have the same variance.  The default value is
            None.
            .. versionadded:: 1.5.0
        window : {[beg, end]}, optional
            Window to use for the returned Hermite instance. The default
            value is ``[-1,1]``
            .. versionadded:: 1.6.0

        Returns
        -------
        least_squares_fit : instance of Hermite
            The Hermite instance is the least squares fit to the data and
            has the domain specified in the call.

        [residuals, rank, singular_values, rcond] : only if `full` = True
            Residuals of the least squares fit, the effective rank of the
            scaled Vandermonde matrix and its singular values, and the
            specified value of `rcond`. For more details, see
            `linalg.lstsq`.

        See Also
        --------
        hermfit : similar function

        """
        if domain is None:
            domain = pu.getdomain(x)
        elif domain == []:
            domain = [-1,1]

        if window == []:
            window = [-1,1]

        xnew = pu.mapdomain(x, domain, window)
        res = hermfit(xnew, y, deg, w=w, rcond=rcond, full=full)
        if full :
            [coef, status] = res
            return Hermite(coef, domain=domain, window=window), status
        else :
            coef = res
            return Hermite(coef, domain=domain, window=window)

    @staticmethod
    def fromroots(roots, domain=[-1,1], window=[-1,1]) :
        """Return Hermite instance with specified roots.

        Returns an instance of Hermite representing the product
        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is the
        list of roots.

        Parameters
        ----------
        roots : array_like
            List of roots.

        Returns
        -------
        object : Hermite instance
            Series with the specified roots.

        See Also
        --------
        hermfromroots : equivalent function

        """
        if domain is None :
            domain = pu.getdomain(roots)
        rnew = pu.mapdomain(roots, domain, window)
        coef = hermfromroots(rnew)
        return Hermite(coef, domain=domain, window=window)

    @staticmethod
    def identity(domain=[-1,1], window=[-1,1]) :
        """Identity function.

        If ``p`` is the returned Hermite object, then ``p(x) == x`` for all
        values of x.

        Parameters
        ----------
        domain : array_like
            The resulting array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain.
        window : array_like
            The resulting array must be if the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the window.

        Returns
        -------
        identity : Hermite instance

        """
        off, scl = pu.mapparms(window, domain)
        coef = hermline(off, scl)
        return Hermite(coef, domain, window)

    @staticmethod
    def basis(deg, domain=[-1,1], window=[-1,1]):
        """Hermite polynomial of degree `deg`.

        Returns an instance of the Hermite polynomial of degree `d`.

        Parameters
        ----------
        deg : int
            Degree of the Hermite polynomial. Must be >= 0.
        domain : array_like
            The resulting array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain.
        window : array_like
            The resulting array must be if the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the window.

        Returns
        p : Hermite instance

        Notes
        -----
        .. versionadded:: 1.7.0

        """
        ideg = int(deg)
        if ideg != deg or ideg < 0:
            raise ValueError("deg must be non-negative integer")
        return Hermite([0]*ideg + [1], domain, window)

    @staticmethod
    def cast(series, domain=[-1,1], window=[-1,1]):
        """Convert instance to equivalent Hermite series.

        The `series` is expected to be an instance of some polynomial
        series of one of the types supported by by the numpy.polynomial
        module, but could be some other class that supports the convert
        method.

        Parameters
        ----------
        series : series
            The instance series to be converted.
        domain : array_like
            The resulting array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain.
        window : array_like
            The resulting array must be if the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the window.

        Returns
        p : Hermite instance
            A Hermite series equal to the `poly` series.

        See Also
        --------
        convert -- similar instance method

        Notes
        -----
        .. versionadded:: 1.7.0

        """
        return series.convert(domain, Hermite, window)

#END REPLACE
