.. _routines.polynomial:

Polynomials
***********

Polynomials in NumPy can be *created*, *manipulated*, and even *fitted* using
the :doc:`convenience classes <routines.polynomials.classes>`
of the `numpy.polynomial` package, introduced in NumPy 1.4.

Prior to NumPy 1.4, `numpy.poly1d` was the class of choice and it is still
available in order to maintain backward compatibility.
However, the newer `polynomial package <numpy.polynomial>` is more complete
and its `convenience classes <routines.polynomials.classes>` provide a
more consistent, better-behaved interface for working with polynomial
expressions.
Therefore :mod:`numpy.polynomial` is recommended for new coding.

Transitioning from `numpy.poly1d` to `numpy.polynomial`
-------------------------------------------------------

As noted above, the :class:`poly1d class <numpy.poly1d>` and associated
functions defined in ``numpy.lib.polynomial``, such as `numpy.polyfit`
and `numpy.poly`, are considered legacy and should **not** be used in new
code.
Since NumPy version 1.4, the `numpy.polynomial` package is preferred for
working with polynomials.

Quick Reference
~~~~~~~~~~~~~~~

The following table highlights some of the main differences between the
legacy polynomial module and the polynomial package for common tasks.
The `~numpy.polynomial.Polynomial` class is imported for brevity::

    from numpy.polynomial import Polynomial


+------------------------+------------------------------+---------------------------------------+
|  How to...             | Legacy (`numpy.poly1d`)      | `numpy.polynomial`                    |
+------------------------+------------------------------+---------------------------------------+
| Create a               | ``p = np.poly1d([1, 2, 3])`` | ``p = Polynomial([3, 2, 1])``         |
| polynomial object      |                              |                                       |
| from coefficients [1]_ |                              |                                       |
+------------------------+------------------------------+---------------------------------------+
| Create a polynomial    | ``r = np.poly([-1, 1])``     | ``p = Polynomial.fromroots([-1, 1])`` |
| object from roots      | ``p = np.poly1d(r)``         |                                       |
+------------------------+------------------------------+---------------------------------------+
| Fit a polynomial of    |                              |                                       |
| degree ``deg`` to data | ``np.polyfit(x, y, deg)``    | ``Polynomial.fit(x, y, deg)``         |
+------------------------+------------------------------+---------------------------------------+


.. [1] Note the reversed ordering of the coefficients

Transition Guide
~~~~~~~~~~~~~~~~

There are significant differences between ``numpy.lib.polynomial`` and
the the polynomial package, `numpy.polynomial`.
The most significant differences is the ordering of the coefficients for the
polynomial expressions.
The  various routines in `numpy.polynomial` all
deal with series whose coefficients go from degree zero upward,
which is the *reverse order* of the poly1d convention.
The easy way to remember this is that indices
correspond to degree, i.e., ``coef[i]`` is the coefficient of the term of
degree *i*.

Though the difference in convention may be confusing, it is straightforward to
convert from the old-style polynomial API to the new.
For example, the following demonstrates how you would convert a `numpy.poly1d`
instance representing the expression :math:`x^{2} + 2x + 3` to a
`~numpy.polynomial.polynomial.Polynomial` instance representing the same
expression::

    >>> c = [1, 2, 3]                          # x**2 +2*x + 3
    >>> p1d = np.poly1d(c)                     # Legacy
    >>> p = np.polynomial.Polynomial(c[::-1])

In addition to the ``coef`` attribute, polynomials from the polynomial
package also have ``domain`` and ``window`` attributes that define any
particular polynomial instance. These attributes are most relevant when
polynomials to data, though it should be noted that polynomials with
different ``domain`` and ``window`` attributes are not considered equal, and
can't be mixed in arithmetic::

    >>> p1 = np.polynomial.Polynomial([1, 2, 3])
    >>> p1
    Polynomial([1., 2., 3.], domain=[-1,  1], window=[-1,  1])
    >>> p2 = np.polynomial.Polynomial([1, 2, 3], domain=[-2, 2])
    >>> p1 == p2
    False
    >>> p1 + p2
    TypeError: Domains differ

See the documentation for the
`convenience classes <routines.polynomials.classes>` for further details on
the ``domain`` and ``window`` attributes.

Another major difference bewteen the legacy polynomial module and the
polynomial package is polynomial fitting. In the legacy code, fitting was
done via the `~numpy.polyfit` function. In the polynomial package, the
`Polynomial.fit` classmethod is preferred. For example, consider a simple
linear fit to the following data:

.. ipython::

    rg = np.random.default_rng()
    x = np.arange(10)
    y = np.arange(10) + rg.standard_normal(10)

With the legacy polynomial module, a linear fit (i.e. polynomial of degree 1)
could be applied to these data with `~numpy.polyfit`:

.. ipython::

    np.polyfit(x, y, deg=1)

With the new polynomial API, the `~numpy.polynomial.polynomial.Polynomial.fit`
class method is preferred:

.. ipython::

    p_fitted = np.polynomial.Polynomial.fit(x, y, deg=1)
    p_fitted

Note that the coefficients are given *in scaled domain* defined by the linear
mapping between the ``window`` and ``domain``.
`~numpy.polynomial.polynomial.Polynomial.convert` can be used to get the
coefficients in the unscaled data domain.

.. ipython::

    p_fitted.convert()

Documentation for the `~numpy.polynomial` Package
-------------------------------------------------

In addition to standard power series polynomials, the polynomial package
provides several additional kinds of polynomials including Chebyshev,
Hermite (two subtypes), Laguerre, and Legendre polynomials.
Each of these has an associated
`convenience class <routines.polynomials.classes>` available from the
`numpy.polynomial` namespace that provides a consistent interface for working
with polynomials regardless of their type.

.. toctree::
   :maxdepth: 1

   routines.polynomials.classes

Documentation pertaining to specific functions defined for each kind of the
kinds of polynomials can be found in the corresponding module documentation:

.. toctree::
   :maxdepth: 1

   routines.polynomials.polynomial
   routines.polynomials.chebyshev
   routines.polynomials.hermite
   routines.polynomials.hermite_e
   routines.polynomials.laguerre
   routines.polynomials.legendre
   routines.polynomials.polyutils


Documentation for Legacy Polynomials
------------------------------------

.. toctree::
   :maxdepth: 2

   routines.polynomials.poly1d
