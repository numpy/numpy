.. _routines.polynomial:

Polynomials
***********

Polynomials in NumPy can be *created*, *manipulated*, and even *fitted* using
the :doc:`convenience classes <routines.polynomials.classes>`
of the `numpy.polynomial` package, introduced in NumPy 1.4.

Prior to NumPy 1.4, `numpy.poly1d` was the class of choice and it is still
available in order to maintain backward compatibility.
However, the newer `polynomial package <numpy.polynomial>` is more complete
and its `convenience classes <routines.polynomials.classes` provide a
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
legacy polynomial module and the polynomial package.
The `~numpy.polynomial.Polynomial` class is imported for brevity::

    from numpy.polynomial import Polynomial


+---------------+------------------------------+----------------------------------+
|  How to...    | Legacy (`numpy.poly1d`)      | `numpy.polynomial`               |
+---------------+------------------------------+----------------------------------+
| Create a      | ``p = np.poly1d([1, 2, 3])`` | ``p = Polynomial([3, 2, 1])``    |
| polynomial    |                              |                                  |
| object [1]_   |                              |                                  |
+---------------+------------------------------+----------------------------------+
| Fit data with |                              |                                  |
| a polynomial  | ``np.polyfit(x, y, degree)`` | ``Polynomial.fit(x, y, degree)`` |
| expression    |                              |                                  |
+---------------+------------------------------+----------------------------------+


.. [1] Note the reversed ordering of the coefficients


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
instance representing the expression :math:`x^{2} + 2 \cdot x + 3` to a
`numpy.polynomial.Polynomial` instance representing the same expression::

    >>> p1d = np.poly1d([1, 2, 3])                    # Legacy
    >>> p = np.polynomial.Polynomial(p1d.coef[::-1])  # Preferred



.. toctree::
   :maxdepth: 1

   routines.polynomials.classes
   routines.polynomials.polynomial
   routines.polynomials.chebyshev
   routines.polynomials.hermite
   routines.polynomials.hermite_e
   routines.polynomials.laguerre
   routines.polynomials.legendre
   routines.polynomials.polyutils


.. toctree::
   :maxdepth: 2

   routines.polynomials.poly1d
