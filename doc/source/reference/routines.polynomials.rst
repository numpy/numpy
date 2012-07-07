Polynomials
***********

Polynomials in NumPy can be *created*, *manipulated*, and even *fitted* using
the :doc:`routines.polynomials.classes`
of the `numpy.polynomial` package, introduced in NumPy 1.4.

Prior to NumPy 1.4, `numpy.poly1d` was the class of choice and it is still
available in order to maintain backward compatibility.
However, the newer Polynomial package is more complete than `numpy.poly1d`
and its convenience classes are better behaved in the numpy environment.
Therefore Polynomial is recommended for new coding.

Transition notice
-----------------
The  various routines in the Polynomial package all deal with
series whose coefficients go from degree zero upward,
which is the *reverse order* of the Poly1d convention.
The easy way to remember this is that indexes
correspond to degree, i.e., coef[i] is the coefficient of the term of
degree i.


.. toctree::
   :maxdepth: 2

   routines.polynomials.package

.. toctree::
   :maxdepth: 2

   routines.polynomials.poly1d
