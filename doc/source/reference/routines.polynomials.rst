Polynomials
***********

The polynomial package is newer and more complete than poly1d and the
convenience classes are better behaved in the numpy environment. When
backwards compatibility is not an issue it should be the package of choice.
Note that the  various routines in the polynomial package all deal with
series whose coefficients go from degree zero upward, which is the reverse
of the poly1d convention. The easy way to remember this is that indexes
correspond to degree, i.e., coef[i] is the coefficient of the term of
degree i.


.. toctree::
   :maxdepth: 2

   routines.polynomials.poly1d


.. toctree::
   :maxdepth: 3

   routines.polynomials.package
