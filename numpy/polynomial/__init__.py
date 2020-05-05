"""
A sub-package for efficiently dealing with polynomials.

Within the documentation for this sub-package, a "finite power series,"
i.e., a polynomial (also referred to simply as a "series") is represented
by a 1-D numpy array of the polynomial's coefficients, ordered from lowest
order term to highest.  For example, array([1,2,3]) represents
``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
applicable to the specific module in question, e.g., `polynomial` (which
"wraps" the "standard" basis) or `chebyshev`.  For optimal performance,
all operations on polynomials, including evaluation at an argument, are
implemented as operations on the coefficients.  Additional (module-specific)
information can be found in the docstring for the module of interest.

This package provides *convenience classes* for each of six different kinds
of polynomials::

         ============    ================
         Name            Provides
         ============    ================
         Polynomial      Power series
         Chebyshev       Chebyshev series
         Legendre        Legendre series
         Laguerre        Laguerre series
         Hermite         Hermite series
         HermiteE        HermiteE series
         ============    ================

These *convenience classes* provide a consistent interface for creating,
manipulating, and fitting data with polynomials of different bases, and are the
preferred way for interacting with polynomials. The convenience classes are
available from `numpy.polynomial`, eliminating the need to navigate to the
corresponding submodules, e.g. ``np.polynomial.Polynomial``
or ``np.polynomial.Chebyshev`` instead of 
``np.polynomial.polynomial.Polynomial`` or 
``np.polynomial.chebyshev.Chebyshev``, respectively.
It is strongly recommended that the class-based interface is used instead of
functions from individual submodules for the sake of consistency and brevity.
For example, to fit a Chebyshev polynomial with degree ``1`` to data given
by arrays ``xdata`` and ``ydata``, the ``fit`` class method::

    >>> from numpy.polynomial import Chebyshev
    >>> c = Chebyshev.fit(xdata, ydata, deg=1)

is preferred over the ``chebfit`` function from the 
`numpy.polynomial.chebyshev` module::

    >>> from numpy.polynomial.chebyshev import chebfit
    >>> c = chebfit(xdata, ydata, deg=1)

See `routines.polynomials.classes` for a more detailed introduction to the
polynomial convenience classes.

"""
from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
