Using the Convenience Classes
=============================

The classes in the polynomial package can be imported directly from
numpy.polynomial as well as from the corresponding modules.::

    >>> from numpy.polynomial import Polynomial as P
    >>> p = P([0, 0, 1])
    >>> p**2
    Polynomial([ 0.,  0.,  0.,  0.,  1.], [-1.,  1.], [-1.,  1.])

Because most of the functionality in the modules of the polynomial package
is available through the corresponding classes, including fitting, shifting
and scaling, and conversion between classes, it should not be necessary to
use the functions in the modules except for multi-dimensional work.
