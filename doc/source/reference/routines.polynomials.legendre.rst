Legendre Module (:mod:`numpy.polynomial.legendre`)
==================================================

.. versionadded:: 1.6.0

.. currentmodule:: numpy.polynomial.legendre

This module provides a number of objects (mostly functions) useful for
dealing with Legendre series, including a `Legendre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Legendre Class
--------------

.. autosummary::
   :toctree: generated/

   Legendre

Basics
------

.. autosummary::
   :toctree: generated/

   legval
   legval2d
   legval3d
   leggrid2d
   leggrid3d
   legroots
   legfromroots

Fitting
-------

.. autosummary::
   :toctree: generated/

   legfit
   legvander
   legvander2d
   legvander3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   legder
   legint

Algebra
-------

.. autosummary::
   :toctree: generated/

   legadd
   legsub
   legmul
   legmulx
   legdiv
   legpow

Quadrature
----------

.. autosummary::
   :toctree: generated/

   leggauss
   legweight

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   legcompanion
   legdomain
   legzero
   legone
   legx
   legtrim
   legline
   leg2poly
   poly2leg
