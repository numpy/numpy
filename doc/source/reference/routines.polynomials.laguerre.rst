Laguerre Module (:mod:`numpy.polynomial.laguerre`)
==================================================

.. versionadded:: 1.6.0

.. currentmodule:: numpy.polynomial.laguerre

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Laguerre Class
--------------

.. autosummary::
   :toctree: generated/

   Laguerre

Basics
------

.. autosummary::
   :toctree: generated/

   lagval
   lagval2d
   lagval3d
   laggrid2d
   laggrid3d
   lagroots
   lagfromroots

Fitting
-------

.. autosummary::
   :toctree: generated/

   lagfit
   lagvander
   lagvander2d
   lagvander3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   lagder
   lagint

Algebra
-------

.. autosummary::
   :toctree: generated/

   lagadd
   lagsub
   lagmul
   lagmulx
   lagdiv
   lagpow

Quadrature
----------

.. autosummary::
   :toctree: generated/

   laggauss
   lagweight

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   lagcompanion
   lagdomain
   lagzero
   lagone
   lagx
   lagtrim
   lagline
   lag2poly
   poly2lag
