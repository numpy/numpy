Chebyshev Module (:mod:`numpy.polynomial.chebyshev`)
====================================================

.. versionadded:: 1.4.0

.. currentmodule:: numpy.polynomial.chebyshev

This module provides a number of objects (mostly functions) useful for
dealing with Chebyshev series, including a `Chebyshev` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Chebyshev Class
---------------

.. autosummary::
   :toctree: generated/

   Chebyshev

Basics
------

.. autosummary::
   :toctree: generated/

   chebval
   chebval2d
   chebval3d
   chebgrid2d
   chebgrid3d
   chebroots
   chebfromroots

Fitting
-------

.. autosummary::
   :toctree: generated/

   chebfit
   chebvander
   chebvander2d
   chebvander3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   chebder
   chebint

Algebra
-------

.. autosummary::
   :toctree: generated/

   chebadd
   chebsub
   chebmul
   chebmulx
   chebdiv
   chebpow

Quadrature
----------

.. autosummary::
   :toctree: generated/

   chebgauss
   chebweight

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   chebcompanion
   chebdomain
   chebzero
   chebone
   chebx
   chebtrim
   chebline
   cheb2poly
   poly2cheb
