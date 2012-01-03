Hermite Module, "Physicists'" (:mod:`numpy.polynomial.hermite`)
===============================================================

.. versionadded:: 1.6.0

.. currentmodule:: numpy.polynomial.hermite

This module provides a number of objects (mostly functions) useful for
dealing with Hermite series, including a `Hermite` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Hermite Class
-------------

.. autosummary::
   :toctree: generated/

   Hermite

Basics
------

.. autosummary::
   :toctree: generated/

   hermval
   hermval2d
   hermval3d
   hermgrid2d
   hermgrid3d
   hermroots
   hermfromroots

Fitting
-------

.. autosummary::
   :toctree: generated/

   hermfit
   hermvander
   hermvander2d
   hermvander3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   hermder
   hermint

Algebra
-------

.. autosummary::
   :toctree: generated/

   hermadd
   hermsub
   hermmul
   hermmulx
   hermdiv
   hermpow

Quadrature
----------

.. autosummary::
   :toctree: generated/

   hermgauss
   hermweight

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   hermcompanion
   hermdomain
   hermzero
   hermone
   hermx
   hermtrim
   hermline
   herm2poly
   poly2herm
