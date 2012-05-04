Polynomial Module (:mod:`numpy.polynomial.polynomial`)
======================================================

.. versionadded:: 1.4.0

.. currentmodule:: numpy.polynomial.polynomial

This module provides a number of objects (mostly functions) useful for
dealing with Polynomial series, including a `Polynomial` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Polynomial Class
----------------

.. autosummary::
   :toctree: generated/

   Polynomial

Basics
------

.. autosummary::
   :toctree: generated/

   polyval
   polyval2d
   polyval3d
   polygrid2d
   polygrid3d
   polyroots
   polyfromroots

Fitting
-------

.. autosummary::
   :toctree: generated/

   polyfit
   polyvander
   polyvander2d
   polyvander3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   polyder
   polyint

Algebra
-------

.. autosummary::
   :toctree: generated/

   polyadd
   polysub
   polymul
   polymulx
   polydiv
   polypow

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   polycompanion
   polydomain
   polyzero
   polyone
   polyx
   polytrim
   polyline
