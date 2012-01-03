HermiteE Module, "Probabilists'" (:mod:`numpy.polynomial.hermite_e`)
====================================================================

.. versionadded:: 1.6.0

.. currentmodule:: numpy.polynomial.hermite_e

This module provides a number of objects (mostly functions) useful for
dealing with HermiteE series, including a `HermiteE` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

HermiteE Class
--------------

.. autosummary::
   :toctree: generated/

   HermiteE

Basics
------

.. autosummary::
   :toctree: generated/

   hermeval
   hermeval2d
   hermeval3d
   hermegrid2d
   hermegrid3d
   hermeroots
   hermefromroots

Fitting
-------

.. autosummary::
   :toctree: generated/

   hermefit
   hermevander
   hermevander2d
   hermevander3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   hermeder
   hermeint

Algebra
-------

.. autosummary::
   :toctree: generated/

   hermeadd
   hermesub
   hermemul
   hermemulx
   hermediv
   hermepow

Quadrature
----------

.. autosummary::
   :toctree: generated/

   hermegauss
   hermeweight

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   hermecompanion
   hermedomain
   hermezero
   hermeone
   hermex
   hermetrim
   hermeline
   herme2poly
   poly2herme
