.. _routines.array-creation:

Array creation routines
=======================

.. seealso:: :ref:`Array creation <arrays.creation>`

.. currentmodule:: numpy

From shape or value
-------------------
.. autosummary::
   :toctree: generated/

   empty
   empty_like
   eye
   identity
   ones
   ones_like
   zeros
   zeros_like
   full
   full_like

From existing data
------------------
.. autosummary::
   :toctree: generated/

   array
   asarray
   asanyarray
   ascontiguousarray
   asmatrix
   astype
   copy
   frombuffer
   from_dlpack
   fromfile
   fromfunction
   fromiter
   fromstring
   loadtxt

.. _routines.array-creation.rec:

Creating record arrays
----------------------

.. note:: Please refer to :ref:`arrays.classes.rec` for
   record arrays.

.. autosummary::
   :toctree: generated/

   rec.array
   rec.fromarrays
   rec.fromrecords
   rec.fromstring
   rec.fromfile

.. _routines.array-creation.char:

Creating character arrays (:mod:`numpy.char`)
---------------------------------------------

.. note:: :mod:`numpy.char` is used to create character
   arrays.

.. autosummary::
   :toctree: generated/

   char.array
   char.asarray

Numerical ranges
----------------
.. autosummary::
   :toctree: generated/

   arange
   linspace
   logspace
   geomspace
   meshgrid
   mgrid
   ogrid

Building matrices
-----------------
.. autosummary::
   :toctree: generated/

   diag
   diagflat
   tri
   tril
   triu
   vander

The matrix class
----------------
.. autosummary::
   :toctree: generated/

   bmat
