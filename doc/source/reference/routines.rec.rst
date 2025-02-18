.. _routines.rec:

Record Arrays (:mod:`numpy.rec`)
================================

.. currentmodule:: numpy.rec

.. module:: numpy.rec

Record arrays expose the fields of structured arrays as properties.

Most commonly, ndarrays contain elements of a single type, e.g. floats,
integers, bools etc.  However, it is possible for elements to be combinations
of these using structured types, such as:

.. try_examples::

  >>> import numpy as np
  >>> a = np.array([(1, 2.0), (1, 2.0)],
  ...     dtype=[('x', np.int64), ('y', np.float64)])
  >>> a
  array([(1, 2.), (1, 2.)], dtype=[('x', '<i8'), ('y', '<f8')])

  Here, each element consists of two fields: x (and int), and y (a float).
  This is known as a structured array.  The different fields are analogous
  to columns in a spread-sheet.  The different fields can be accessed as
  one would a dictionary:

  >>> a['x']
  array([1, 1])

  >>> a['y']
  array([2., 2.])

  Record arrays allow us to access fields as properties:

  >>> ar = np.rec.array(a)
  >>> ar.x
  array([1, 1])
  >>> ar.y
  array([2., 2.])

Functions
---------

.. autosummary::
   :toctree: generated/

   array
   find_duplicate
   format_parser
   fromarrays
   fromfile
   fromrecords
   fromstring

Also, the `numpy.recarray` class and the `numpy.record` scalar dtype are present
in this namespace.
