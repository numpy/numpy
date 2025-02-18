.. _routines.char:

Legacy fixed-width string functionality
=======================================

.. currentmodule:: numpy.char

.. module:: numpy.char

.. legacy::

   The string operations in this module, as well as the `numpy.char.chararray`
   class, are planned to be deprecated in the future. Use `numpy.strings`
   instead.

The `numpy.char` module provides a set of vectorized string
operations for arrays of type `numpy.str_` or `numpy.bytes_`. For example

.. try_examples::

   >>> import numpy as np
   >>> np.char.capitalize(["python", "numpy"])
   array(['Python', 'Numpy'], dtype='<U6')
   >>> np.char.add(["num", "doc"], ["py", "umentation"])
   array(['numpy', 'documentation'], dtype='<U13')

The methods in this module are based on the methods in :py:mod:`string`

String operations
-----------------

.. autosummary::
   :toctree: generated/

   add
   multiply
   mod
   capitalize
   center
   decode
   encode
   expandtabs
   join
   ljust
   lower
   lstrip
   partition
   replace
   rjust
   rpartition
   rsplit
   rstrip
   split
   splitlines
   strip
   swapcase
   title
   translate
   upper
   zfill

Comparison
----------

Unlike the standard numpy comparison operators, the ones in the `char`
module strip trailing whitespace characters before performing the
comparison.

.. autosummary::
   :toctree: generated/

   equal
   not_equal
   greater_equal
   less_equal
   greater
   less
   compare_chararrays

String information
------------------

.. autosummary::
   :toctree: generated/

   count
   endswith
   find
   index
   isalpha
   isalnum
   isdecimal
   isdigit
   islower
   isnumeric
   isspace
   istitle
   isupper
   rfind
   rindex
   startswith
   str_len

Convenience class
-----------------

.. autosummary::
   :toctree: generated/

   array
   asarray
   chararray
