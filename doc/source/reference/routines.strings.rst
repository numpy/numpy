.. _routines.strings:

String functionality
====================

.. currentmodule:: numpy.strings

.. module:: numpy.strings

The `numpy.strings` module provides a set of universal functions operating
on arrays of type `numpy.str_` or `numpy.bytes_`.
For example,

.. try_examples::

   >>> np.strings.add(["num", "doc"], ["py", "umentation"])
   array(['numpy', 'documentation'], dtype='<U13')

These universal functions are also used in `numpy.char`, which provides
the `numpy.char.chararray` array subclass, in order for those routines
to get the performance benefits as well.

.. note::

   Prior to NumPy 2.0, all string functionality was in `numpy.char`, which
   only operated on fixed-width strings. That module will not be getting
   updates and will be deprecated at some point in the future.

String operations
-----------------

.. autosummary::
   :toctree: generated/

   add
   center
   capitalize
   decode
   encode
   expandtabs
   ljust
   lower
   lstrip
   mod
   multiply
   partition
   replace
   rjust
   rpartition
   rstrip
   slice
   strip
   swapcase
   title
   translate
   upper
   zfill

Comparison
----------

The `numpy.strings` module also exports the comparison universal functions
that can now operate on string arrays as well.

.. autosummary::
   :toctree: generated/

   equal
   not_equal
   greater_equal
   less_equal
   greater
   less

String information
------------------

.. autosummary::
   :toctree: generated/

   count
   endswith
   find
   index
   isalnum
   isalpha
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
