String universal functions
==========================

.. currentmodule:: numpy.strings

.. module:: numpy.strings

The `numpy.strings` module provides a set of universal functions operating
on arrays of type `numpy.str_` or `numpy.bytes_`.
For example

      >>> np.strings.add(["num", "doc"], ["py", "umentation"])
      array(['numpy', 'documentation'], dtype='<U13')

These universal functions are also used in `numpy.char`, which provides
the `numpy.char.chararray` array subclass, in order for those routines
to get the performance benefits as well.

String operations
-----------------

.. autosummary::
   :toctree: generated/

   add
   lstrip
   rstrip
   strip

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
   isalpha
   isdecimal
   isdigit
   isnumeric
   isspace
   rfind
   startswith
   str_len
