String universal functions
==========================

.. currentmodule:: numpy.strings

.. module:: numpy.strings

The `numpy.strings` module provides a set of universal functions operating
on arrays of type `numpy.str_`, `numpy.bytes_` or `StringDType`.
For example

      >>> np.strings.add(["num", "doc"], ["py", "umentation"])
      array(['numpy', 'documentation'], dtype='<U13')

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
