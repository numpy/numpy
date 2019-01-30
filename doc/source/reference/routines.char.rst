String operations
*****************

.. currentmodule:: numpy.core.defchararray

This module provides a set of vectorized string operations for arrays
of type `numpy.string_` or `numpy.unicode_`.   All of them are based on
the string methods in the Python standard library.

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
