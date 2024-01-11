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
to get the performance benefits as well. The one difference between
`numpy.char` and `numpy.strings` is that the routines in `numpy.char`
provide some sensible defaults for keyword argument is certain routines.
For example

      >>> np.char.endswith('hello', 'lo')
      np.True_
      >>> np.strings.endswith('hello', 'lo', 0, len('hello'))
      np.True_
      >>> np.strings.endswith('hello', 'lo')
      Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
      TypeError: endswith() takes from 4 to 5 positional arguments but 2 were given

This is because `numpy.strings` exports the universal functions directly,
while `numpy.char` wraps them with Python-level functions.

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
