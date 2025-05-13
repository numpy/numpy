.. _routines.testing:
.. module:: numpy.testing

Test support
============

.. currentmodule:: numpy.testing

Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy
tests in a single location, so that :ref:`test scripts
<development-environment>` can just import it and work right away. For
background, see the :ref:`testing-guidelines`


Asserts
-------
.. autosummary::
   :toctree: generated/

   assert_allclose
   assert_array_almost_equal_nulp
   assert_array_max_ulp
   assert_array_equal
   assert_array_less
   assert_equal
   assert_raises
   assert_raises_regex
   assert_warns
   assert_no_warnings
   assert_no_gc_cycles
   assert_string_equal

Asserts (not recommended)
-------------------------
It is recommended to use one of `assert_allclose`,
`assert_array_almost_equal_nulp` or `assert_array_max_ulp` instead of these
functions for more consistent floating point comparisons.

.. autosummary::
   :toctree: generated/

   assert_
   assert_almost_equal
   assert_approx_equal
   assert_array_almost_equal
   print_assert_equal

Decorators
----------
.. autosummary::
   :toctree: generated/

   decorate_methods

Test running
------------
.. autosummary::
   :toctree: generated/

   clear_and_catch_warnings
   measure
   rundocs
   suppress_warnings

.. module:: numpy.testing.overrides

Testing custom array containers (:mod:`numpy.testing.overrides`)
----------------------------------------------------------------

These functions can be useful when testing custom array container
implementations which make use of ``__array_ufunc__``/``__array_function__``.

.. currentmodule:: numpy.testing.overrides

.. autosummary::
   :toctree: generated/

   allows_array_function_override
   allows_array_ufunc_override
   get_overridable_numpy_ufuncs
   get_overridable_numpy_array_functions


Guidelines
----------

.. toctree::

   testing
