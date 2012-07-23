Test Support (:mod:`numpy.testing`)
===================================

.. currentmodule:: numpy.testing

Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy
tests in a single location, so that test scripts can just import it and
work right away.


Asserts
=======
.. autosummary::
   :toctree: generated/

   assert_almost_equal
   assert_approx_equal
   assert_array_almost_equal
   assert_allclose
   assert_array_almost_equal_nulp
   assert_array_max_ulp
   assert_array_equal
   assert_array_less
   assert_equal
   assert_raises
   assert_warns
   assert_string_equal

Decorators
----------
.. autosummary::
   :toctree: generated/

   decorators.deprecated
   decorators.knownfailureif
   decorators.setastest
   decorators.skipif
   decorators.slow
   decorate_methods


Test Running
------------
.. autosummary::
   :toctree: generated/

   Tester
   run_module_suite
   rundocs
