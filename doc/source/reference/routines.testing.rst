.. module:: numpy.testing

Test Support (:mod:`numpy.testing`)
===================================

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
   assert_string_equal

Asserts (not recommended)
-------------------------
It is recommended to use one of `assert_allclose`,
`assert_array_almost_equal_nulp` or `assert_array_max_ulp` instead of these
functions for more consistent floating point comparisons.

.. autosummary::
   :toctree: generated/

   assert_almost_equal
   assert_approx_equal
   assert_array_almost_equal

Decorators
----------
.. autosummary::
   :toctree: generated/

   dec.deprecated
   dec.knownfailureif
   dec.setastest
   dec.skipif
   dec.slow
   decorate_methods

Test Running
------------
.. autosummary::
   :toctree: generated/

   Tester
   run_module_suite
   rundocs
   suppress_warnings

Guidelines
----------

.. toctree::

   testing
