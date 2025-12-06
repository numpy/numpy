Floating point error handling
=============================

.. currentmodule:: numpy

Error handling settings are maintained on a per-thread basis, allowing different
threads to have independent configurations. For more information, see
:ref:`misc-error-handling` and :ref:`thread_safety`.

Setting and getting error handling
----------------------------------

.. autosummary::
   :toctree: generated/

   seterr
   geterr
   seterrcall
   geterrcall
   errstate
