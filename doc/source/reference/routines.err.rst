Floating point error handling
*****************************

NumPy supports floating-point exceptions as defined in the IEEE 745
standard [1]_:

- Invalid operation: result is not an expressible number.  This typically
  indicates that a new NaN (not a number) was produced, e.g. by ``0. / 0.``.
- Division by zero: infinite result obtained from finite numbers,
  e.g. by ``1. / 0.``
- Overflow: result too large to be expressed.
- Underflow: result so close to zero that some precision was lost.

NumPy does not warn for comparisons involving a NaN, such as ``NaN < 0.``.
This differs from the IEEE standard, which specifies warning for the default
operators: ``<, <=, >, >=`` when at least one value is a NaN.

The floating-point error handling can be customized using the below functions.
By default all except "underflow" give a warning.

In some cases NumPy will use these floating point error settings
also for Integer operations.

.. admonition:: Advanced details

   * Floating-point errors rely on the compiler, hardware, and math library.
     NumPy tries to ensure correct warning behavior, but some systems may
     have incomplete support or choose speed over correct floating-point errors.
     For example the MacOS math library is known to not indicate some errors.

   * IEEE defines a "signalling NaN" or sNaN.  NumPy will never create these.
     If you import data containing such an sNaN you may see unexpected
     warnings.  In that case you can use::

         with np.errstate(invalid="ignore"):
             np.add(arr, 0, out=arr)

     to convert all signalling NaNs to normal (quiet) ones.  Signalling NaNs
     are expected to signal an error on almost all operations.  However, NumPy
     may not always behave IEEE conform with respect to warnings.


.. [1] https://en.wikipedia.org/wiki/IEEE_754


.. currentmodule:: numpy

Setting and getting error handling
----------------------------------

.. autosummary::
   :toctree: generated/

   seterr
   geterr
   seterrcall
   geterrcall
   errstate

Internal functions
------------------

.. autosummary::
   :toctree: generated/

   seterrobj
   geterrobj
