.. _float16:

Half-precision API
==================

The header file ``<numpy/float16.h>`` provides helper routines for working
with IEEE 754-2008 16-bit floating-point values

Unlike ``<numpy/halffloat.h>``, this header is a header only implmentation
and does not require linking against the ``npymath`` static library. All
functionality is implemented directly in the header using ``<math.h>`` and
``<fenv.h>``.

Like ``<numpy/halffloat.h>`` The underlying storage type for ``npy_half`` is
``npy_uint16``. As a consequence, you cannot safely treat ``npy_half`` as
an ordinary C floating type. For example, 1.0 looks like ``0x3c00`` to C,
and if you do an equality comparison between the different signed zeros, you
will get ``-0.0 != 0.0`` (``0x8000 != 0x0000``), which is not the expected
behavior for an IEEE 754 floating type.

For these reasons, NumPy provides an API to work with npy_half values
accessible by including ``<numpy/float16.h>``.


Rather than manipulating the raw bits directly. For operations that are not
provided directly(for example arithmetic operations), the recommended
approach is to convert to ``float`` or ``double``, perform the computation,
and then convert back, as shown below::

    #include "numpy/arrayobject.h"
    #include "numpy/float16.h"

    npy_half sum(int n, npy_half *array)
    {
      float ret = 0;
      while(n--) {
          ret += npy_half_to_float(*array++);
      }
      return npy_float_to_half(ret);
    }

The API in ``<numpy/float16.h>`` closely mirrors the one in
``<numpy/halffloat.h>``, but uses an ``npy_float16_*`` prefix for the
functions and is designed to be usable without any additional libraries.

External references
-------------------

The following external resources describe the IEEE 754 half-precision
format that ``npy_half`` implements:

* `754-2008 IEEE Standard for Floating-Point Arithmetic`__
* `Half-precision Float Wikipedia Article`__.
* `OpenGL Half Float Pixel Support`__
* `The OpenEXR image format`__.

__ https://ieeexplore.ieee.org/document/4610935/
__ https://en.wikipedia.org/wiki/Half-precision_floating-point_format
__ https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_half_float_pixel.txt
__ https://www.openexr.com/about.html

Constants
---------

:c:macro:`NPY_HALF_ZERO`
  Bit pattern for positive zero.

:c:macro:`NPY_HALF_PZERO`
  Alias for ``NPY_HALF_ZERO`` (positive zero).

:c:macro:`NPY_HALF_NZERO`
  Bit pattern for negative zero.

:c:macro:`NPY_HALF_ONE`
  Bit pattern for the value 1.0.

:c:macro:`NPY_HALF_NEGONE`
  Bit pattern for the value -1.0.

:c:macro:`NPY_HALF_PINF`
  Bit pattern for positive infinity.

:c:macro:`NPY_HALF_NINF`
  Bit pattern for negative infinity.

:c:macro:`NPY_HALF_NAN`
  Bit pattern for a NaN value, guaranteed to have its sign bit unset.

:c:macro:`NPY_MAX_HALF`
  Bit pattern for the largest finite half-precision value representable in
  IEEE 754 (``65504.0``).

Conversion functions
--------------------

.. c:function:: float npy_float16_to_float(npy_half h)

   Converts a half-precision value to a single-precision float value.

.. c:function:: double npy_float16_to_double(npy_half h)

   Converts a half-precision value to a double-precision doubl) value.

.. c:function:: npy_half npy_float_to_float16(float f)

   Converts a single-precision value to a half-precision value. The result is
   rounded to the nearest representable value, with ties going to the nearest
   even. If the input is outside the finite range of ``npy_half``, a floating
   point overflow or underflow status flag is raised via ``feraiseexcept``
   (when available).

.. c:function:: npy_half npy_double_to_float16(double d)

   Converts a double-precision value to a half-precision value. The result is
   rounded to the nearest representable value, with ties going to the nearest
   even. If the input is outside the finite range of ``npy_half``, a floating
   point overflow or underflow status flag is raised via ``feraiseexcept``
   (when available).

Comparison functions
--------------------

The following functions implement IEEE 754–aware comparisons on the stored
``npy_half`` bit patterns, including correct handling of signed zeros and
NaNs.

.. c:function:: int npy_float16_eq(npy_half h1, npy_half h2)

   Compare two half-precision values for equality (``h1 == h2``). NaN
   arguments always compare unequal, and positive and negative zero compare
   as equal.

.. c:function:: int npy_float16_ne(npy_half h1, npy_half h2)

   Compare two half-precision values for inequality (``h1 != h2``). This is
   the logical negation of :c:func:`npy_float16_eq`.

.. c:function:: int npy_float16_le(npy_half h1, npy_half h2)

   Compare two half-precision values (``h1 <= h2``), with NaNs propagating to
   a false result.

.. c:function:: int npy_float16_lt(npy_half h1, npy_half h2)

   Compare two half-precision values (``h1 < h2``), with NaNs propagating to
   a false result.

.. c:function:: int npy_float16_ge(npy_half h1, npy_half h2)

   Compare two half-precision values (``h1 >= h2``), with NaNs propagating to
   a false result.

.. c:function:: int npy_float16_gt(npy_half h1, npy_half h2)

   Compare two half-precision values (``h1 > h2``), with NaNs propagating to
   a false result.

NaN free comparison variants
----------------------------

The following helpers are optimized for the cases where it is known in advance
that neither argument is a NaN. If a NaN is passed anyway, the result is
undefined.

.. c:function:: int npy_float16_eq_nonan(npy_half h1, npy_half h2)

   Compare two non NaN half-precision values for equality
   (``h1 == h2``). Treats positive and negative zero as equal.

.. c:function:: int npy_float16_lt_nonan(npy_half h1, npy_half h2)

   Compare two non NaN half-precision values (``h1 < h2``).

.. c:function:: int npy_float16_le_nonan(npy_half h1, npy_half h2)

   Compare two non NaN half-precision values (``h1 <= h2``).

Classification and sign inspection
----------------------------------

.. c:function:: int npy_float16_iszero(npy_half h)

   Test whether the value is either positive or negative zero. This
   may be slightly faster than comparing with ``NPY_HALF_ZERO``.

.. c:function:: int npy_float16_isnan(npy_half h)

   Test whether the value is a NaN.

.. c:function:: int npy_float16_isinf(npy_half h)

   Test whether the value is positive or negative infinity.

.. c:function:: int npy_float16_isfinite(npy_half h)

   Test whether the value is finite (i.e. not NaN and not infinite).

.. c:function:: int npy_float16_signbit(npy_half h)

   Return 1 if the sign bit of ``h`` is set (a negative value, including
   negative zero and negative inf), and 0 otherwise.

.. c:function:: npy_half npy_float16_copysign(npy_half x, npy_half y)

   Return ``x`` with its sign bit taken from ``y``. Works for all values,
   including inf and NaNs.

Next representable value
------------------------------------

.. c:function:: npy_half npy_float16_nextafter(npy_half x, npy_half y)

   Return the next representable half-precision value after ``x`` in the
   direction of ``y``. The behavior mirrors :c:func:`npy_nextafter`.

Divmod
------
.. c:function:: npy_half npy_float16_divmod(npy_half h1, npy_half h2, npy_half *modulus)

   Compute the result of dividing ``h1`` by ``h2``
   consistent with Python’s :func:`divmod` for floating values.

Bit-level conversion helpers
----------------------------

These functions expose the raw bit level conversions between binary
floating point formats, without going through a C floating type explicitly.

.. c:function:: npy_uint16 npy_floatbits_to_float16bits(npy_uint32 f)

   Low-level function which converts a 32-bit single-precision float,
   stored as a uint32, into a 16-bit half-precision float.

.. c:function:: npy_uint16 npy_doublebits_to_float16bits(npy_uint64 d)

   Low-level function which converts a 64-bit double-precision float, stored
   as a uint64, into a 16-bit half-precision float.

.. c:function:: npy_uint32 npy_float16bits_to_floatbits(npy_uint16 h)

   Low-level function which converts a 16-bit half-precision float
   into a 32-bit single-precision float, stored as a uint32.

.. c:function:: npy_uint64 npy_float16bits_to_doublebits(npy_uint16 h)

   Low-level function which converts a 16-bit half-precision float
   into a 64-bit double-precision float, stored as a uint64.

