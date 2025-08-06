NumPy core math library
=======================

The numpy core math library (``npymath``) is a first step in this direction. This
library contains most math-related C99 functionality, which can be used on
platforms where C99 is not well supported. The core math functions have the
same API as the C99 ones, except for the ``npy_*`` prefix.

The available functions are defined in ``<numpy/npy_math.h>`` - please refer to
this header when in doubt.

.. note::

   An effort is underway to make ``npymath`` smaller (since C99 compatibility
   of compilers has improved over time) and more easily vendorable or usable as
   a header-only dependency. That will avoid problems with shipping a static
   library built with a compiler which may not match the compiler used by a
   downstream package or end user. See
   `gh-20880 <https://github.com/numpy/numpy/issues/20880>`__ for details.

Floating point classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:macro:: NPY_NAN

    This macro is defined to a NaN (Not a Number), and is guaranteed to have
    the signbit unset ('positive' NaN). The corresponding single and extension
    precision macro are available with the suffix F and L.

.. c:macro:: NPY_INFINITY

    This macro is defined to a positive inf. The corresponding single and
    extension precision macro are available with the suffix F and L.

.. c:macro:: NPY_PZERO

    This macro is defined to positive zero. The corresponding single and
    extension precision macro are available with the suffix F and L.

.. c:macro:: NPY_NZERO

    This macro is defined to negative zero (that is with the sign bit set). The
    corresponding single and extension precision macro are available with the
    suffix F and L.

.. c:macro:: npy_isnan(x)

    This is an alias for C99 isnan: works for single, double
    and extended precision, and return a non 0 value if x is a NaN.

.. c:macro:: npy_isfinite(x)

    This is an alias for C99 isfinite: works for single,
    double and extended precision, and return a non 0 value if x is neither a
    NaN nor an infinity.

.. c:macro:: npy_isinf(x)

    This is an alias for C99 isinf: works for single, double
    and extended precision, and return a non 0 value if x is infinite (positive
    and negative).

.. c:macro:: npy_signbit(x)

    This is an alias for C99 signbit: works for single, double
    and extended precision, and return a non 0 value if x has the signbit set
    (that is the number is negative).

.. c:macro:: npy_copysign(x, y)

    This is an alias for  C99 copysign: return x with the same sign
    as y. Works for any value, including inf and nan. Single and extended
    precisions are available with suffix f and l.

Useful math constants
~~~~~~~~~~~~~~~~~~~~~

The following math constants are available in ``npy_math.h``. Single
and extended precision are also available by adding the ``f`` and
``l`` suffixes respectively.

.. c:macro:: NPY_E

    Base of natural logarithm (:math:`e`)

.. c:macro:: NPY_LOG2E

    Logarithm to base 2 of the Euler constant (:math:`\frac{\ln(e)}{\ln(2)}`)

.. c:macro:: NPY_LOG10E

    Logarithm to base 10 of the Euler constant (:math:`\frac{\ln(e)}{\ln(10)}`)

.. c:macro:: NPY_LOGE2

    Natural logarithm of 2 (:math:`\ln(2)`)

.. c:macro:: NPY_LOGE10

    Natural logarithm of 10 (:math:`\ln(10)`)

.. c:macro:: NPY_PI

    Pi (:math:`\pi`)

.. c:macro:: NPY_PI_2

    Pi divided by 2 (:math:`\frac{\pi}{2}`)

.. c:macro:: NPY_PI_4

    Pi divided by 4 (:math:`\frac{\pi}{4}`)

.. c:macro:: NPY_1_PI

    Reciprocal of pi (:math:`\frac{1}{\pi}`)

.. c:macro:: NPY_2_PI

    Two times the reciprocal of pi (:math:`\frac{2}{\pi}`)

.. c:macro:: NPY_EULER

    The Euler constant
        :math:`\lim_{n\rightarrow\infty}({\sum_{k=1}^n{\frac{1}{k}}-\ln n})`

Low-level floating point manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Those can be useful for precise floating point comparison.

.. c:function:: double npy_nextafter(double x, double y)

    This is an alias to C99 nextafter: return next representable
    floating point value from x in the direction of y. Single and extended
    precisions are available with suffix f and l.

.. c:function:: double npy_spacing(double x)

    This is a function equivalent to Fortran intrinsic. Return distance between
    x and next representable floating point value from x, e.g. spacing(1) ==
    eps. spacing of nan and +/- inf return nan. Single and extended precisions
    are available with suffix f and l.

.. c:function:: void npy_set_floatstatus_divbyzero()

    Set the divide by zero floating point exception

.. c:function:: void npy_set_floatstatus_overflow()

    Set the overflow floating point exception

.. c:function:: void npy_set_floatstatus_underflow()

    Set the underflow floating point exception

.. c:function:: void npy_set_floatstatus_invalid()

    Set the invalid floating point exception

.. c:function:: int npy_get_floatstatus()

    Get floating point status. Returns a bitmask with following possible flags:

    * NPY_FPE_DIVIDEBYZERO
    * NPY_FPE_OVERFLOW
    * NPY_FPE_UNDERFLOW
    * NPY_FPE_INVALID

    Note that :c:func:`npy_get_floatstatus_barrier` is preferable as it prevents
    aggressive compiler optimizations reordering the call relative to
    the code setting the status, which could lead to incorrect results.

.. c:function:: int npy_get_floatstatus_barrier(char*)

    Get floating point status. A pointer to a local variable is passed in to
    prevent aggressive compiler optimizations from reordering this function call
    relative to the code setting the status, which could lead to incorrect
    results.

    Returns a bitmask with following possible flags:

    * NPY_FPE_DIVIDEBYZERO
    * NPY_FPE_OVERFLOW
    * NPY_FPE_UNDERFLOW
    * NPY_FPE_INVALID

.. c:function:: int npy_clear_floatstatus()

    Clears the floating point status. Returns the previous status mask.

    Note that :c:func:`npy_clear_floatstatus_barrier` is preferable as it
    prevents aggressive compiler optimizations reordering the call relative to
    the code setting the status, which could lead to incorrect results.

.. c:function:: int npy_clear_floatstatus_barrier(char*)

    Clears the floating point status. A pointer to a local variable is passed in to
    prevent aggressive compiler optimizations from reordering this function call.
    Returns the previous status mask.

.. _complex-numbers:

Support for complex numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

C99-like complex functions have been added. Those can be used if you wish to
implement portable C extensions. Since NumPy 2.0 we use C99 complex types as
the underlying type:

.. code-block:: c

    typedef double _Complex npy_cdouble;
    typedef float _Complex npy_cfloat;
    typedef long double _Complex npy_clongdouble;

MSVC does not support the ``_Complex`` type itself, but has added support for
the C99 ``complex.h`` header by providing its own implementation. Thus, under
MSVC, the equivalent MSVC types will be used:

.. code-block:: c

    typedef _Dcomplex npy_cdouble;
    typedef _Fcomplex npy_cfloat;
    typedef _Lcomplex npy_clongdouble;

Because MSVC still does not support C99 syntax for initializing a complex
number, you need to restrict to C90-compatible syntax, e.g.:

.. code-block:: c

        /* a = 1 + 2i \*/
        npy_complex a = npy_cpack(1, 2);
        npy_complex b;

        b = npy_log(a);

A few utilities have also been added in
``numpy/npy_math.h``, in order to retrieve or set the real or the imaginary
part of a complex number:

.. code-block:: c

    npy_cdouble c;
    npy_csetreal(&c, 1.0);
    npy_csetimag(&c, 0.0);
    printf("%d + %di\n", npy_creal(c), npy_cimag(c));

.. versionchanged:: 2.0.0

    The underlying C types for all of numpy's complex types have been changed to
    use C99 complex types. Up until now the following was being used to represent
    complex types:

    .. code-block:: c

        typedef struct { double real, imag; } npy_cdouble;
        typedef struct { float real, imag; } npy_cfloat;
        typedef struct {npy_longdouble real, imag;} npy_clongdouble;

    Using the ``struct`` representation ensured that complex numbers could be used
    on all platforms, even the ones without support for built-in complex types. It
    also meant that a static library had to be shipped together with NumPy to
    provide a C99 compatibility layer for downstream packages to use. In recent
    years however, support for native complex types has been improved immensely,
    with MSVC adding built-in support for the ``complex.h`` header in 2019.

    To ease cross-version compatibility, macros that use the new set APIs have
    been added.

    .. code-block:: c

        #define NPY_CSETREAL(z, r) npy_csetreal(z, r)
        #define NPY_CSETIMAG(z, i) npy_csetimag(z, i)

    A compatibility layer is also provided in ``numpy/npy_2_complexcompat.h``. It
    checks whether the macros exist, and falls back to the 1.x syntax in case they
    don't.

    .. code-block:: c

        #include <numpy/npy_math.h>

        #ifndef NPY_CSETREALF
        #define NPY_CSETREALF(c, r) (c)->real = (r)
        #endif
        #ifndef NPY_CSETIMAGF
        #define NPY_CSETIMAGF(c, i) (c)->imag = (i)
        #endif

    We suggest all downstream packages that need this functionality to copy-paste
    the compatibility layer code into their own sources and use that, so that
    they can continue to support both NumPy 1.x and 2.x without issues. Note also
    that the ``complex.h`` header is included in ``numpy/npy_common.h``, which
    makes ``complex`` a reserved keyword.

.. _linking-npymath:

Linking against the core math library in an extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the core math library that NumPy ships as a static library in your own
Python extension, you need to add the ``npymath`` compile and link options to your
extension. The exact steps to take will depend on the build system you are using.
The generic steps to take are:

1. Add the numpy include directory (= the value of ``np.get_include()``) to
   your include directories,
2. The ``npymath`` static library resides in the ``lib`` directory right next
   to numpy's include directory (i.e., ``pathlib.Path(np.get_include()) / '..'
   / 'lib'``). Add that to your library search directories,
3. Link with ``libnpymath`` and ``libm``.

.. note::

   Keep in mind that when you are cross compiling, you must use the ``numpy``
   for the platform you are building for, not the native one for the build
   machine. Otherwise you pick up a static library built for the wrong
   architecture.

When you build with ``numpy.distutils`` (deprecated), then use this in your ``setup.py``:

        .. hidden in a comment so as to be included in refguide but not rendered documentation
                >>> import numpy.distutils.misc_util
                >>> config = np.distutils.misc_util.Configuration(None, '', '.')
                >>> with open('foo.c', 'w') as f: pass

        >>> from numpy.distutils.misc_util import get_info
        >>> info = get_info('npymath')
        >>> _ = config.add_extension('foo', sources=['foo.c'], extra_info=info)

In other words, the usage of ``info`` is exactly the same as when using
``blas_info`` and co.

When you are building with `Meson <https://mesonbuild.com>`__, use::

    # Note that this will get easier in the future, when Meson has
    # support for numpy built in; most of this can then be replaced
    # by `dependency('numpy')`.
    incdir_numpy = run_command(py3,
      [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
      ],
      check: true
    ).stdout().strip()

    inc_np = include_directories(incdir_numpy)

    cc = meson.get_compiler('c')
    npymath_path = incdir_numpy / '..' / 'lib'
    npymath_lib = cc.find_library('npymath', dirs: npymath_path)

    py3.extension_module('module_name',
      ...
      include_directories: inc_np,
      dependencies: [npymath_lib],

Half-precision functions
~~~~~~~~~~~~~~~~~~~~~~~~

The header file ``<numpy/halffloat.h>`` provides functions to work with
IEEE 754-2008 16-bit floating point values. While this format is
not typically used for numerical computations, it is useful for
storing values which require floating point but do not need much precision.
It can also be used as an educational tool to understand the nature
of floating point round-off error.

Like for other types, NumPy includes a typedef npy_half for the 16 bit
float.  Unlike for most of the other types, you cannot use this as a
normal type in C, since it is a typedef for npy_uint16.  For example,
1.0 looks like 0x3c00 to C, and if you do an equality comparison
between the different signed zeros, you will get -0.0 != 0.0
(0x8000 != 0x0000), which is incorrect.

For these reasons, NumPy provides an API to work with npy_half values
accessible by including ``<numpy/halffloat.h>`` and linking to ``npymath``.
For functions that are not provided directly, such as the arithmetic
operations, the preferred method is to convert to float
or double and back again, as in the following example.

.. code-block:: c

        npy_half sum(int n, npy_half *array) {
            float ret = 0;
            while(n--) {
                ret += npy_half_to_float(*array++);
            }
            return npy_float_to_half(ret);
        }

External Links:

* `754-2008 IEEE Standard for Floating-Point Arithmetic`__
* `Half-precision Float Wikipedia Article`__.
* `OpenGL Half Float Pixel Support`__
* `The OpenEXR image format`__.

__ https://ieeexplore.ieee.org/document/4610935/
__ https://en.wikipedia.org/wiki/Half-precision_floating-point_format
__ https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_half_float_pixel.txt
__ https://www.openexr.com/about.html

.. c:macro:: NPY_HALF_ZERO

    This macro is defined to positive zero.

.. c:macro:: NPY_HALF_PZERO

    This macro is defined to positive zero.

.. c:macro:: NPY_HALF_NZERO

    This macro is defined to negative zero.

.. c:macro:: NPY_HALF_ONE

    This macro is defined to 1.0.

.. c:macro:: NPY_HALF_NEGONE

    This macro is defined to -1.0.

.. c:macro:: NPY_HALF_PINF

    This macro is defined to +inf.

.. c:macro:: NPY_HALF_NINF

    This macro is defined to -inf.

.. c:macro:: NPY_HALF_NAN

    This macro is defined to a NaN value, guaranteed to have its sign bit unset.

.. c:function:: float npy_half_to_float(npy_half h)

   Converts a half-precision float to a single-precision float.

.. c:function:: double npy_half_to_double(npy_half h)

   Converts a half-precision float to a double-precision float.

.. c:function:: npy_half npy_float_to_half(float f)

   Converts a single-precision float to a half-precision float.  The
   value is rounded to the nearest representable half, with ties going
   to the nearest even.  If the value is too small or too big, the
   system's floating point underflow or overflow bit will be set.

.. c:function:: npy_half npy_double_to_half(double d)

   Converts a double-precision float to a half-precision float.  The
   value is rounded to the nearest representable half, with ties going
   to the nearest even.  If the value is too small or too big, the
   system's floating point underflow or overflow bit will be set.

.. c:function:: int npy_half_eq(npy_half h1, npy_half h2)

   Compares two half-precision floats (h1 == h2).

.. c:function:: int npy_half_ne(npy_half h1, npy_half h2)

   Compares two half-precision floats (h1 != h2).

.. c:function:: int npy_half_le(npy_half h1, npy_half h2)

   Compares two half-precision floats (h1 <= h2).

.. c:function:: int npy_half_lt(npy_half h1, npy_half h2)

   Compares two half-precision floats (h1 < h2).

.. c:function:: int npy_half_ge(npy_half h1, npy_half h2)

   Compares two half-precision floats (h1 >= h2).

.. c:function:: int npy_half_gt(npy_half h1, npy_half h2)

   Compares two half-precision floats (h1 > h2).

.. c:function:: int npy_half_eq_nonan(npy_half h1, npy_half h2)

   Compares two half-precision floats that are known to not be NaN (h1 == h2).  If
   a value is NaN, the result is undefined.

.. c:function:: int npy_half_lt_nonan(npy_half h1, npy_half h2)

   Compares two half-precision floats that are known to not be NaN (h1 < h2).  If
   a value is NaN, the result is undefined.

.. c:function:: int npy_half_le_nonan(npy_half h1, npy_half h2)

   Compares two half-precision floats that are known to not be NaN (h1 <= h2).  If
   a value is NaN, the result is undefined.

.. c:function:: int npy_half_iszero(npy_half h)

   Tests whether the half-precision float has a value equal to zero.  This may be slightly
   faster than calling npy_half_eq(h, NPY_ZERO).

.. c:function:: int npy_half_isnan(npy_half h)

   Tests whether the half-precision float is a NaN.

.. c:function:: int npy_half_isinf(npy_half h)

   Tests whether the half-precision float is plus or minus Inf.

.. c:function:: int npy_half_isfinite(npy_half h)

   Tests whether the half-precision float is finite (not NaN or Inf).

.. c:function:: int npy_half_signbit(npy_half h)

   Returns 1 is h is negative, 0 otherwise.

.. c:function:: npy_half npy_half_copysign(npy_half x, npy_half y)

    Returns the value of x with the sign bit copied from y.  Works for any value,
    including Inf and NaN.

.. c:function:: npy_half npy_half_spacing(npy_half h)

    This is the same for half-precision float as npy_spacing and npy_spacingf
    described in the low-level floating point section.

.. c:function:: npy_half npy_half_nextafter(npy_half x, npy_half y)

    This is the same for half-precision float as npy_nextafter and npy_nextafterf
    described in the low-level floating point section.

.. c:function:: npy_uint16 npy_floatbits_to_halfbits(npy_uint32 f)

   Low-level function which converts a 32-bit single-precision float, stored
   as a uint32, into a 16-bit half-precision float.

.. c:function:: npy_uint16 npy_doublebits_to_halfbits(npy_uint64 d)

   Low-level function which converts a 64-bit double-precision float, stored
   as a uint64, into a 16-bit half-precision float.

.. c:function:: npy_uint32 npy_halfbits_to_floatbits(npy_uint16 h)

   Low-level function which converts a 16-bit half-precision float
   into a 32-bit single-precision float, stored as a uint32.

.. c:function:: npy_uint64 npy_halfbits_to_doublebits(npy_uint16 h)

   Low-level function which converts a 16-bit half-precision float
   into a 64-bit double-precision float, stored as a uint64.
