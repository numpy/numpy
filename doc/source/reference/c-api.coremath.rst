Numpy core libraries
====================

.. sectionauthor:: David Cournapeau

.. versionadded:: 1.3.0

Starting from numpy 1.3.0, we are working on separating the pure C,
"computational" code from the python dependent code. The goal is twofolds:
making the code cleaner, and enabling code reuse by other extensions outside
numpy (scipy, etc...).

Numpy core math library
-----------------------

The numpy core math library ('npymath') is a first step in this direction. This
library contains most math-related C99 functionality, which can be used on
platforms where C99 is not well supported. The core math functions have the
same API as the C99 ones, except for the npy_* prefix.

The available functions are defined in npy_math.h - please refer to this header
in doubt.

Floating point classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cvar:: NPY_NAN

    This macro is defined to a NaN (Not a Number), and is guaranteed to have
    the signbit unset ('positive' NaN). The corresponding single and extension
    precision macro are available with the suffix F and L.

.. cvar:: NPY_INFINITY

    This macro is defined to a positive inf. The corresponding single and
    extension precision macro are available with the suffix F and L.

.. cvar:: NPY_PZERO

    This macro is defined to positive zero. The corresponding single and
    extension precision macro are available with the suffix F and L.

.. cvar:: NPY_NZERO

    This macro is defined to negative zero (that is with the sign bit set). The
    corresponding single and extension precision macro are available with the
    suffix F and L.

.. cfunction:: int npy_isnan(x)

    This is a macro, and is equivalent to C99 isnan: works for single, double
    and extended precision, and return a non 0 value is x is a NaN.

.. cfunction:: int npy_isfinite(x)

    This is a macro, and is equivalent to C99 isfinite: works for single,
    double and extended precision, and return a non 0 value is x is neither a
    NaN or a infinity.

.. cfunction:: int npy_isinf(x)

    This is a macro, and is equivalent to C99 isinf: works for single, double
    and extended precision, and return a non 0 value is x is infinite (positive
    and negative).

.. cfunction:: int npy_signbit(x)

    This is a macro, and is equivalent to C99 signbit: works for single, double
    and extended precision, and return a non 0 value is x has the signbit set
    (that is the number is negative).

.. cfunction:: double npy_copysign(double x, double y)

    This is a function equivalent to C99 copysign: return x with the same sign
    as y. Works for any value, including inf and nan. Single and extended
    precisions are available with suffix f and l.

    .. versionadded:: 1.4.0

Useful math constants
~~~~~~~~~~~~~~~~~~~~~

The following math constants are available in npy_math.h. Single and extended
precision are also available by adding the F and L suffixes respectively.

.. cvar:: NPY_E

    Base of natural logarithm (:math:`e`)

.. cvar:: NPY_LOG2E

    Logarithm to base 2 of the Euler constant (:math:`\frac{\ln(e)}{\ln(2)}`)

.. cvar:: NPY_LOG10E

    Logarithm to base 10 of the Euler constant (:math:`\frac{\ln(e)}{\ln(10)}`)

.. cvar:: NPY_LOGE2

    Natural logarithm of 2 (:math:`\ln(2)`)

.. cvar:: NPY_LOGE10

    Natural logarithm of 10 (:math:`\ln(10)`)

.. cvar:: NPY_PI

    Pi (:math:`\pi`)

.. cvar:: NPY_PI_2

    Pi divided by 2 (:math:`\frac{\pi}{2}`)

.. cvar:: NPY_PI_4

    Pi divided by 4 (:math:`\frac{\pi}{4}`)

.. cvar:: NPY_1_PI

    Reciprocal of pi (:math:`\frac{1}{\pi}`)

.. cvar:: NPY_2_PI

    Two times the reciprocal of pi (:math:`\frac{2}{\pi}`)

.. cvar:: NPY_EULER

    The Euler constant (:math:`\lim_{n\rightarrow \infty}{\sum_{k=1}^n{\frac{1}{k}} - \ln n}`)

Linking against the core math library in an extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 1.4.0

To use the core math library in your own extension, you need to add the npymath
compile and link options to your extension in your setup.py:

        >>> from numpy.distutils.misc_utils import get_info
        >>> info = get_info('npymath')
        >>> config.add_extension('foo', sources=['foo.c'], extra_info=**info)

In other words, the usage of info is exactly the same as when using blas_info
and co.
