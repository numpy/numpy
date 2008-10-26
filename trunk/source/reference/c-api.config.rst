Configuration defines
=====================

.. sectionauthor:: Travis E. Oliphant

When NumPy is built, a configuration file is constructed and placed as
config.h in the NumPy include directory. This configuration file
ensures that specific macros are defined and defines other macros
based on whether or not your system has certain features. It is
included by the arrayobject.h file.


Guaranteed to be defined
------------------------

The :cdata:`SIZEOF_{CTYPE}` constants are defined so that sizeof
information is available to the pre-processor.

.. cvar:: CHAR_BIT

    The number of bits of a char. The char is the unit of all sizeof
    definitions

.. cvar:: SIZEOF_SHORT

    sizeof(short)

.. cvar:: SIZEOF_INT

    sizeof(int)

.. cvar:: SIZEOF_LONG

    sizeof(long)

.. cvar:: SIZEOF_LONG_LONG

    sizeof(longlong) where longlong is defined appropriately on the
    platform (A macro defines **SIZEOF_LONGLONG** as well.)

.. cvar:: SIZEOF_PY_LONG_LONG
    

.. cvar:: SIZEOF_FLOAT

    sizeof(float)

.. cvar:: SIZEOF_DOUBLE

    sizeof(double)

.. cvar:: SIZEOF_LONG_DOUBLE

    sizeof(longdouble) (A macro defines **SIZEOF_LONGDOUBLE** as well.)

.. cvar:: SIZEOF_PY_INTPTR_T

    Size of a pointer on this platform (sizeof(void \*)) (A macro defines
    SIZEOF_INTP as well.)


Possible defines
----------------

These defines will cause the compilation to ignore compatibility code
that is placed in NumPy and use the system code instead. If they are
not defined, then the system does not have that capability. 

.. cvar:: HAVE_LONGDOUBLE_FUNCS

    System has C99 long double math functions.

.. cvar:: HAVE_FLOAT_FUNCS

    System has C99 float math functions.

.. cvar:: HAVE_INVERSE_HYPERBOLIC

    System has inverse hyperbolic functions: asinh, acosh, and atanh.

.. cvar:: HAVE_INVERSE_HYPERBOLIC_FLOAT

    System has C99 float extensions to inverse hyperbolic functions:
    asinhf, acoshf, atanhf

.. cvar:: HAVE_INVERSE_HYPERBOLIC_LONGDOUBLE

    System has C99 long double extensions to inverse hyperbolic functions:
    asinhl, acoshl, atanhl.

.. cvar:: HAVE_ISNAN

    System has an isnan function.

.. cvar:: HAVE_ISINF

    System has an isinf function.

.. cvar:: HAVE_LOG1P

    System has the log1p function: :math:`\log\left(x+1\right)`.

.. cvar:: HAVE_EXPM1

    System has the expm1 function: :math:`\exp\left(x\right)-1`.

.. cvar:: HAVE_RINT

    System has the rint function.

