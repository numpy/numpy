Data Type API
=============

.. sectionauthor:: Travis E. Oliphant

The standard array can have 24 different data types (and has some
support for adding your own types). These data types all have an
enumerated type, an enumerated type-character, and a corresponding
array scalar Python type object (placed in a hierarchy). There are
also standard C typedefs to make it easier to manipulate elements of
the given data type. For the numeric types, there are also bit-width
equivalent C typedefs and named typenumbers that make it easier to
select the precision desired.

.. warning::

    The names for the types in c code follows c naming conventions
    more closely. The Python names for these types follow Python
    conventions.  Thus, :cdata:`NPY_FLOAT` picks up a 32-bit float in
    C, but :class:`numpy.float_` in Python corresponds to a 64-bit
    double. The bit-width names can be used in both Python and C for
    clarity.


Enumerated Types
----------------

There is a list of enumerated types defined providing the basic 24
data types plus some useful generic names. Whenever the code requires
a type number, one of these enumerated types is requested. The types
are all called :cdata:`NPY_{NAME}`:

.. cvar:: NPY_BOOL

    The enumeration value for the boolean type, stored as one byte.
    It may only be set to the values 0 and 1.

.. cvar:: NPY_BYTE
.. cvar:: NPY_INT8

    The enumeration value for an 8-bit/1-byte signed integer.

.. cvar:: NPY_SHORT
.. cvar:: NPY_INT16

    The enumeration value for a 16-bit/2-byte signed integer.

.. cvar:: NPY_INT
.. cvar:: NPY_INT32

    The enumeration value for a 32-bit/4-byte signed integer.

.. cvar:: NPY_LONG

    Equivalent to either NPY_INT or NPY_LONGLONG, depending on the
    platform.

.. cvar:: NPY_LONGLONG
.. cvar:: NPY_INT64

    The enumeration value for a 64-bit/8-byte signed integer.

.. cvar:: NPY_UBYTE
.. cvar:: NPY_UINT8

    The enumeration value for an 8-bit/1-byte unsigned integer.

.. cvar:: NPY_USHORT
.. cvar:: NPY_UINT16

    The enumeration value for a 16-bit/2-byte unsigned integer.

.. cvar:: NPY_UINT
.. cvar:: NPY_UINT32

    The enumeration value for a 32-bit/4-byte unsigned integer.

.. cvar:: NPY_ULONG

    Equivalent to either NPY_UINT or NPY_ULONGLONG, depending on the
    platform.

.. cvar:: NPY_ULONGLONG
.. cvar:: NPY_UINT64

    The enumeration value for a 64-bit/8-byte unsigned integer.

.. cvar:: NPY_HALF
.. cvar:: NPY_FLOAT16

    The enumeration value for a 16-bit/2-byte IEEE 754-2008 compatible floating
    point type.

.. cvar:: NPY_FLOAT
.. cvar:: NPY_FLOAT32

    The enumeration value for a 32-bit/4-byte IEEE 754 compatible floating
    point type.

.. cvar:: NPY_DOUBLE
.. cvar:: NPY_FLOAT64

    The enumeration value for a 64-bit/8-byte IEEE 754 compatible floating
    point type.

.. cvar:: NPY_LONGDOUBLE

    The enumeration value for a platform-specific floating point type which is
    at least as large as NPY_DOUBLE, but larger on many platforms.

.. cvar:: NPY_CFLOAT
.. cvar:: NPY_COMPLEX64

    The enumeration value for a 64-bit/8-byte complex type made up of
    two NPY_FLOAT values.

.. cvar:: NPY_CDOUBLE
.. cvar:: NPY_COMPLEX128

    The enumeration value for a 128-bit/16-byte complex type made up of
    two NPY_DOUBLE values.

.. cvar:: NPY_CLONGDOUBLE

    The enumeration value for a platform-specific complex floating point
    type which is made up of two NPY_LONGDOUBLE values.

.. cvar:: NPY_DATETIME

    The enumeration value for a data type which holds dates or datetimes with
    a precision based on selectable date or time units.

.. cvar:: NPY_TIMEDELTA

    The enumeration value for a data type which holds lengths of times in
    integers of selectable date or time units.

.. cvar:: NPY_STRING

    The enumeration value for ASCII strings of a selectable size. The
    strings have a fixed maximum size within a given array.

.. cvar:: NPY_UNICODE

    The enumeration value for UCS4 strings of a selectable size. The
    strings have a fixed maximum size within a given array.

.. cvar:: NPY_OBJECT

    The enumeration value for references to arbitrary Python objects.

.. cvar:: NPY_VOID

    Primarily used to hold struct dtypes, but can contain arbitrary
    binary data.

Some useful aliases of the above types are

.. cvar:: NPY_INTP

    The enumeration value for a signed integer type which is the same
    size as a (void \*) pointer. This is the type used by all
    arrays of indices.

.. cvar:: NPY_UINTP

    The enumeration value for an unsigned integer type which is the
    same size as a (void \*) pointer.

.. cvar:: NPY_MASK

    The enumeration value of the type used for masks, such as with
    the :cdata:`NPY_ITER_ARRAYMASK` iterator flag. This is equivalent
    to :cdata:`NPY_UINT8`.

.. cvar:: NPY_DEFAULT_TYPE

    The default type to use when no dtype is explicitly specified, for
    example when calling np.zero(shape). This is equivalent to
    :cdata:`NPY_DOUBLE`.

Other useful related constants are

.. cvar:: NPY_NTYPES

    The total number of built-in NumPy types. The enumeration covers
    the range from 0 to NPY_NTYPES-1.

.. cvar:: NPY_NOTYPE

    A signal value guaranteed not to be a valid type enumeration number.

.. cvar:: NPY_USERDEF

    The start of type numbers used for Custom Data types.

The various character codes indicating certain types are also part of
an enumerated list. References to type characters (should they be
needed at all) should always use these enumerations. The form of them
is :cdata:`NPY_{NAME}LTR` where ``{NAME}`` can be

    **BOOL**, **BYTE**, **UBYTE**, **SHORT**, **USHORT**, **INT**,
    **UINT**, **LONG**, **ULONG**, **LONGLONG**, **ULONGLONG**,
    **HALF**, **FLOAT**, **DOUBLE**, **LONGDOUBLE**, **CFLOAT**,
    **CDOUBLE**, **CLONGDOUBLE**, **DATETIME**, **TIMEDELTA**,
    **OBJECT**, **STRING**, **VOID**

    **INTP**, **UINTP**

    **GENBOOL**, **SIGNED**, **UNSIGNED**, **FLOATING**, **COMPLEX**

The latter group of ``{NAME}s`` corresponds to letters used in the array
interface typestring specification.


Defines
-------

Max and min values for integers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cvar:: NPY_MAX_INT{bits}

.. cvar:: NPY_MAX_UINT{bits}

.. cvar:: NPY_MIN_INT{bits}

    These are defined for ``{bits}`` = 8, 16, 32, 64, 128, and 256 and provide
    the maximum (minimum) value of the corresponding (unsigned) integer
    type. Note: the actual integer type may not be available on all
    platforms (i.e. 128-bit and 256-bit integers are rare).

.. cvar:: NPY_MIN_{type}

    This is defined for ``{type}`` = **BYTE**, **SHORT**, **INT**,
    **LONG**, **LONGLONG**, **INTP**

.. cvar:: NPY_MAX_{type}

    This is defined for all defined for ``{type}`` = **BYTE**, **UBYTE**,
    **SHORT**, **USHORT**, **INT**, **UINT**, **LONG**, **ULONG**,
    **LONGLONG**, **ULONGLONG**, **INTP**, **UINTP**


Number of bits in data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All :cdata:`NPY_SIZEOF_{CTYPE}` constants have corresponding
:cdata:`NPY_BITSOF_{CTYPE}` constants defined. The :cdata:`NPY_BITSOF_{CTYPE}`
constants provide the number of bits in the data type.  Specifically,
the available ``{CTYPE}s`` are

    **BOOL**, **CHAR**, **SHORT**, **INT**, **LONG**,
    **LONGLONG**, **FLOAT**, **DOUBLE**, **LONGDOUBLE**


Bit-width references to enumerated typenums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the numeric data types (integer, floating point, and complex)
have constants that are defined to be a specific enumerated type
number. Exactly which enumerated type a bit-width type refers to is
platform dependent. In particular, the constants available are
:cdata:`PyArray_{NAME}{BITS}` where ``{NAME}`` is **INT**, **UINT**,
**FLOAT**, **COMPLEX** and ``{BITS}`` can be 8, 16, 32, 64, 80, 96, 128,
160, 192, 256, and 512.  Obviously not all bit-widths are available on
all platforms for all the kinds of numeric types. Commonly 8-, 16-,
32-, 64-bit integers; 32-, 64-bit floats; and 64-, 128-bit complex
types are available.


Integer that can hold a pointer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constants **NPY_INTP** and **NPY_UINTP** refer to an
enumerated integer type that is large enough to hold a pointer on the
platform. Index arrays should always be converted to **NPY_INTP**
, because the dimension of the array is of type npy_intp.


C-type names
------------

There are standard variable types for each of the numeric data types
and the bool data type. Some of these are already available in the
C-specification. You can create variables in extension code with these
types.


Boolean
^^^^^^^

.. ctype:: npy_bool

    unsigned char; The constants :cdata:`NPY_FALSE` and
    :cdata:`NPY_TRUE` are also defined.


(Un)Signed Integer
^^^^^^^^^^^^^^^^^^

Unsigned versions of the integers can be defined by pre-pending a 'u'
to the front of the integer name.

.. ctype:: npy_(u)byte

    (unsigned) char

.. ctype:: npy_(u)short

    (unsigned) short

.. ctype:: npy_(u)int

    (unsigned) int

.. ctype:: npy_(u)long

    (unsigned) long int

.. ctype:: npy_(u)longlong

    (unsigned long long int)

.. ctype:: npy_(u)intp

    (unsigned) Py_intptr_t (an integer that is the size of a pointer on
    the platform).


(Complex) Floating point
^^^^^^^^^^^^^^^^^^^^^^^^

.. ctype:: npy_(c)float

    float

.. ctype:: npy_(c)double

    double

.. ctype:: npy_(c)longdouble

    long double

complex types are structures with **.real** and **.imag** members (in
that order).


Bit-width names
^^^^^^^^^^^^^^^

There are also typedefs for signed integers, unsigned integers,
floating point, and complex floating point types of specific bit-
widths. The available type names are

    :ctype:`npy_int{bits}`, :ctype:`npy_uint{bits}`, :ctype:`npy_float{bits}`,
    and :ctype:`npy_complex{bits}`

where ``{bits}`` is the number of bits in the type and can be **8**,
**16**, **32**, **64**, 128, and 256 for integer types; 16, **32**
, **64**, 80, 96, 128, and 256 for floating-point types; and 32,
**64**, **128**, 160, 192, and 512 for complex-valued types. Which
bit-widths are available is platform dependent. The bolded bit-widths
are usually available on all platforms.


Printf Formatting
-----------------

For help in printing, the following strings are defined as the correct
format specifier in printf and related commands.

    :cdata:`NPY_LONGLONG_FMT`, :cdata:`NPY_ULONGLONG_FMT`,
    :cdata:`NPY_INTP_FMT`, :cdata:`NPY_UINTP_FMT`,
    :cdata:`NPY_LONGDOUBLE_FMT`
