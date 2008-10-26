Data Type API
=============

.. sectionauthor:: Travis E. Oliphant

The standard array can have 21 different data types (and has some
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

There is a list of enumerated types defined providing the basic 21
data types plus some useful generic names. Whenever the code requires
a type number, one of these enumerated types is requested. The types
are all called :cdata:`NPY_{NAME}` where ``{NAME}`` can be

    **BOOL**, **BYTE**, **UBYTE**, **SHORT**, **USHORT**, **INT**,
    **UINT**, **LONG**, **ULONG**, **LONGLONG**, **ULONGLONG**,
    **FLOAT**, **DOUBLE**, **LONGDOUBLE**, **CFLOAT**, **CDOUBLE**,
    **CLONGDOUBLE**, **OBJECT**, **STRING**, **UNICODE**, **VOID**

    **NTYPES**, **NOTYPE**, **USERDEF**, **DEFAULT_TYPE** 

The various character codes indicating certain types are also part of
an enumerated list. References to type characters (should they be
needed at all) should always use these enumerations. The form of them
is :cdata:`NPY_{NAME}LTR` where ``{NAME}`` can be 

    **BOOL**, **BYTE**, **UBYTE**, **SHORT**, **USHORT**, **INT**,
    **UINT**, **LONG**, **ULONG**, **LONGLONG**, **ULONGLONG**,
    **FLOAT**, **DOUBLE**, **LONGDOUBLE**, **CFLOAT**, **CDOUBLE**,
    **CLONGDOUBLE**, **OBJECT**, **STRING**, **VOID**

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

The constants **PyArray_INTP** and **PyArray_UINTP** refer to an
enumerated integer type that is large enough to hold a pointer on the
platform. Index arrays should always be converted to **PyArray_INTP**
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
