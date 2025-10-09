

Data type API
=============

.. sectionauthor:: Travis E. Oliphant

The standard array can have 25 different data types (and has some
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
    conventions.  Thus, :c:data:`NPY_FLOAT` picks up a 32-bit float in
    C, but :class:`numpy.float64` in Python corresponds to a 64-bit
    double. The bit-width names can be used in both Python and C for
    clarity.


Enumerated types
----------------

.. c:enum:: NPY_TYPES

    There is a list of enumerated types defined providing the basic 25
    data types plus some useful generic names. Whenever the code requires
    a type number, one of these enumerated types is requested. The types
    are all called ``NPY_{NAME}``:

    .. c:enumerator:: NPY_BOOL

        The enumeration value for the boolean type, stored as one byte.
        It may only be set to the values 0 and 1.

    .. c:enumerator:: NPY_BYTE
    .. c:enumerator:: NPY_INT8

        The enumeration value for an 8-bit/1-byte signed integer.

    .. c:enumerator:: NPY_SHORT
    .. c:enumerator:: NPY_INT16

        The enumeration value for a 16-bit/2-byte signed integer.

    .. c:enumerator:: NPY_INT
    .. c:enumerator:: NPY_INT32

        The enumeration value for a 32-bit/4-byte signed integer.

    .. c:enumerator:: NPY_LONG

        Equivalent to either NPY_INT or NPY_LONGLONG, depending on the
        platform.

    .. c:enumerator:: NPY_LONGLONG
    .. c:enumerator:: NPY_INT64

        The enumeration value for a 64-bit/8-byte signed integer.

    .. c:enumerator:: NPY_UBYTE
    .. c:enumerator:: NPY_UINT8

        The enumeration value for an 8-bit/1-byte unsigned integer.

    .. c:enumerator:: NPY_USHORT
    .. c:enumerator:: NPY_UINT16

        The enumeration value for a 16-bit/2-byte unsigned integer.

    .. c:enumerator:: NPY_UINT
    .. c:enumerator:: NPY_UINT32

        The enumeration value for a 32-bit/4-byte unsigned integer.

    .. c:enumerator:: NPY_ULONG

        Equivalent to either NPY_UINT or NPY_ULONGLONG, depending on the
        platform.

    .. c:enumerator:: NPY_ULONGLONG
    .. c:enumerator:: NPY_UINT64

        The enumeration value for a 64-bit/8-byte unsigned integer.

    .. c:enumerator:: NPY_HALF
    .. c:enumerator:: NPY_FLOAT16

        The enumeration value for a 16-bit/2-byte IEEE 754-2008 compatible floating
        point type.

    .. c:enumerator:: NPY_FLOAT
    .. c:enumerator:: NPY_FLOAT32

        The enumeration value for a 32-bit/4-byte IEEE 754 compatible floating
        point type.

    .. c:enumerator:: NPY_DOUBLE
    .. c:enumerator:: NPY_FLOAT64

        The enumeration value for a 64-bit/8-byte IEEE 754 compatible floating
        point type.

    .. c:enumerator:: NPY_LONGDOUBLE

        The enumeration value for a platform-specific floating point type which is
        at least as large as NPY_DOUBLE, but larger on many platforms.

    .. c:enumerator:: NPY_CFLOAT
    .. c:enumerator:: NPY_COMPLEX64

        The enumeration value for a 64-bit/8-byte complex type made up of
        two NPY_FLOAT values.

    .. c:enumerator:: NPY_CDOUBLE
    .. c:enumerator:: NPY_COMPLEX128

        The enumeration value for a 128-bit/16-byte complex type made up of
        two NPY_DOUBLE values.

    .. c:enumerator:: NPY_CLONGDOUBLE

        The enumeration value for a platform-specific complex floating point
        type which is made up of two NPY_LONGDOUBLE values.

    .. c:enumerator:: NPY_DATETIME

        The enumeration value for a data type which holds dates or datetimes with
        a precision based on selectable date or time units.

    .. c:enumerator:: NPY_TIMEDELTA

        The enumeration value for a data type which holds lengths of times in
        integers of selectable date or time units.

    .. c:enumerator:: NPY_STRING

        The enumeration value for null-padded byte strings of a selectable
        size. The strings have a fixed maximum size within a given array.

    .. c:enumerator:: NPY_UNICODE

        The enumeration value for UCS4 strings of a selectable size. The
        strings have a fixed maximum size within a given array.

    .. c:enumerator:: NPY_VSTRING

        The enumeration value for UTF-8 variable-width strings. Note that this
        dtype holds an array of references, with string data stored outside of
        the array buffer. Use the C API for working with numpy variable-width
        static strings to access the string data in each array entry.

        .. note::
            This DType is new-style and is not included in ``NPY_NTYPES_LEGACY``.

    .. c:enumerator:: NPY_OBJECT

        The enumeration value for references to arbitrary Python objects.

    .. c:enumerator:: NPY_VOID

        Primarily used to hold struct dtypes, but can contain arbitrary
        binary data.

    Some useful aliases of the above types are

    .. c:enumerator:: NPY_INTP

        The enumeration value for a signed integer of type ``Py_ssize_t``
        (same as ``ssize_t`` if defined). This is the type used by all
        arrays of indices.

        .. versionchanged:: 2.0
            Previously, this was the same as ``intptr_t`` (same size as a
            pointer).  In practice, this is identical except on very niche
            platforms.
            You can use the ``'p'`` character code for the pointer meaning.

    .. c:enumerator:: NPY_UINTP

        The enumeration value for an unsigned integer type that is identical
        to a ``size_t``.

        .. versionchanged:: 2.0
            Previously, this was the same as ``uintptr_t`` (same size as a
            pointer).  In practice, this is identical except on very niche
            platforms.
            You can use the ``'P'`` character code for the pointer meaning.

    .. c:enumerator:: NPY_MASK

        The enumeration value of the type used for masks, such as with
        the :c:data:`NPY_ITER_ARRAYMASK` iterator flag. This is equivalent
        to :c:data:`NPY_UINT8`.

    .. c:enumerator:: NPY_DEFAULT_TYPE

        The default type to use when no dtype is explicitly specified, for
        example when calling np.zero(shape). This is equivalent to
        :c:data:`NPY_DOUBLE`.

Other useful related constants are

.. c:macro:: NPY_NTYPES_LEGACY

    The number of built-in NumPy types written using the legacy DType
    system. New NumPy dtypes will be written using the new DType API and may not
    function in the same manner as legacy DTypes. Use this macro if you want to
    handle legacy DTypes using different code paths or if you do not want to
    update code that uses ``NPY_NTYPES_LEGACY`` and does not work correctly with new
    DTypes.

    .. note::
        Newly added DTypes such as ``NPY_VSTRING`` will not be counted
        in ``NPY_NTYPES_LEGACY``.

.. c:macro:: NPY_NOTYPE

    A signal value guaranteed not to be a valid type enumeration number.

.. c:macro:: NPY_USERDEF

    The start of type numbers used for legacy Custom Data types.
    New-style user DTypes currently are currently *not* assigned a type-number.

    .. note::
        The total number of user dtypes is limited to below ``NPY_VSTRING``.
        Higher numbers are reserved to future new-style DType use.

The various character codes indicating certain types are also part of
an enumerated list. References to type characters (should they be
needed at all) should always use these enumerations. The form of them
is ``NPY_{NAME}LTR`` where ``{NAME}`` can be

    **BOOL**, **BYTE**, **UBYTE**, **SHORT**, **USHORT**, **INT**,
    **UINT**, **LONG**, **ULONG**, **LONGLONG**, **ULONGLONG**,
    **HALF**, **FLOAT**, **DOUBLE**, **LONGDOUBLE**, **CFLOAT**,
    **CDOUBLE**, **CLONGDOUBLE**, **DATETIME**, **TIMEDELTA**,
    **OBJECT**, **STRING**, **UNICODE**, **VSTRING**, **VOID**

    **INTP**, **UINTP**

    **GENBOOL**, **SIGNED**, **UNSIGNED**, **FLOATING**, **COMPLEX**

The latter group of ``{NAME}s`` corresponds to letters used in the array
interface typestring specification.


Defines
-------

Max and min values for integers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NPY_MAX_INT{bits}``, ``NPY_MAX_UINT{bits}``, ``NPY_MIN_INT{bits}``
    These are defined for ``{bits}`` = 8, 16, 32, 64, 128, and 256 and provide
    the maximum (minimum) value of the corresponding (unsigned) integer
    type. Note: the actual integer type may not be available on all
    platforms (i.e. 128-bit and 256-bit integers are rare).

``NPY_MIN_{type}``
    This is defined for ``{type}`` = **BYTE**, **SHORT**, **INT**,
    **LONG**, **LONGLONG**, **INTP**

``NPY_MAX_{type}``
    This is defined for all defined for ``{type}`` = **BYTE**, **UBYTE**,
    **SHORT**, **USHORT**, **INT**, **UINT**, **LONG**, **ULONG**,
    **LONGLONG**, **ULONGLONG**, **INTP**, **UINTP**


Number of bits in data types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ``NPY_SIZEOF_{CTYPE}`` constants have corresponding
``NPY_BITSOF_{CTYPE}`` constants defined. The ``NPY_BITSOF_{CTYPE}``
constants provide the number of bits in the data type.  Specifically,
the available ``{CTYPE}s`` are

    **BOOL**, **CHAR**, **SHORT**, **INT**, **LONG**,
    **LONGLONG**, **FLOAT**, **DOUBLE**, **LONGDOUBLE**


Bit-width references to enumerated typenums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the numeric data types (integer, floating point, and complex)
have constants that are defined to be a specific enumerated type
number. Exactly which enumerated type a bit-width type refers to is
platform dependent. In particular, the constants available are
``PyArray_{NAME}{BITS}`` where ``{NAME}`` is **INT**, **UINT**,
**FLOAT**, **COMPLEX** and ``{BITS}`` can be 8, 16, 32, 64, 80, 96, 128,
160, 192, 256, and 512.  Obviously not all bit-widths are available on
all platforms for all the kinds of numeric types. Commonly 8-, 16-,
32-, 64-bit integers; 32-, 64-bit floats; and 64-, 128-bit complex
types are available.


Further integer aliases
~~~~~~~~~~~~~~~~~~~~~~~

The constants **NPY_INTP** and **NPY_UINTP** refer to an ``Py_ssize_t``
and ``size_t``.
Although in practice normally true, these types are strictly speaking not
pointer sized and the character codes ``'p'`` and ``'P'`` can be used for
pointer sized integers.
(Before NumPy 2, ``intp`` was pointer size, but this almost never matched
the actual use, which is the reason for the name.)

Since NumPy 2, **NPY_DEFAULT_INT** is additionally defined.
The value of the macro is runtime dependent:  Since NumPy 2, it maps to
``NPY_INTP`` while on earlier versions it maps to ``NPY_LONG``.

C-type names
------------

There are standard variable types for each of the numeric data types
and the bool data type. Some of these are already available in the
C-specification. You can create variables in extension code with these
types.


Boolean
~~~~~~~

.. c:type:: npy_bool

    unsigned char; The constants :c:data:`NPY_FALSE` and
    :c:data:`NPY_TRUE` are also defined.


(Un)Signed Integer
~~~~~~~~~~~~~~~~~~

Unsigned versions of the integers can be defined by prepending a 'u'
to the front of the integer name.

.. c:type:: npy_byte

    char

.. c:type:: npy_ubyte

    unsigned char

.. c:type:: npy_short

    short

.. c:type:: npy_ushort

    unsigned short

.. c:type:: npy_int

    int

.. c:type:: npy_uint

    unsigned int

.. c:type:: npy_int16

    16-bit integer

.. c:type:: npy_uint16

    16-bit unsigned integer

.. c:type:: npy_int32

    32-bit integer

.. c:type:: npy_uint32

    32-bit unsigned integer

.. c:type:: npy_int64

    64-bit integer

.. c:type:: npy_uint64

    64-bit unsigned integer

.. c:type:: npy_long

    long int

.. c:type:: npy_ulong

    unsigned long int

.. c:type:: npy_longlong

    long long int

.. c:type:: npy_ulonglong

    unsigned long long int

.. c:type:: npy_intp

    ``Py_ssize_t`` (a signed integer with the same size as the C ``size_t``).
    This is the correct integer for lengths or indexing.  In practice this is
    normally the size of a pointer, but this is not guaranteed.

    .. note::
        Before NumPy 2.0, this was the same as ``Py_intptr_t``.
        While a better match, this did not match actual usage in practice.
        On the Python side, we still support ``np.dtype('p')`` to fetch a dtype
        compatible with storing pointers, while ``n`` is the correct character
        for the ``ssize_t``.

.. c:type:: npy_uintp

    The C ``size_t``/``Py_size_t``.


(Complex) Floating point
~~~~~~~~~~~~~~~~~~~~~~~~

.. c:type:: npy_half

    16-bit float

.. c:type:: npy_float

    32-bit float

.. c:type:: npy_cfloat

    32-bit complex float

.. c:type:: npy_double

    64-bit double

.. c:type:: npy_cdouble

    64-bit complex double

.. c:type:: npy_longdouble

    long double

.. c:type:: npy_clongdouble

    long complex double

complex types are structures with **.real** and **.imag** members (in
that order).


Bit-width names
~~~~~~~~~~~~~~~

There are also typedefs for signed integers, unsigned integers,
floating point, and complex floating point types of specific bit-
widths. The available type names are

    ``npy_int{bits}``, ``npy_uint{bits}``, ``npy_float{bits}``,
    and ``npy_complex{bits}``

where ``{bits}`` is the number of bits in the type and can be **8**,
**16**, **32**, **64**, 128, and 256 for integer types; 16, **32**
, **64**, 80, 96, 128, and 256 for floating-point types; and 32,
**64**, **128**, 160, 192, and 512 for complex-valued types. Which
bit-widths are available is platform dependent. The bolded bit-widths
are usually available on all platforms.


Time and timedelta
~~~~~~~~~~~~~~~~~~

.. c:type:: npy_datetime

    date or datetime (alias of :c:type:`npy_int64`)

.. c:type:: npy_timedelta

    length of time (alias of :c:type:`npy_int64`)


Printf formatting
-----------------

For help in printing, the following strings are defined as the correct
format specifier in printf and related commands.

.. c:macro:: NPY_LONGLONG_FMT

.. c:macro:: NPY_ULONGLONG_FMT

.. c:macro:: NPY_INTP_FMT

.. c:macro:: NPY_UINTP_FMT

.. c:macro:: NPY_LONGDOUBLE_FMT
  
