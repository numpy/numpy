.. _basics.types:

**********
Data types
**********

.. seealso:: :ref:`Data type objects <arrays.dtypes>`

Array types and conversions between types
=========================================

NumPy supports a much greater variety of numerical types than Python does.
This section shows which are available, and how to modify an array's data-type.

NumPy numerical types are instances of `numpy.dtype` (data-type) objects, each
having unique characteristics.  Once you have imported NumPy using ``import
numpy as np`` you can create arrays with a specified dtype using the scalar
types in the numpy top-level API, e.g. `numpy.bool`, `numpy.float32`, etc.

These scalar types as arguments to the dtype keyword that many numpy functions
or methods accept. For example::

    >>> z = np.arange(3, dtype=np.uint8)
    >>> z
    array([0, 1, 2], dtype=uint8)

Array types can also be referred to by character codes, for example::

  >>> np.array([1, 2, 3], dtype='f')
  array([1.,  2.,  3.], dtype=float32)
  >>> np.array([1, 2, 3], dtype='d')
  array([1.,  2.,  3.], dtype=float64)

See :ref:`arrays.dtypes.constructing` for more information about specifying and
constructing data type objects, including how to specify parameters like the
byte order.

To determine the type of an array, look at the dtype attribute::

    >>> z.dtype
    dtype('uint8')

dtype objects also contain information about the type, such as its bit-width
and its byte-order.  The data type can also be used indirectly to query
properties of the type, such as whether it is an integer::

    >>> d = np.dtype(np.int64)
    >>> d
    dtype('int64')

    >>> np.issubdtype(d, np.integer)
    True

    >>> np.issubdtype(d, np.floating)
    False

To convert the type of an array, use the .astype() method. For example::

    >>> z.astype(np.float64)                 #doctest: +NORMALIZE_WHITESPACE
    array([0.,  1.,  2.])

Note that, above, we could have used the *Python* float object as a dtype
instead of `numpy.float64`.  NumPy knows that
:class:`int` refers to `numpy.int_`, :class:`bool` means
`numpy.bool`, that :class:`float` is `numpy.float64` and
:class:`complex` is `numpy.complex128`.  The other data-types do not have
Python equivalents.

Sometimes the conversion can overflow, for instance when converting a `numpy.int64` value
300 to `numpy.int8`. NumPy follows C casting rules, so that value would overflow and
become 44 ``(300 - 256)``. If you wish to avoid such overflows, you can specify that the
overflow action fail by using ``same_value`` for the ``casting`` argument (see also
:ref:`overflow-errors`)::

    >>> z.astype(np.float64, casting="same_value")   #doctest: +NORMALIZE_WHITESPACE
    array([0.,  1.,  2.])


Numerical Data Types
--------------------

There are 5 basic numerical types representing booleans (``bool``), integers
(``int``), unsigned integers (``uint``) floating point (``float``) and
``complex``. A basic numerical type name combined with a numeric bitsize defines
a concrete type.  The bitsize is the number of bits that are needed to represent
a single value in memory. For example, `numpy.float64` is a 64 bit
floating point data type. Some types, such as `numpy.int_` and
`numpy.intp`, have differing bitsizes, dependent on the platforms
(e.g. 32-bit vs. 64-bit CPU architectures).  This should be taken into account
when interfacing with low-level code (such as C or Fortran) where the raw memory
is addressed.

Data Types for Strings and Bytes
--------------------------------

In addition to numerical types, NumPy also supports storing unicode strings, via
the `numpy.str_` dtype (``U`` character code), null-terminated byte sequences via
`numpy.bytes_` (``S`` character code), and arbitrary byte sequences, via
`numpy.void` (``V`` character code).

All of the above are *fixed-width* data types. They are parameterized by a
width, in either bytes or unicode points, that a single data element in the
array must fit inside. This means that storing an array of byte sequences or
strings using this dtype requires knowing or calculating the sizes of the
longest text or byte sequence in advance.

As an example, we can create an array storing the words ``"hello"`` and
``"world!"``::

  >>> np.array(["hello", "world!"])
  array(['hello', 'world!'], dtype='<U6')

Here the data type is detected as a unicode string that is a maximum of 6 code
points long, enough to store both entries without truncation. If we specify a
shorter or longer data type, the string is either truncated or zero-padded to
fit in the specified width::

  >>> np.array(["hello", "world!"], dtype="U5")
  array(['hello', 'world'], dtype='<U5')
  >>> np.array(["hello", "world!"], dtype="U7")
  array(['hello', 'world!'], dtype='<U7')

We can see the zero-padding a little more clearly if we use the bytes data
type and ask NumPy to print out the bytes in the array buffer::

  >>> np.array(["hello", "world"], dtype="S7").tobytes()
  b'hello\x00\x00world\x00\x00'

Each entry is padded with two extra null bytes. Note however that NumPy cannot
tell the difference between intentionally stored trailing nulls and padding
nulls::

  >>> x = [b"hello\0\0", b"world"]
  >>> a = np.array(x, dtype="S7")
  >>> print(a[0])
  b"hello"
  >>> a[0] == x[0]
  False

If you need to store and round-trip any trailing null bytes, you will need to
use an unstructured void data type::

  >>> a = np.array(x, dtype="V7")
  >>> a
  array([b'\x68\x65\x6C\x6C\x6F\x00\x00', b'\x77\x6F\x72\x6C\x64\x00\x00'],
        dtype='|V7')
  >>> a[0] == np.void(x[0])
  True

Advanced types, not listed above, are explored in section
:ref:`structured_arrays`.

.. _canonical-python-and-c-types:

Relationship Between NumPy Data Types and C Data Types
======================================================

NumPy provides both bit sized type names and names based on the names of C types.
Since the definition of C types are platform dependent, this means the explicitly
bit sized should be preferred to avoid platform-dependent behavior in programs
using NumPy.

To ease integration with C code, where it is more natural to refer to
platform-dependent C types, NumPy also provides type aliases that correspond
to the C types for the platform. Some dtypes have trailing underscore to avoid
confusion with builtin python type names, such as `numpy.bool_`.

.. list-table::
    :header-rows: 1

    * - Canonical Python API name
      - Python API "C-like" name
      - Actual C type
      - Description

    * - `numpy.bool` or `numpy.bool_`
      - N/A
      - ``bool`` (defined in ``stdbool.h``)
      - Boolean (True or False) stored as a byte.

    * - `numpy.int8`
      - `numpy.byte`
      - ``signed char``
      - Platform-defined integer type with 8 bits.

    * - `numpy.uint8`
      - `numpy.ubyte`
      - ``unsigned char``
      - Platform-defined integer type with 8 bits without sign.

    * - `numpy.int16`
      - `numpy.short`
      - ``short``
      - Platform-defined integer type with 16 bits.

    * - `numpy.uint16`
      - `numpy.ushort`
      - ``unsigned short``
      - Platform-defined integer type with 16 bits without sign.

    * - `numpy.int32`
      - `numpy.intc`
      - ``int``
      - Platform-defined integer type with 32 bits.

    * - `numpy.uint32`
      - `numpy.uintc`
      - ``unsigned int``
      - Platform-defined integer type with 32 bits without sign.

    * - `numpy.intp`
      - N/A
      - ``ssize_t``/``Py_ssize_t``
      - Platform-defined integer of size ``size_t``; used e.g. for sizes.

    * - `numpy.uintp`
      - N/A
      - ``size_t``
      - Platform-defined integer type capable of storing the maximum
        allocation size.

    * - N/A
      - ``'p'``
      - ``intptr_t``
      - Guaranteed to hold pointers. Character code only (Python and C).

    * - N/A
      - ``'P'``
      - ``uintptr_t``
      - Guaranteed to hold pointers without sign. Character code only (Python and C).

    * - `numpy.int32` or `numpy.int64`
      - `numpy.long`
      - ``long``
      - Platform-defined integer type with at least 32 bits.

    * - `numpy.uint32` or `numpy.uint64`
      - `numpy.ulong`
      - ``unsigned long``
      - Platform-defined integer type with at least 32 bits without sign.

    * - N/A
      - `numpy.longlong`
      - ``long long``
      - Platform-defined integer type with at least 64 bits.

    * - N/A
      - `numpy.ulonglong`
      - ``unsigned long long``
      - Platform-defined integer type with at least 64 bits without sign.

    * - `numpy.float16`
      - `numpy.half`
      - N/A
      - Half precision float:
        sign bit, 5 bits exponent, 10 bits mantissa.

    * - `numpy.float32`
      - `numpy.single`
      - ``float``
      - Platform-defined single precision float:
        typically sign bit, 8 bits exponent, 23 bits mantissa.

    * - `numpy.float64`
      - `numpy.double`
      - ``double``
      - Platform-defined double precision float:
        typically sign bit, 11 bits exponent, 52 bits mantissa.

    * - ``numpy.float96`` or `numpy.float128`
      - `numpy.longdouble`
      - ``long double``
      - Platform-defined extended-precision float.

    * - `numpy.complex64`
      - `numpy.csingle`
      - ``float complex``
      - Complex number, represented by two single-precision floats (real and imaginary components).

    * - `numpy.complex128`
      - `numpy.cdouble`
      - ``double complex``
      - Complex number, represented by two double-precision floats (real and imaginary components).

    * - ``numpy.complex192`` or `numpy.complex256`
      - `numpy.clongdouble`
      - ``long double complex``
      - Complex number, represented by two extended-precision floats (real and imaginary components).

Since many of these have platform-dependent definitions, a set of fixed-size
aliases are provided (See :ref:`sized-aliases`).

Array scalars
=============

NumPy generally returns elements of arrays as array scalars (a scalar
with an associated dtype).  Array scalars differ from Python scalars, but
for the most part they can be used interchangeably (the primary
exception is for versions of Python older than v2.x, where integer array
scalars cannot act as indices for lists and tuples).  There are some
exceptions, such as when code requires very specific attributes of a scalar
or when it checks specifically whether a value is a Python scalar. Generally,
problems are easily fixed by explicitly converting array scalars
to Python scalars, using the corresponding Python type function
(e.g., :class:`int`, :class:`float`, :class:`complex`, :class:`str`).

The primary advantage of using array scalars is that
they preserve the array type (Python may not have a matching scalar type
available, e.g. ``int16``).  Therefore, the use of array scalars ensures
identical behaviour between arrays and scalars, irrespective of whether the
value is inside an array or not.  NumPy scalars also have many of the same
methods arrays do.

.. _overflow-errors:

Overflow errors
===============

The fixed size of NumPy numeric types may cause overflow errors when a value
requires more memory than available in the data type. For example,
`numpy.power` evaluates ``100 ** 9`` correctly for 64-bit integers,
but gives -1486618624 (incorrect) for a 32-bit integer.

    >>> np.power(100, 9, dtype=np.int64)
    1000000000000000000
    >>> np.power(100, 9, dtype=np.int32)
    np.int32(-1486618624)

The behaviour of NumPy and Python integer types differs significantly for
integer overflows and may confuse users expecting NumPy integers to behave
similar to Python's :class:`int`. Unlike NumPy, the size of Python's
:class:`int` is flexible. This means Python integers may expand to accommodate
any integer and will not overflow.

NumPy provides `numpy.iinfo` and `numpy.finfo` to verify the
minimum or maximum values of NumPy integer and floating point values
respectively ::

    >>> np.iinfo(int) # Bounds of the default integer on this system.
    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)
    >>> np.iinfo(np.int32) # Bounds of a 32-bit integer
    iinfo(min=-2147483648, max=2147483647, dtype=int32)
    >>> np.iinfo(np.int64) # Bounds of a 64-bit integer
    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)

If 64-bit integers are still too small the result may be cast to a
floating point number. Floating point numbers offer a larger, but inexact,
range of possible values.

    >>> np.power(100, 100, dtype=np.int64) # Incorrect even with 64-bit int
    0
    >>> np.power(100, 100, dtype=np.float64)
    1e+200

Floating point precision
========================

Many functions in NumPy, especially those in `numpy.linalg`, involve floating-point
arithmetic, which can introduce small inaccuracies due to the way computers 
represent decimal numbers. For instance, when performing basic arithmetic operations 
involving floating-point numbers:

    >>> 0.3 - 0.2 - 0.1  # This does not equal 0 due to floating-point precision
    -2.7755575615628914e-17

To handle such cases, it's advisable to use functions like `np.isclose` to compare 
values, rather than checking for exact equality:

    >>> np.isclose(0.3 - 0.2 - 0.1, 0, rtol=1e-05)  # Check for closeness to 0
    True

In this example, `np.isclose` accounts for the minor inaccuracies that occur in 
floating-point calculations by applying a relative tolerance, ensuring that results
within a small threshold are considered close.

For information about precision in calculations, see `Floating-Point Arithmetic <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>`_.


Extended precision
==================

Python's floating-point numbers are usually 64-bit floating-point numbers,
nearly equivalent to `numpy.float64`. In some unusual situations it may be
useful to use floating-point numbers with more precision. Whether this
is possible in numpy depends on the hardware and on the development
environment: specifically, x86 machines provide hardware floating-point
with 80-bit precision, and while most C compilers provide this as their
``long double`` type, MSVC (standard for Windows builds) makes
``long double`` identical to ``double`` (64 bits). NumPy makes the
compiler's ``long double`` available as `numpy.longdouble` (and
``np.clongdouble`` for the complex numbers). You can find out what your
numpy provides with ``np.finfo(np.longdouble)``.

NumPy does not provide a dtype with more precision than C's
``long double``; in particular, the 128-bit IEEE quad precision
data type (FORTRAN's ``REAL*16``) is not available.

For efficient memory alignment, `numpy.longdouble` is usually stored
padded with zero bits, either to 96 or 128 bits. Which is more efficient
depends on hardware and development environment; typically on 32-bit
systems they are padded to 96 bits, while on 64-bit systems they are
typically padded to 128 bits. ``np.longdouble`` is padded to the system
default; ``np.float96`` and ``np.float128`` are provided for users who
want specific padding. In spite of the names, ``np.float96`` and
``np.float128`` provide only as much precision as ``np.longdouble``,
that is, 80 bits on most x86 machines and 64 bits in standard
Windows builds.

Be warned that even if `numpy.longdouble` offers more precision than
python :class:`float`, it is easy to lose that extra precision, since
python often forces values to pass through ``float``. For example,
the ``%`` formatting operator requires its arguments to be converted
to standard python types, and it is therefore impossible to preserve
extended precision even if many decimal places are requested. It can
be useful to test your code with the value
``1 + np.finfo(np.longdouble).eps``.
