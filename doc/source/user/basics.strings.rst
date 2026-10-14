.. _basics.strings:

****************************************
Working with Arrays of Strings And Bytes
****************************************

While NumPy is primarily a numerical library, it is often convenient
to work with NumPy arrays of strings or bytes. The two most common
use cases are:

* Working with data loaded or memory-mapped from a data file,
  where one or more of the fields in the data is a string or
  bytestring, and the maximum length of the field is known
  ahead of time. This often is used for a name or label field.
* Using NumPy indexing and broadcasting with arrays of Python
  strings of unknown length, which may or may not have data
  defined for every value.

For the first use case, NumPy provides the fixed-width `numpy.void`,
`numpy.str_` and `numpy.bytes_` data types. For the second use case,
numpy provides `numpy.dtypes.StringDType` for text and
`numpy.dtypes.ByteStringDType` for bytes. Below we describe how to
work with both fixed-width and variable-width string arrays, how to
convert between the two representations, and provide some advice for
most efficiently working with string data in NumPy.

Fixed-width data types
======================

Before NumPy 2.0, the fixed-width `numpy.str_`, `numpy.bytes_`, and
`numpy.void` data types were the only types available for working
with strings and bytestrings in NumPy. For this reason, they are used
as the default dtype for strings and bytestrings, respectively:

   >>> np.array(["hello", "world"])
   array(['hello', 'world'], dtype='<U5')

Here the detected data type is ``'<U5'``, or little-endian unicode
string data, with a maximum length of 5 unicode code points.

Similarly for bytestrings:

   >>> np.array([b"hello", b"world"])
   array([b'hello', b'world'], dtype='|S5')

Since this stores individual bytes, the byteorder is `'|'` (not
applicable), and the data type detected is a maximum 5-byte
bytestring.

Fixed-width bytestrings are padded with NUL bytes up to the itemsize,
and trailing NUL bytes are stripped when retrieving a scalar:

   >>> np.array([b"x\x00"], dtype="S2")
   array([b'x'], dtype='|S2')

You can also use `numpy.void` to represent bytestrings:

   >>> np.array([b"hello", b"world"]).astype(np.void)
   array([b'\x68\x65\x6C\x6C\x6F', b'\x77\x6F\x72\x6C\x64'], dtype='|V5')

This is most useful when working with byte streams that are not well
represented as bytestrings, and instead are better thought of as
collections of 8-bit integers.

.. _stringdtype:

Variable-width strings
======================

.. versionadded:: 2.0

.. note::

   `numpy.dtypes.StringDType` is a new addition to NumPy, implemented
   using the new support in NumPy for flexible user-defined data
   types and is not as extensively tested in production workflows as
   the older NumPy data types.

Often, real-world string data does not have a predictable length. In
these cases it is awkward to use fixed-width strings, since storing
all the data without truncation requires knowing the length of the
longest string one would like to store in the array before the array
is created.

To support situations like this, NumPy provides
`numpy.dtypes.StringDType`, which stores variable-width string data
in a UTF-8 encoding in a NumPy array:

  >>> from numpy.dtypes import StringDType
  >>> data = ["this is a longer string", "short string"]
  >>> arr = np.array(data, dtype=StringDType())
  >>> arr
  array(['this is a longer string', 'short string'], dtype=StringDType())

Note that unlike fixed-width strings, ``StringDType`` is not parameterized by
the maximum length of an array element, arbitrarily long or short strings can
live in the same array without needing to reserve storage for padding bytes in
the short strings.

Also note that unlike fixed-width strings and most other NumPy data
types, ``StringDType`` does not always store the string data in the "main"
``ndarray`` data buffer. Instead, the array buffer can store short strings or
metadata about where longer string data are stored in memory. This
difference means that code expecting the array buffer to contain
string data will not function correctly, and will need to be updated
to support ``StringDType``.

Missing data support
--------------------

Often string datasets are not complete, and a special label is needed
to indicate that a value is missing. By default ``StringDType`` does
not have any special support for missing values, besides the fact
that empty strings are used to populate empty arrays:

  >>> np.empty(3, dtype=StringDType())
  array(['', '', ''], dtype=StringDType())

Optionally, you can create an instance of ``StringDType`` with
support for missing values by passing ``na_object`` as a keyword
argument for the initializer:

  >>> dt = StringDType(na_object=None)
  >>> arr = np.array(["this array has", None, "as an entry"], dtype=dt)
  >>> arr
  array(['this array has', None, 'as an entry'],
        dtype=StringDType(na_object=None))
  >>> arr[1] is None
  True
  
The ``na_object`` can be any arbitrary python object.
Common choices are `numpy.nan`, ``float('nan')``, ``None``, an object
specifically intended to represent missing data like ``pandas.NA``,
or a (hopefully) unique string like ``"__placeholder__"``.

NumPy has special handling for NaN-like sentinels and string
sentinels.

NaN-like Missing Data Sentinels
+++++++++++++++++++++++++++++++

A NaN-like sentinel returns itself as the result of arithmetic
operations. This includes the python ``nan`` float and the Pandas
missing data sentinel ``pd.NA``. NaN-like sentinels inherit these
behaviors in string operations. This means that, for example, the
result of addition with any other string is the sentinel:

  >>> dt = StringDType(na_object=np.nan)
  >>> arr = np.array(["hello", np.nan, "world"], dtype=dt)
  >>> arr + arr
  array(['hellohello', nan, 'worldworld'], dtype=StringDType(na_object=nan))

Following the behavior of ``nan`` in float arrays, NaN-like sentinels
sort to the end of the array:

  >>> np.sort(arr)
  array(['hello', 'world', nan], dtype=StringDType(na_object=nan))

Comparisons also follow floating-point NaN semantics: equality and ordered
comparisons involving a NaN-like sentinel return ``False``, while inequality
returns ``True``.

  >>> arr != "hello"
  array([False,  True,  True])

String Missing Data Sentinels
+++++++++++++++++++++++++++++

A string missing data value is an instance of ``str`` or subtype of ``str``. If
such an array is passed to a string operation or a cast, "missing" entries are
treated as if they have a value given by the string sentinel. Comparison
operations similarly use the sentinel value directly for missing entries.

Other Sentinels
+++++++++++++++

Other objects, such as ``None`` are also supported as missing data
sentinels. If any missing data are present in an array using such a
sentinel, then operations such as sorting will raise an error:

  >>> dt = StringDType(na_object=None)
  >>> arr = np.array(["this array has", None, "as an entry"], dtype=dt)
  >>> np.sort(arr)
  Traceback (most recent call last):
  ...
  ValueError: Cannot compare null that is not a nan-like value

Coercing Non-strings
--------------------

By default, non-string data are coerced to strings:

  >>> np.array([1, object(), 3.4], dtype=StringDType())
  array(['1', '<object object at 0x7faa2497dde0>', '3.4'], dtype=StringDType())

If this behavior is not desired, an instance of the DType can be created that
disables string coercion by setting ``coerce=False`` in the initializer:

  >>> np.array([1, object(), 3.4], dtype=StringDType(coerce=False))
  Traceback (most recent call last):
  ...
  ValueError: StringDType only allows string data when string coercion is disabled.

This allows strict data validation in the same pass over the data NumPy uses to
create the array. Setting ``coerce=True`` recovers the default behavior allowing
coercion to strings.

Casting To and From Fixed-Width Strings
---------------------------------------

``StringDType`` supports round-trip casts between `numpy.str_`,
`numpy.bytes_`, and `numpy.void`. Casting to a fixed-width string is
most useful when strings need to be memory-mapped in an ndarray or
when a fixed-width string is needed for reading and writing to a
columnar data format with a known maximum string length. The
`numpy.bytes_` cast is most useful for string data that is known to
contain only ASCII characters, as characters outside this range cannot
be represented in a single byte in the UTF-8 encoding and are rejected.

When converting an array to a fixed-width string dtype with an
unspecified size using `numpy.ndarray.astype`, NumPy infers the size by
inspecting the array values, producing a dtype wide enough to store the
widest entry without truncation::

   >>> arr = np.array(["hello", "world!!"], dtype=StringDType())
   >>> arr.astype(np.str_)
   array(['hello', 'world!!'], dtype='<U7')
   >>> arr.astype("S")
   array([b'hello', b'world!!'], dtype='|S7')

An explicit size can still be passed, truncating entries that do not
fit::

   >>> arr.astype("U4")
   array(['hell', 'worl'], dtype='<U4')

Any valid unicode string can be cast to `numpy.str_`, although
since `numpy.str_` uses a 32-bit UCS4 encoding for all characters,
this will often waste memory for real-world textual data that can be
well-represented by a more memory-efficient encoding.

Additionally, any valid unicode string can be cast to `numpy.void`,
storing the UTF-8 bytes directly in the output array:

  >>> arr = np.array(["hello", "world"], dtype=StringDType())
  >>> arr.astype("V5")
  array([b'\x68\x65\x6C\x6C\x6F', b'\x77\x6F\x72\x6C\x64'], dtype='|V5')

Care must be taken to ensure that the output array has enough space
for the UTF-8 bytes in the string, since the size of a UTF-8
bytestream in bytes is not necessarily the same as the number of
characters in the string.

Conversions in the other direction, from a fixed-width string array to
``StringDType`` are ``"safe"`` casts. The bytes stored in a
`numpy.bytes_` array must be valid UTF-8, and a ``UnicodeDecodeError``
is raised during the cast if they are not.

.. _bytestringdtype:

Variable-width bytestrings
==========================

.. versionadded:: 2.6

`numpy.dtypes.ByteStringDType` is the bytes counterpart of
``StringDType``. It stores variable-width byte sequences exactly as
given and returns nonmissing values as `numpy.vbytes` scalars, a `bytes`
subclass, so values may contain or end in NUL bytes and may hold data that is not
valid text:

  >>> from numpy.dtypes import ByteStringDType
  >>> arr = np.array([b"x\x00", b"\xff\xfe", b""], dtype=ByteStringDType())
  >>> arr
  array([b'x\x00', b'\xff\xfe', b''], dtype=ByteStringDType())
  >>> arr[0]
  np.vbytes(b'x\x00')
  >>> arr.tolist()
  [b'x\x00', b'\xff\xfe', b'']
  >>> np.strings.str_len(arr)
  array([2, 2, 0])

For ordinary Python bytes, the dtype has to be requested explicitly;
``np.array`` still infers the fixed-width `numpy.bytes_` dtype for such data.
Only `bytes` and its subclasses such as `numpy.bytes_` are accepted as
nonmissing values; text is not encoded implicitly:

  >>> np.array(["x"], dtype=ByteStringDType())
  Traceback (most recent call last):
  ...
  TypeError: ByteStringDType only allows bytes data, got an instance of 'str'; convert text to bytes explicitly with str.encode(encoding)

Missing data works as for ``StringDType``, by passing ``na_object`` to
the constructor. There is no ``coerce`` option:

  >>> np.array([b"a", None], dtype=ByteStringDType(na_object=None))
  array([b'a', None], dtype=ByteStringDType(na_object=None))

``ByteStringDType`` promotes with `numpy.bytes_` but not with
`numpy.str_` or ``StringDType``. Array comparisons with ``==`` and ``!=``
follow Python: a bytestring never equals a string. Ordering the two, or
passing both to comparison ufuncs such as `numpy.equal`, raises
``TypeError``. The `numpy.strings` functions that support
``ByteStringDType`` operate on bytes in the same way as the
corresponding `bytes` methods, see :ref:`routines.strings` for the
list:

  >>> arr + np.array([b"1", b"2", b"3"], dtype="S1")
  array([b'x\x001', b'\xff\xfe2', b'3'], dtype=ByteStringDType())

Casting to and from fixed-width bytes
-------------------------------------

``ByteStringDType`` casts to and from `numpy.bytes_` and `numpy.void`
with no encoding validation. The cast from `numpy.bytes_` is safe;
since the fixed-width value is NUL-padded, its trailing NUL bytes are
not part of the result. Casting an array to an unsized `numpy.bytes_`
or `numpy.void` dtype infers the width from the widest entry, counted
in bytes; an explicit smaller width truncates:

  >>> arr = np.array([b"ab", b"\xff"], dtype=ByteStringDType())
  >>> arr.astype("S")
  array([b'ab', b'\xff'], dtype='|S2')
  >>> arr.astype("S1")
  array([b'a', b'\xff'], dtype='|S1')
  >>> np.array([b"x\x00"], dtype="S4").astype(ByteStringDType())
  array([b'x'], dtype=ByteStringDType())

`numpy.void` stores the bytes as given, and casting back keeps the
padding:

  >>> arr.astype("V3").astype(ByteStringDType())
  array([b'ab\x00', b'\xff\x00\x00'], dtype=ByteStringDType())

There are no casts between ``ByteStringDType`` and `numpy.str_`,
``StringDType``, or numeric types.

Converting between text and bytes
---------------------------------

Use `numpy.strings.encode` and `numpy.strings.decode` to convert
between ``StringDType`` and ``ByteStringDType``. For these conversions only
the UTF-8 encoding with ``errors="strict"`` is supported, and NUL bytes
are preserved:

  >>> text = np.array(["x\x00", "é"], dtype=StringDType())
  >>> data = np.strings.encode(text, "utf-8", dtype=ByteStringDType())
  >>> data
  array([b'x\x00', b'\xc3\xa9'], dtype=ByteStringDType())
  >>> np.strings.decode(data, "utf-8")
  array(['x\x00', 'é'], dtype=StringDType())

Called without ``dtype`` on ``StringDType`` input,
`numpy.strings.encode` returns the fixed-width `numpy.bytes_` result
and warns that the default will change to ``ByteStringDType``.
