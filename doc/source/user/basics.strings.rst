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
numpy provides `numpy.dtypes.StringDType`. Below we describe how to
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

Since this is a one-byte encoding, the byteorder is `'|'` (not
applicable), and the data type detected is a maximum 5 character
bytestring.

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
types, ``StringDType`` does not store the string data in the "main"
``ndarray`` data buffer. Instead, the array buffer is used to store
metadata about where the string data are stored in memory. This
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

Optionally, you can pass create an instance of ``StringDType`` with
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
sentinel, then string operations will raise an error:

  >>> dt = StringDType(na_object=None)
  >>> arr = np.array(["this array has", None, "as an entry"])
  >>> np.sort(arr)
  Traceback (most recent call last):
  ...
  TypeError: '<' not supported between instances of 'NoneType' and 'str'

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
columnar data format with a known maximum string length.

In all cases, casting to a fixed-width string requires specifying the
maximum allowed string length::

   >>> arr = np.array(["hello", "world"], dtype=StringDType())
   >>> arr.astype(np.str_)  # doctest: +IGNORE_EXCEPTION_DETAIL
   Traceback (most recent call last):
   ...
   TypeError: Casting from StringDType to a fixed-width dtype with an
   unspecified size is not currently supported, specify an explicit
   size for the output dtype instead.

   The above exception was the direct cause of the following
   exception:

   TypeError: cannot cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>.
   >>> arr.astype("U5")
   array(['hello', 'world'], dtype='<U5')
   
The `numpy.bytes_` cast is most useful for string data that is known
to contain only ASCII characters, as characters outside this range
cannot be represented in a single byte in the UTF-8 encoding and are
rejected.

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
