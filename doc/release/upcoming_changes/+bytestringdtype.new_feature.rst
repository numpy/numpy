Variable-width bytestring arrays
--------------------------------
`numpy.dtypes.ByteStringDType` stores variable-width byte sequences, preserving
embedded and trailing NUL bytes and accepting data that is not valid text.
It accepts Python `bytes` and its subclasses, with optional missing-data
support through ``na_object``. Nonmissing scalar indexing returns
`numpy.vbytes`, a subclass of `bytes` and `numpy.generic`.

Request the dtype explicitly when constructing arrays from Python bytes;
ordinary bytes input continues to infer fixed-width `numpy.bytes_` arrays.
A subset of `numpy.strings` operations supports the new dtype.
`numpy.strings.encode` can convert `numpy.dtypes.StringDType` input to
ByteStringDType with ``dtype=np.dtypes.ByteStringDType()``, and
`numpy.strings.decode` converts it back. These variable-width conversions
support UTF-8 with strict error handling.

See :ref:`bytestringdtype` for supported operations, casts, and storage
limitations.
