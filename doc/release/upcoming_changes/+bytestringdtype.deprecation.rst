Default output dtype of encoding StringDType arrays
---------------------------------------------------
Calling `numpy.strings.encode` on `numpy.dtypes.StringDType` input without
specifying ``dtype`` emits a ``FutureWarning``. It currently returns
fixed-width bytes, but a future release will return
`numpy.dtypes.ByteStringDType`.

Pass ``dtype=np.bytes_`` to retain the fixed-width result without a warning,
or ``dtype=np.dtypes.ByteStringDType()`` to select the variable-width result
now. The variable-width path supports only UTF-8 with strict error handling;
the fixed-width path retains support for other codecs and error handlers.
Encoding fixed-width Unicode arrays is unchanged.
