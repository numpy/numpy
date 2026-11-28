``numpy.sinc`` follows ufunc type resolution
--------------------------------------------
Since `numpy.sinc` is now a ufunc, it resolves dtypes the same way as
`numpy.sin`. Boolean and 8-bit integer inputs now return ``float16`` and
16-bit integer inputs ``float32``, instead of always ``float64``. Array
subclasses are now preserved, and ``numpy.sinc`` is overridden through
``__array_ufunc__`` instead of ``__array_function__``.
