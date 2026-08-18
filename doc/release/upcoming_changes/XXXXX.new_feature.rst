New ``axis`` keyword argument for `numpy.searchsorted`
------------------------------------------------------
`numpy.searchsorted` and ``ndarray.searchsorted`` now accept an ``axis``
keyword and no longer require the sorted array to be one-dimensional. Each
1-D slice along ``axis``, the last one by default or the flattened array when
``axis=None``, is searched independently. The searched keys are the last axis
of ``v`` and any leading axes broadcast against the other dimensions of ``a``,
so one set of keys can be searched in every slice. ``sorter`` may now have the
same shape as ``a``. This feature is available for all built-in dtypes except
``StringDType``, which remains one-dimensional.
