``np.repeat`` accepts any integer dtype as ``repeats``
------------------------------------------------------
`numpy.repeat` now casts ``repeats`` with same-kind casting, like `numpy.take`.
Integer arrays such as ``uint64``, or ``int64`` on 32-bit platforms, no longer raise a ``TypeError``.
