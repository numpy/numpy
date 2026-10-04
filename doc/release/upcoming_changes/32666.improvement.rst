``np.put`` accepts any integer dtype as indices
-----------------------------------------------
`numpy.put` now casts ``ind`` with same-kind casting, like `numpy.take`.
Integer arrays such as ``uint64``, or ``int64`` on 32-bit platforms, no longer raise a ``TypeError``.
