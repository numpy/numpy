`numpy.linalg.pinv` now uses the computation dtype for ``rtol=None`` and empty input
------------------------------------------------------------------------------------
The default tolerance used for ``rtol=None`` was derived from the input dtype,
which raised ``ValueError`` for integer input even though the singular value
decomposition is computed in floating point.  The shortcut for empty input
returned the input dtype, while the decomposition path returns the promoted
one.  Both now use the promoted computation dtype, so ``rtol=None`` works for
integer input and an empty integer array gives a ``float64`` result instead of
an integer one.  As a consequence, an empty array of a dtype that
`numpy.linalg` does not support, such as ``float16`` or ``longdouble``, now
raises the same ``TypeError`` as a non-empty one.
