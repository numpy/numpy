``equal_nan`` option for `np.unique` changed behavior
-----------------------------------------------------
fixes gh-23286.
fixes gh-20873.

For multi-dimensional arrays with given axis and for structured 1-dimensional
arrays, if ``equal_nan`` is ``True``, then uniques are chosen as if the
comparison ``(1, np.nan) == (1, np.nan)`` was true. For example,
``np.unique([[0, np.nan], [0, np.nan], [np.nan, 0]], equal_nan=True, axis=0)``
returns ``[[0, np.nan], [np.nan, 0]]``. The same applies if the input was a
1-dimensional structured array of tuples ``(float, float)``, except that the
result is structured.

Previously, the result was ``[[0, np.nan], [0, np.nan], [1, np.nan]]``, which
is the result one would expect when ``equal_nan`` is ``False``. This happened
because the value of ``equal_nan`` was basically (but not explicitly) ignored
and assumed as ``False`` for multi-dimensional arrays and 1-dimensional
structured arrays. After this update, this output is obtained only when
``equal_nan`` is ``False``, which matches the semantics of "nans being equal".

This change makes the behavior of ``equal_nan`` more consistent.

