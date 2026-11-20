``numpy.sinc`` is now a ufunc
-----------------------------
`numpy.sinc` is now a ufunc, so it supports the usual ufunc arguments such as
``out`` and ``where`` and is considerably faster. The argument is now reduced
exactly before evaluating the sine, which makes the result accurate near
nonzero integers and exactly zero at them, where previously the relative error
could be large. Infinite inputs now return zero instead of ``nan``.
