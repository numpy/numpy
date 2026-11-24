``take_along_axis`` and ``apply_over_axes`` accept array-likes
--------------------------------------------------------------
`numpy.take_along_axis` and `numpy.apply_over_axes` now accept array-likes
such as lists, which previously raised an ``AttributeError``.
`numpy.apply_over_axes` also no longer converts array subclasses such as
masked arrays to base-class ndarrays before calling ``func``.
