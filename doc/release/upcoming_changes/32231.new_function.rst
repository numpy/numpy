New function `numpy.minmax`
---------------------------

A new function ``np.minmax(a, axis=..., out=..., keepdims=..., initial=...,
where=...)`` was added, which returns a ``(min, max)`` tuple holding the
minimum and the maximum of an array along a given axis.  It is equivalent to
``(np.min(a, ...), np.max(a, ...))`` but computes both in a single fused pass
over the data.
