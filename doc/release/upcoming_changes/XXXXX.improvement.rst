* `numpy.take_along_axis` with ``axis=None`` now preserves array subclasses,
  as it already did for integer axes. Previously, subclasses such as masked
  arrays raised an error, which also broke `numpy.ma.MaskedArray.sort` with
  ``axis=None``.
