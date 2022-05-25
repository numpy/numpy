* `PyArray_CanCastArrayTo` does NOT honor the warning request.
  Potential alternative: Honor it, but not if the warning is
  raised (because it swallows errors :().
