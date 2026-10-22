* Casting a StringDType array to bool now correctly handles missing data
  that is a string. Previously it would treat an empty string as truthy
  and a non-empty string as falsey.
* `numpy.nonzero` now correctly classifies non-empty strings as nonzero.
