* Equality of ``datetime64`` and ``timedelta64`` dtypes whose units differ by a
  metric prefix, such as ``m8[ns]`` and ``m8[1000ps]``, no longer depends on the
  operand order, and units that only nearly match, such as ``m8[2ns]`` and
  ``m8[2001ps]``, are no longer considered equal.
