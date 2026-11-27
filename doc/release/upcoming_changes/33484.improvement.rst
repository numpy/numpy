* The hash of ``datetime64`` and ``timedelta64`` dtypes now includes the time
  unit. Previously all such dtypes hashed to the same value regardless of
  their unit, so for example ``m8[ns]`` and ``m8[2ns]`` collided in dicts and
  sets. Dtypes that compare equal, such as ``m8[ns]`` and ``m8[1000ps]``,
  still hash equal.
