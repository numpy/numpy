``datetime64``/``timedelta64`` arithmetic raises on overflow
------------------------------------------------------------

Addition, subtraction, and integer multiplication of ``datetime64`` and
``timedelta64`` values now raise ``OverflowError`` when the result would
overflow ``int64`` or land on the ``NaT`` sentinel value.  Previously these
operations silently wrapped, often producing a value that was
indistinguishable from ``NaT``.  This matches the overflow checking already
performed by unit-conversion casts.
