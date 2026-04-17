New ``PyArray_CorrelateLags`` C API function
--------------------------------------------
A new public C API function `PyArray_CorrelateLags` has been added for
computing one-dimensional cross-correlation at a specified lag range
``[minlag, maxlag)`` with a given ``lagstep``.  The function accepts arrays
in any order and any valid lag-range orientation; internal swapping and
output reversal are handled transparently so the result is aligned with the
caller's input order and lag direction.
