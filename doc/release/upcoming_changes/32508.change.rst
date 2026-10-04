``MaskedArray`` resets a fill_value that cannot be represented in a new dtype
-----------------------------------------------------------------------------
A fill_value copied from a source array is now reset to the default fill_value
for the new dtype when it cannot be represented in that dtype.  Previously the
copied value could be stale: ufuncs that change dtype left the result holding a
fill_value typed for the old dtype, which raised a ``TypeError`` only when
something later validated it (such as ``.view()``), and a floating point
fill_value that overflows an integer dtype was kept as an out-of-range value
together with a ``RuntimeWarning``.  Both now fall back to the default,
including when a masked array is passed to ``MaskedArray`` with a different
``dtype``.  A fill_value given explicitly by the user is still validated and
raises as before.  This can now raise a ``ComplexWarning`` if the fill_value is
complex and the new dtype is real.
