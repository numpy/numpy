Reduction loops are now used by ``accumulate``
----------------------------------------------
Ufuncs with more than one output that register a dedicated reduction loop on
their ``ArrayMethod`` now support :meth:`~numpy.ufunc.accumulate` as well as
:meth:`~numpy.ufunc.reduce`. Both return one array per output. See
:ref:`c-api.reduction-loop-tutorial` for a worked example.
