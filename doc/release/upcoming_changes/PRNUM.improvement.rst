``f2py`` converts character array results of call-backs to the declared length
-------------------------------------------------------------------------------
A character array returned by a Python call-back for an ``intent(out)``
argument is now converted to the length declared in the call-back signature,
as for the array arguments of the wrapped routine itself. Previously the
strings kept the length the call-back produced and were copied into the
Fortran array byte for byte, so shorter strings ran into the neighbouring
elements.
