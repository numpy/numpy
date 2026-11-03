``f2py`` converts character array results of call-backs to the declared length
-------------------------------------------------------------------------------
A character array returned by a Python call-back for an ``intent(out)``
argument is now converted to the length declared in the call-back signature,
as for the array arguments of the wrapped routine itself. Previously the
strings kept the length the call-back produced and were copied into the
Fortran array byte for byte, so shorter strings ran into the neighbouring
elements.

``f2py`` call-backs that raise no longer abort with Python thread-local storage
-------------------------------------------------------------------------------
When the call-back pointers are kept in the Python thread state (with
``F2PY_USE_PYTHON_TLS`` defined, or with a C compiler without thread-local
storage), an exception raised by a call-back is now propagated to the caller
of the wrapped routine. Previously it aborted the interpreter with
``Fatal Python error: F2PySwapThreadLocalCallbackPtr: PyLong_AsVoidPtr failed``.
