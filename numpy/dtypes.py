"""
This module is home to specific dtypes related functionality and their classes.
For more general information about dtypes, also see `numpy.dtype` and
:ref:`arrays.dtypes`.

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The dtypes module is new in NumPy 1.25.  Previously DType classes were
    only accessible indirectly.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classes can be used in ``isinstance`` checks and can
also be instantiated or used directly.  Direct use of these classes is not
typical, since their scalar counterparts (e.g. ``np.float64``) or strings
like ``"float64"`` can be used.
"""

# See doc/source/reference/routines.dtypes.rst for module-level docs
__all__ = ["register_dlpack_dtype"]


def register_dlpack_dtype(dlpack_key, dtype):
    """
    register_dlpack_dtype(dlpack_key, dtype, /)

    Register a NumPy dtype for a DLPack ``(code, bits)`` pair so that
    `numpy.from_dlpack` can import it and ``ndarray.__dlpack__`` can export
    it.  Built-in dtype mappings take priority on import.

    If you think a conflict is possible but unproblematic you may wrap this
    into a try/except block.

    .. warning::
        It is the responsibility of the registering user to ensure that the
        mapping is valid.

    .. note::
        This function was added primarily for ``ml_dtypes`` and may be
        replaced with a different mechanism in the future.

    Parameters
    ----------
    dlpack_key : tuple of int
        ``(dl_dtype_code, dl_bits)`` matching the DLPack ``DLDataType``,
        lanes is assumed to be always 1, currently.
    dtype : dtype
        A NumPy dtype instance.

    Raises
    ------
    ValueError : If a conflicting registration was already done.

    See Also
    --------
    numpy.from_dlpack

    """
    # Deferred to avoid circular import.
    from numpy._core._multiarray_umath import _register_dlpack_dtype

    return _register_dlpack_dtype(dlpack_key, dtype)


def _add_dtype_helper(DType, alias):
    # Function to add DTypes a bit more conveniently without channeling them
    # through `numpy._core._multiarray_umath` namespace or similar.
    from numpy import dtypes

    setattr(dtypes, DType.__name__, DType)
    __all__.append(DType.__name__)

    if alias:
        alias = alias.removeprefix("numpy.dtypes.")
        setattr(dtypes, alias, DType)
        __all__.append(alias)
