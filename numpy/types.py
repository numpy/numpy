"""
Names of builtin NumPy Types (:mod:`numpy.types`)
==================================================

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The types module is new in NumPy 1.25.  Older exceptions remain
    available through the main NumPy namespace for compatibility.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classe can be used for ``isisntance`` checks but are
otherwise not typically useful as of now.

For general information see `numpy.dtype` and :ref:`arrays.dtypes`.

.. list-table::
    :header-rows: 1

    * - Group
      - DType class

    * - Boolean
      - ``BoolDType``

    * - Bit-sized integers
      - ``Int8DType``, ``UInt8DType``, ``Int16DType``, ``UInt16DType``,
        ``Int32DType``, ``UInt32DType``, ``Int64DType``, ``UInt64DType``

    * - C-named integers (may be aliases)
      - ``ByteDType``, ``UByteDType``, ``ShortDType``, ``UShortDType``,
        ``IntDType``, ``UIntDType``, ``LongDType``, ``ULongDType``,
        ``LongLongDType``, ``ULongLongDType``

    * - Floating point
      - ``Float16DType``, ``Float32DType``, ``Float64DType``,
        ``LongDoubleDType``

    * - Complex
      - ``Complex64DType``, ``Complex128DType``, ``CLongDoubleDType``

    * - Strings
      - ``BytesDType``, ``BytesDType``

    * - Times
      - ``DateTime64DType``, ``TimeDelta64DType``

    * - Others
      - ``ObjectDType``, ``VoidDType``

"""

__all__ = []


def _add_dtype_helper(DType, alias):
    # Function to add DTypes a bit more conveniently without channeling them
    # through `numpy.core._multiarray_umath` namespace or similar.
    from numpy import types

    setattr(types, DType.__name__, DType)
    __all__.append(DType.__name__)

    if alias:
        alias = alias.removeprefix("numpy.types.")
        setattr(types, alias, DType)
        __all__.append(alias)
