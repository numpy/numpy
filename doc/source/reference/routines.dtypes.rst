.. _routines.dtypes:

Data type classes (:mod:`numpy.dtypes`)
=======================================

.. automodule:: numpy.dtypes
   :no-members:

Boolean
-------

.. attribute:: BoolDType

Bit-sized integers
------------------

.. attribute:: Int8DType
               UInt8DType
               Int16DType
               UInt16DType
               Int32DType
               UInt32DType
               Int64DType
               UInt64DType

C-named integers (may be aliases)
---------------------------------

.. attribute:: ByteDType
               UByteDType
               ShortDType
               UShortDType
               IntDType
               UIntDType
               LongDType
               ULongDType
               LongLongDType
               ULongLongDType

Floating point
--------------

.. attribute:: Float16DType
               Float32DType
               Float64DType
               LongDoubleDType

Complex
-------

.. attribute:: Complex64DType
               Complex128DType
               CLongDoubleDType

Strings and Bytestrings
-----------------------

.. attribute:: StrDType
               BytesDType
               StringDType

Times
-----

.. attribute:: DateTime64DType
               TimeDelta64DType

Others
------

.. attribute:: ObjectDType
               VoidDType

Abstract DTypes
---------------

These classes cannot be instantiated.  They mirror the numeric part of the
:ref:`scalar type hierarchy <arrays.scalars>` and the concrete DType classes
above inherit from them, so that they can be used in ``isinstance`` and
``issubclass`` checks as well as for the ``kind`` argument of `numpy.isdtype`.
Note that booleans are not considered numeric here.

.. attribute:: NumberAbstractDType
               IntegerAbstractDType
               SignedIntegerAbstractDType
               UnsignedIntegerAbstractDType
               InexactAbstractDType
               FloatingAbstractDType
               ComplexFloatingAbstractDType


Routines for DType authors
--------------------------

.. autosummary::
   :toctree: generated/

   register_dlpack_dtype
