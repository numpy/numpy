.. _arrays.scalars:

*******
Scalars
*******

.. currentmodule:: numpy

Python defines only one type of a particular data class (there is only
one integer type, one floating-point type, etc.). This can be
convenient in applications that don't need to be concerned with all
the ways data can be represented in a computer.  For scientific
computing, however, more control is often needed.

In NumPy, there are 24 new fundamental Python types to describe
different types of scalars. These type descriptors are mostly based on
the types available in the C language that CPython is written in, with
several additional types compatible with Python's types.

Array scalars have the same attributes and methods as :class:`ndarrays
<ndarray>`. [#]_ This allows one to treat items of an array partly on
the same footing as arrays, smoothing out rough edges that result when
mixing scalar and array operations.

Array scalars live in a hierarchy (see the Figure below) of data
types. They can be detected using the hierarchy: For example,
``isinstance(val, np.generic)`` will return :py:data:`True` if *val* is
an array scalar object. Alternatively, what kind of array scalar is
present can be determined using other members of the data type
hierarchy. Thus, for example ``isinstance(val, np.complexfloating)``
will return :py:data:`True` if *val* is a complex valued type, while
``isinstance(val, np.flexible)`` will return true if *val* is one
of the flexible itemsize array types (:class:`str_`,
:class:`bytes_`, :class:`void`).

.. figure:: figures/dtype-hierarchy.png

   **Figure:** Hierarchy of type objects representing the array data
   types. Not shown are the two integer types :class:`intp` and
   :class:`uintp` which are used for indexing (the same as the
   default integer since NumPy 2).


.. TODO - use something like this instead of the diagram above, as it generates
   links to the classes and is a vector graphic. Unfortunately it looks worse
   and the html <map> element providing the linked regions is misaligned.

   .. inheritance-diagram:: byte short intc int_ longlong ubyte ushort uintc uint ulonglong half single double longdouble csingle cdouble clongdouble bool_ datetime64 timedelta64 object_ bytes_ str_ void

.. [#] However, array scalars are immutable, so none of the array
       scalar attributes are settable.

.. _arrays.scalars.character-codes:

.. _arrays.scalars.built-in:

Built-in scalar types
=====================

The built-in scalar types are shown below. The C-like names are associated with character codes,
which are shown in their descriptions. Use of the character codes, however,
is discouraged.

Some of the scalar types are essentially equivalent to fundamental
Python types and therefore inherit from them as well as from the
generic array scalar type:

====================  ===========================  =============
Array scalar type     Related Python type          Inherits?
====================  ===========================  =============
:class:`int_`         :class:`int`                 Python 2 only
:class:`double`       :class:`float`               yes
:class:`cdouble`      :class:`complex`             yes
:class:`bytes_`       :class:`bytes`               yes
:class:`str_`         :class:`str`                 yes
:class:`bool_`        :class:`bool`                no
:class:`datetime64`   :class:`datetime.datetime`   no
:class:`timedelta64`  :class:`datetime.timedelta`  no
====================  ===========================  =============

The :class:`bool_` data type is very similar to the Python
:class:`bool` but does not inherit from it because Python's
:class:`bool` does not allow itself to be inherited from, and
on the C-level the size of the actual bool data is not the same as a
Python Boolean scalar.

.. warning::

   The :class:`int_` type does **not** inherit from the
   :class:`int` built-in under Python 3, because type :class:`int` is no
   longer a fixed-width integer type.

.. tip:: The default data type in NumPy is :class:`double`.

.. autoclass:: numpy.generic
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.number
   :members: __init__
   :exclude-members: __init__

Integer types
-------------

.. autoclass:: numpy.integer
   :members: __init__
   :exclude-members: __init__

.. note::

   The numpy integer types mirror the behavior of C integers, and can therefore
   be subject to :ref:`overflow-errors`.

Signed integer types
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: numpy.signedinteger
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.byte
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.short
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.intc
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.int_
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.long
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.longlong
   :members: __init__
   :exclude-members: __init__

Unsigned integer types
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: numpy.unsignedinteger
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.ubyte
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.ushort
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.uintc
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.uint
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.ulong
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.ulonglong
   :members: __init__
   :exclude-members: __init__

Inexact types
-------------

.. autoclass:: numpy.inexact
   :members: __init__
   :exclude-members: __init__

.. note::

   Inexact scalars are printed using the fewest decimal digits needed to
   distinguish their value from other values of the same datatype,
   by judicious rounding. See the ``unique`` parameter of
   `format_float_positional` and `format_float_scientific`.

   This means that variables with equal binary values but whose datatypes are of
   different precisions may display differently:

   .. try_examples::

      >>> import numpy as np

      >>> f16 = np.float16("0.1")
      >>> f32 = np.float32(f16)
      >>> f64 = np.float64(f32)
      >>> f16 == f32 == f64
      True
      >>> f16, f32, f64
      (0.1, 0.099975586, 0.0999755859375)

      Note that none of these floats hold the exact value :math:`\frac{1}{10}`;
      ``f16`` prints as ``0.1`` because it is as close to that value as possible,
      whereas the other types do not as they have more precision and therefore have
      closer values.

      Conversely, floating-point scalars of different precisions which approximate
      the same decimal value may compare unequal despite printing identically:

      >>> f16 = np.float16("0.1")
      >>> f32 = np.float32("0.1")
      >>> f64 = np.float64("0.1")
      >>> f16 == f32 == f64
      False
      >>> f16, f32, f64
      (0.1, 0.1, 0.1)

Floating-point types
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: numpy.floating
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.half
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.single
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.double
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.longdouble
   :members: __init__
   :exclude-members: __init__

Complex floating-point types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: numpy.complexfloating
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.csingle
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.cdouble
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.clongdouble
   :members: __init__
   :exclude-members: __init__

Other types
-----------

.. autoclass:: numpy.bool_
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.bool
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.datetime64
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.timedelta64
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.object_
   :members: __init__
   :exclude-members: __init__

.. note::

   The data actually stored in object arrays
   (*i.e.*, arrays having dtype :class:`object_`) are references to
   Python objects, not the objects themselves. Hence, object arrays
   behave more like usual Python :class:`lists <list>`, in the sense
   that their contents need not be of the same Python type.

   The object type is also special because an array containing
   :class:`object_` items does not return an :class:`object_` object
   on item access, but instead returns the actual object that
   the array item refers to.

.. index:: flexible

The following data types are **flexible**: they have no predefined
size and the data they describe can be of different length in different
arrays. (In the character codes ``#`` is an integer denoting how many
elements the data type consists of.)

.. autoclass:: numpy.flexible
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.character
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.bytes_
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.str_
   :members: __init__
   :exclude-members: __init__

.. autoclass:: numpy.void
   :members: __init__
   :exclude-members: __init__


.. warning::

   See :ref:`Note on string types<string-dtype-note>`.

   Numeric Compatibility: If you used old typecode characters in your
   Numeric code (which was never recommended), you will need to change
   some of them to the new characters. In particular, the needed
   changes are ``c -> S1``, ``b -> B``, ``1 -> b``, ``s -> h``, ``w ->
   H``, and ``u -> I``. These changes make the type character
   convention more consistent with other Python modules such as the
   :mod:`struct` module.

.. _sized-aliases:

Sized aliases
-------------

Along with their (mostly)
C-derived names, the integer, float, and complex data-types are also
available using a bit-width convention so that an array of the right
size can always be ensured. Two aliases (:class:`numpy.intp` and :class:`numpy.uintp`)
pointing to the integer type that is sufficiently large to hold a C pointer
are also provided.

.. note that these are documented with ..attribute because that is what
   autoclass does for aliases under the hood.

.. attribute:: int8
               int16
               int32
               int64

   Aliases for the signed integer types (one of `numpy.byte`, `numpy.short`,
   `numpy.intc`, `numpy.int_`, `numpy.long` and `numpy.longlong`)
   with the specified number of bits.

   Compatible with the C99 ``int8_t``, ``int16_t``, ``int32_t``, and
   ``int64_t``, respectively.

.. attribute:: uint8
               uint16
               uint32
               uint64

   Alias for the unsigned integer types (one of `numpy.ubyte`, `numpy.ushort`,
   `numpy.uintc`, `numpy.uint`, `numpy.ulong` and `numpy.ulonglong`)
   with the specified number of bits.

   Compatible with the C99 ``uint8_t``, ``uint16_t``, ``uint32_t``, and
   ``uint64_t``, respectively.

.. attribute:: intp

   Alias for the signed integer type (one of `numpy.byte`, `numpy.short`,
   `numpy.intc`, `numpy.int_`, `numpy.long` and `numpy.longlong`)
   that is used as a default integer and for indexing.

   Compatible with the C ``Py_ssize_t``.

   :Character code: ``'n'``

   .. versionchanged:: 2.0
      Before NumPy 2, this had the same size as a pointer.  In practice this
      is almost always identical, but the character code ``'p'`` maps to the C
      ``intptr_t``.  The character code ``'n'`` was added in NumPy 2.0.

.. attribute:: uintp

   Alias for the unsigned integer type that is the same size as ``intp``.

   Compatible with the C ``size_t``.

   :Character code: ``'N'``

   .. versionchanged:: 2.0
      Before NumPy 2, this had the same size as a pointer.  In practice this
      is almost always identical, but the character code ``'P'`` maps to the C
      ``uintptr_t``.  The character code ``'N'`` was added in NumPy 2.0.

.. autoclass:: numpy.float16

.. autoclass:: numpy.float32

.. autoclass:: numpy.float64

.. attribute:: float96
               float128

   Alias for `numpy.longdouble`, named after its size in bits.
   The existence of these aliases depends on the platform.

.. autoclass:: numpy.complex64

.. autoclass:: numpy.complex128

.. attribute:: complex192
               complex256

   Alias for `numpy.clongdouble`, named after its size in bits.
   The existence of these aliases depends on the platform.

Attributes
==========

The array scalar objects have an :obj:`array priority
<class.__array_priority__>` of :c:data:`NPY_SCALAR_PRIORITY`
(-1,000,000.0). They also do not (yet) have a :attr:`ctypes <ndarray.ctypes>`
attribute. Otherwise, they share the same attributes as arrays:

.. autosummary::
   :toctree: generated/

   generic.flags
   generic.shape
   generic.strides
   generic.ndim
   generic.data
   generic.size
   generic.itemsize
   generic.base
   generic.dtype
   generic.real
   generic.imag
   generic.flat
   generic.T
   generic.__array_interface__
   generic.__array_struct__
   generic.__array_priority__
   generic.__array_wrap__


Indexing
========
.. seealso:: :ref:`arrays.indexing`, :ref:`arrays.dtypes`

Array scalars can be indexed like 0-dimensional arrays: if *x* is an
array scalar,

- ``x[()]`` returns a copy of array scalar
- ``x[...]`` returns a 0-dimensional :class:`ndarray`
- ``x['field-name']`` returns the array scalar in the field *field-name*.
  (*x* can have fields, for example, when it corresponds to a structured data type.)

Methods
=======

Array scalars have exactly the same methods as arrays. The default
behavior of these methods is to internally convert the scalar to an
equivalent 0-dimensional array and to call the corresponding array
method. In addition, math operations on array scalars are defined so
that the same hardware flags are set and used to interpret the results
as for :ref:`ufunc <ufuncs>`, so that the error state used for ufuncs
also carries over to the math on array scalars.

The exceptions to the above rules are given below:

.. autosummary::
   :toctree: generated/

   generic.__array__
   generic.__array_wrap__
   generic.squeeze
   generic.byteswap
   generic.__reduce__
   generic.__setstate__
   generic.setflags

Utility method for typing:

.. autosummary::
   :toctree: generated/

   number.__class_getitem__


Defining new types
==================

There are two ways to effectively define a new array scalar type
(apart from composing structured types :ref:`dtypes <arrays.dtypes>` from
the built-in scalar types): One way is to simply subclass the
:class:`ndarray` and overwrite the methods of interest. This will work to
a degree, but internally certain behaviors are fixed by the data type of
the array.  To fully customize the data type of an array you need to
define a new data-type, and register it with NumPy. Such new types can only
be defined in C, using the :ref:`NumPy C-API <c-api>`.
