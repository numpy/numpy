.. currentmodule:: numpy

.. _arrays.dtypes:

**********************************
Data type objects (:class:`dtype`)
**********************************

A data type object (an instance of :class:`numpy.dtype` class)
describes how the bytes in the fixed-size block of memory
corresponding to an array item should be interpreted. It describes the
following aspects of the data:

1. Type of the data (integer, float, Python object, etc.)
2. Size of the data (how many bytes is in *e.g.* the integer)
3. Byte order of the data (:term:`little-endian` or :term:`big-endian`)
4. If the data type is :term:`structured data type`, an aggregate of other
   data types, (*e.g.*, describing an array item consisting of
   an integer and a float),

   1. what are the names of the ":term:`fields <field>`" of the structure,
      by which they can be :ref:`accessed <arrays.indexing.fields>`,
   2. what is the data-type of each :term:`field`, and
   3. which part of the memory block each field takes.

5. If the data type is a sub-array, what is its shape and data type.

.. index::
   pair: dtype; scalar

To describe the type of scalar data, there are several :ref:`built-in
scalar types <arrays.scalars.built-in>` in NumPy for various precision
of integers, floating-point numbers, *etc*. An item extracted from an
array, *e.g.*, by indexing, will be a Python object whose type is the
scalar type associated with the data type of the array.

Note that the scalar types are not :class:`dtype` objects, even though
they can be used in place of one whenever a data type specification is
needed in NumPy.

.. index::
   pair: dtype; field

Structured data types are formed by creating a data type whose
:term:`field` contain other data types. Each field has a name by
which it can be :ref:`accessed <arrays.indexing.fields>`. The parent data
type should be of sufficient size to contain all its fields; the
parent is nearly always based on the :class:`void` type which allows
an arbitrary item size. Structured data types may also contain nested
structured sub-array data types in their fields.

.. index::
   pair: dtype; sub-array

Finally, a data type can describe items that are themselves arrays of
items of another data type. These sub-arrays must, however, be of a
fixed size.

If an array is created using a data-type describing a sub-array,
the dimensions of the sub-array are appended to the shape
of the array when the array is created. Sub-arrays in a field of a
structured type behave differently, see :ref:`arrays.indexing.fields`.

Sub-arrays always have a C-contiguous memory layout.

.. admonition:: Example

   A simple data type containing a 32-bit big-endian integer:
   (see :ref:`arrays.dtypes.constructing` for details on construction)

   .. try_examples::

      >>> import numpy as np

      >>> dt = np.dtype('>i4')
      >>> dt.byteorder
      '>'
      >>> dt.itemsize
      4
      >>> dt.name
      'int32'
      >>> dt.type is np.int32
      True

   The corresponding array scalar type is :class:`int32`.

.. admonition:: Example

   A structured data type containing a 16-character string (in field 'name')
   and a sub-array of two 64-bit floating-point number (in field 'grades'):

    .. try_examples::

      >>> import numpy as np

      >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
      >>> dt['name']
      dtype('<U16')
      >>> dt['grades']
      dtype(('<f8', (2,)))

   Items of an array of this data type are wrapped in an :ref:`array
   scalar <arrays.scalars>` type that also has two fields:

   .. try_examples::

      >>> import numpy as np

      >>> x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
      >>> x[1]
      ('John', [6., 7.])
      >>> x[1]['grades']
      array([6.,  7.])
      >>> type(x[1])
      <class 'numpy.void'>
      >>> type(x[1]['grades'])
      <class 'numpy.ndarray'>

.. _arrays.dtypes.constructing:

Specifying and constructing data types
======================================

Whenever a data-type is required in a NumPy function or method, either
a :class:`dtype` object or something that can be converted to one can
be supplied.  Such conversions are done by the :class:`dtype`
constructor:

.. autosummary::
   :toctree: generated/

   dtype

What can be converted to a data-type object is described below:

:class:`dtype` object
   .. index::
      triple: dtype; construction; from dtype

   Used as-is.

None
   .. index::
      triple: dtype; construction; from None

   The default data type: :class:`float64`.

.. index::
   triple: dtype; construction; from type

Array-scalar types
    The 24 built-in :ref:`array scalar type objects
    <arrays.scalars.built-in>` all convert to an associated data-type object.
    This is true for their sub-classes as well.

    Note that not all data-type information can be supplied with a
    type-object: for example, `flexible` data-types have
    a default *itemsize* of 0, and require an explicitly given size
    to be useful.

    .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype(np.int32)      # 32-bit integer
         >>> dt = np.dtype(np.complex128) # 128-bit complex floating-point number

Generic types
    The generic hierarchical type objects convert to corresponding
    type objects according to the associations:

    =====================================================  ===================
    :class:`number`, :class:`inexact`, :class:`floating`   :class:`float64`
    :class:`complexfloating`                               :class:`complex128`
    :class:`integer`, :class:`signedinteger`               :class:`int\_`
    :class:`unsignedinteger`                               :class:`uint`
    :class:`generic`, :class:`flexible`                    :class:`void`
    =====================================================  ===================

    .. deprecated:: 1.19

        This conversion of generic scalar types is deprecated.
        This is because it can be unexpected in a context such as
        ``arr.astype(dtype=np.floating)``, which casts an array of ``float32``
        to an array of ``float64``, even though ``float32`` is a subdtype of
        ``np.floating``.

.. _dtype-constructing-from-python-types:

Built-in Python types
    Several Python types are equivalent to a corresponding
    array scalar when used to generate a :class:`dtype` object:

    ===================      ===============
    Python type              NumPy type
    ===================      ===============
    :class:`int`             :class:`int\_`
    :class:`bool`            :class:`bool\_`
    :class:`float`           :class:`float64`
    :class:`complex`         :class:`complex128`
    :class:`bytes`           :class:`bytes\_`
    :class:`str`             :class:`str\_`
    :class:`memoryview`      :class:`void`
    (all others)             :class:`object_`
    ===================      ===============

    Note that ``str_`` corresponds to UCS4 encoded unicode strings.

    .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype(float)   # Python-compatible floating-point number
         >>> dt = np.dtype(int)     # Python-compatible integer
         >>> dt = np.dtype(object)  # Python object

    .. note::

        All other types map to ``object_`` for convenience. Code should expect
        that such types may map to a specific (new) dtype in the future.

Types with ``.dtype``
    Any type object with a ``dtype`` attribute: The attribute will be
    accessed and used directly. The attribute must return something
    that is convertible into a dtype object.

.. index::
   triple: dtype; construction; from string

Several kinds of strings can be converted. Recognized strings can be
prepended with ``'>'`` (:term:`big-endian`), ``'<'``
(:term:`little-endian`), or ``'='`` (hardware-native, the default), to
specify the byte order.

One-character strings
    Each built-in data-type has a character code
    (the updated Numeric typecodes), that uniquely identifies it.

    .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype('b')  # byte, native byte order
         >>> dt = np.dtype('>H') # big-endian unsigned short
         >>> dt = np.dtype('<f') # little-endian single-precision float
         >>> dt = np.dtype('d')  # double-precision floating-point number

Array-protocol type strings (see :ref:`arrays.interface`)
   The first character specifies the kind of data and the remaining
   characters specify the number of bytes per item, except for Unicode,
   where it is interpreted as the number of characters.  The item size
   must correspond to an existing type, or an error will be raised.  The
   supported kinds are

   ================   ========================
   ``'?'``            boolean
   ``'b'``            (signed) byte
   ``'B'``            unsigned byte
   ``'i'``            (signed) integer
   ``'u'``            unsigned integer
   ``'f'``            floating-point
   ``'c'``            complex-floating point
   ``'m'``            timedelta
   ``'M'``            datetime
   ``'O'``            (Python) objects
   ``'S'``, ``'a'``   zero-terminated bytes (not recommended)
   ``'U'``            Unicode string
   ``'V'``            raw data (:class:`void`)
   ================   ========================

   .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype('i4')   # 32-bit signed integer
         >>> dt = np.dtype('f8')   # 64-bit floating-point number
         >>> dt = np.dtype('c16')  # 128-bit complex floating-point number
         >>> dt = np.dtype('S25')  # 25-length zero-terminated bytes
         >>> dt = np.dtype('U25')  # 25-character string

   .. _string-dtype-note:

   .. admonition:: Note on string types

    For backward compatibility with existing code originally written to support
    Python 2, ``S`` and ``a`` typestrings are zero-terminated bytes.
    For unicode strings, use ``U``, `numpy.str_`.  For signed bytes that do not
    need zero-termination ``b`` or ``i1`` can be used.

String with comma-separated fields
   A short-hand notation for specifying the format of a structured data type is
   a comma-separated string of basic formats.

   A basic format in this context is an optional shape specifier
   followed by an array-protocol type string. Parenthesis are required
   on the shape if it has more than one dimension. NumPy allows a modification
   on the format in that any string that can uniquely identify the
   type can be used to specify the data-type in a field.
   The generated data-type fields are named ``'f0'``, ``'f1'``, ...,
   ``'f<N-1>'`` where N (>1) is the number of comma-separated basic
   formats in the string. If the optional shape specifier is provided,
   then the data-type for the corresponding field describes a sub-array.

   .. admonition:: Example

      .. try_examples::

      - field named ``f0`` containing a 32-bit integer
      - field named ``f1`` containing a 2 x 3 sub-array
        of 64-bit floating-point numbers
      - field named ``f2`` containing a 32-bit floating-point number

         >>> import numpy as np
         >>> dt = np.dtype("i4, (2,3)f8, f4")

      - field named ``f0`` containing a 3-character string
      - field named ``f1`` containing a sub-array of shape (3,)
        containing 64-bit unsigned integers
      - field named ``f2`` containing a 3 x 4 sub-array
        containing 10-character strings

         >>> import numpy as np
         >>> dt = np.dtype("S3, 3u8, (3,4)S10")

Type strings
   Any string name of a NumPy dtype, e.g.:

   .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype('uint32')   # 32-bit unsigned integer
         >>> dt = np.dtype('float64')  # 64-bit floating-point number

.. index::
   triple: dtype; construction; from tuple

``(flexible_dtype, itemsize)``
    The first argument must be an object that is converted to a
    zero-sized flexible data-type object, the second argument is
    an integer providing the desired itemsize.

    .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype((np.void, 10))  # 10-byte wide data block
         >>> dt = np.dtype(('U', 10))   # 10-character unicode string

``(fixed_dtype, shape)``
    .. index::
       pair: dtype; sub-array

    The first argument is any object that can be converted into a
    fixed-size data-type object. The second argument is the desired
    shape of this type. If the shape parameter is 1, then the
    data-type object used to be equivalent to fixed dtype. This behaviour is
    deprecated since NumPy 1.17 and will raise an error in the future.
    If *shape* is a tuple, then the new dtype defines a sub-array of the given
    shape.

    .. admonition:: Example

      .. try_examples::

         >>> import numpy as np

         >>> dt = np.dtype((np.int32, (2,2)))          # 2 x 2 integer sub-array
         >>> dt = np.dtype(('i4, (2,3)f8, f4', (2,3))) # 2 x 3 structured sub-array

.. index::
   triple: dtype; construction; from list

``[(field_name, field_dtype, field_shape), ...]``
   *obj* should be a list of fields where each field is described by a
   tuple of length 2 or 3. (Equivalent to the ``descr`` item in the
   :obj:`~object.__array_interface__` attribute.)

   The first element, *field_name*, is the field name (if this is
   ``''`` then a standard field name, ``'f#'``, is assigned).  The
   field name may also be a 2-tuple of strings where the first string
   is either a "title" (which may be any string or unicode string) or
   meta-data for the field which can be any object, and the second
   string is the "name" which must be a valid Python identifier.

   The second element, *field_dtype*, can be anything that can be
   interpreted as a data-type.

   The optional third element *field_shape* contains the shape if this
   field represents an array of the data-type in the second
   element. Note that a 3-tuple with a third argument equal to 1 is
   equivalent to a 2-tuple.

   This style does not accept *align* in the :class:`dtype`
   constructor as it is assumed that all of the memory is accounted
   for by the array interface description.

   .. admonition:: Example

      .. try_examples::

         Data-type with fields ``big`` (big-endian 32-bit integer) and
         ``little`` (little-endian 32-bit integer):

         >>> import numpy as np

         >>> dt = np.dtype([('big', '>i4'), ('little', '<i4')])

         Data-type with fields ``R``, ``G``, ``B``, ``A``, each being an
         unsigned 8-bit integer:

         >>> dt = np.dtype([('R','u1'), ('G','u1'), ('B','u1'), ('A','u1')])

.. index::
   triple: dtype; construction; from dict

``{'names': ..., 'formats': ..., 'offsets': ..., 'titles': ..., 'itemsize': ...}``
    This style has two required and three optional keys.  The *names*
    and *formats* keys are required. Their respective values are
    equal-length lists with the field names and the field formats.
    The field names must be strings and the field formats can be any
    object accepted by :class:`dtype` constructor.

    When the optional keys *offsets* and *titles* are provided,
    their values must each be lists of the same length as the *names*
    and *formats* lists. The *offsets* value is a list of byte offsets
    (limited to `ctypes.c_int`) for each field, while the *titles* value is a
    list of titles for each field (``None`` can be used if no title is
    desired for that field). The *titles* can be any object, but when a
    :class:`str` object will add another entry to the
    fields dictionary keyed by the title and referencing the same
    field tuple which will contain the title as an additional tuple
    member.

    The *itemsize* key allows the total size of the dtype to be
    set, and must be an integer large enough so all the fields
    are within the dtype. If the dtype being constructed is aligned,
    the *itemsize* must also be divisible by the struct alignment. Total dtype
    *itemsize* is limited to `ctypes.c_int`.

    .. admonition:: Example

      .. try_examples::

         Data type with fields ``r``, ``g``, ``b``, ``a``, each being
         an 8-bit unsigned integer:

         >>> import numpy as np

         >>> dt = np.dtype({'names': ['r','g','b','a'],
         ...                'formats': [np.uint8, np.uint8, np.uint8, np.uint8]})

         Data type with fields ``r`` and ``b`` (with the given titles),
         both being 8-bit unsigned integers, the first at byte position
         0 from the start of the field and the second at position 2:

         >>> dt = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
         ...                'offsets': [0, 2],
         ...                'titles': ['Red pixel', 'Blue pixel']})


``{'field1': ..., 'field2': ..., ...}``
    This usage is discouraged, because it is ambiguous with the
    other dict-based construction method. If you have a field
    called 'names' and a field called 'formats' there will be
    a conflict.

    This style allows passing in the :attr:`fields <dtype.fields>`
    attribute of a data-type object.

    *obj* should contain string or unicode keys that refer to
    ``(data-type, offset)`` or ``(data-type, offset, title)`` tuples.

    .. admonition:: Example

      .. try_examples::

         Data type containing field ``col1`` (10-character string at
         byte position 0), ``col2`` (32-bit float at byte position 10),
         and ``col3`` (integers at byte position 14):

         >>> import numpy as np

         >>> dt = np.dtype({'col1': ('U10', 0), 'col2': (np.float32, 10),
         ...                'col3': (int, 14)})

``(base_dtype, new_dtype)``
    In NumPy 1.7 and later, this form allows `base_dtype` to be interpreted as
    a structured dtype. Arrays created with this dtype will have underlying
    dtype `base_dtype` but will have fields and flags taken from `new_dtype`.
    This is useful for creating custom structured dtypes, as done in
    :ref:`record arrays <arrays.classes.rec>`.

    This form also makes it possible to specify struct dtypes with overlapping
    fields, functioning like the 'union' type in C. This usage is discouraged,
    however, and the union mechanism is preferred.

    Both arguments must be convertible to data-type objects with the same total
    size.

    .. admonition:: Example

      .. try_examples::

         32-bit integer, whose first two bytes are interpreted as an integer
         via field ``real``, and the following two bytes via field ``imag``.

         >>> import numpy as np

         >>> dt = np.dtype((np.int32,{'real':(np.int16, 0),'imag':(np.int16, 2)}))

         32-bit integer, which is interpreted as consisting of a sub-array
         of shape ``(4,)`` containing 8-bit integers:

         >>> dt = np.dtype((np.int32, (np.int8, 4)))

         32-bit integer, containing fields ``r``, ``g``, ``b``, ``a`` that
         interpret the 4 bytes in the integer as four unsigned integers:

         >>> dt = np.dtype(('i4', [('r','u1'),('g','u1'),('b','u1'),('a','u1')]))


Checking the data type
======================
When checking for a specific data type, use ``==`` comparison.

.. admonition:: Example

   .. try_examples::

      >>> import numpy as np

      >>> a = np.array([1, 2], dtype=np.float32)
      >>> a.dtype == np.float32
      True

As opposed to Python types, a comparison using ``is`` should not be used.

First, NumPy treats data type specifications (everything that can be passed to
the :class:`dtype` constructor) as equivalent to the data type object itself.
This equivalence can only be handled through ``==``, not through ``is``.

.. admonition:: Example

   .. try_examples::

      A :class:`dtype` object is equal to all data type specifications that are
      equivalent to it.

      >>> import numpy as np

      >>> a = np.array([1, 2], dtype=float)
      >>> a.dtype == np.dtype(np.float64)
      True
      >>> a.dtype == np.float64
      True
      >>> a.dtype == float
      True
      >>> a.dtype == "float64"
      True
      >>> a.dtype == "d"
      True

Second, there is no guarantee that data type objects are singletons.

.. admonition:: Example

   .. try_examples::

      Do not use ``is`` because data type objects may or may not be singletons.

      >>> import numpy as np

      >>> np.dtype(float) is np.dtype(float)
      True
      >>> np.dtype([('a', float)]) is np.dtype([('a', float)])
      False

:class:`dtype`
==============

NumPy data type descriptions are instances of the :class:`dtype` class.

Attributes
----------

The type of the data is described by the following :class:`dtype`  attributes:

.. autosummary::
   :toctree: generated/

   dtype.type
   dtype.kind
   dtype.char
   dtype.num
   dtype.str

Size of the data is in turn described by:

.. autosummary::
   :toctree: generated/

   dtype.name
   dtype.itemsize

Endianness of this data:

.. autosummary::
   :toctree: generated/

   dtype.byteorder

Information about sub-data-types in a :term:`structured data type`:

.. autosummary::
   :toctree: generated/

   dtype.fields
   dtype.names

For data types that describe sub-arrays:

.. autosummary::
   :toctree: generated/

   dtype.subdtype
   dtype.shape

Attributes providing additional information:

.. autosummary::
   :toctree: generated/

   dtype.hasobject
   dtype.flags
   dtype.isbuiltin
   dtype.isnative
   dtype.descr
   dtype.alignment
   dtype.base

Metadata attached by the user:

.. autosummary::
   :toctree: generated/

    dtype.metadata


Methods
-------

Data types have the following method for changing the byte order:

.. autosummary::
   :toctree: generated/

   dtype.newbyteorder

The following methods implement the pickle protocol:

.. autosummary::
   :toctree: generated/

   dtype.__reduce__
   dtype.__setstate__

Utility method for typing:

.. autosummary::
   :toctree: generated/

   dtype.__class_getitem__

Comparison operations:

.. autosummary::
   :toctree: generated/

   dtype.__ge__
   dtype.__gt__
   dtype.__le__
   dtype.__lt__
