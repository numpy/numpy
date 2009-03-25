.. index::
   pair: array; interface
   pair: array; protocol

.. _arrays.interface:

*******************
The Array Interface
*******************

.. warning::

   This page describes the old, deprecated array interface. Everything still
   works as described as of numpy 1.2 and on into the foreseeable future), but
   new development should target :pep:`3118` --
   :cfunc:`The Revised Buffer Protocol <PyObject_GetBuffer>`.
   :pep:`3118` was incorporated into Python 2.6 and 3.0, and is additionally
   supported by Cython's numpy buffer support. (See the  Cython numpy
   tutorial.) Cython provides a way to write code that supports the buffer
   protocol with Python versions older than 2.6 because it has a
   backward-compatible implementation utilizing the legacy array interface
   described here.

:version: 3

The array interface (sometimes called array protocol) was created in
2005 as a means for array-like Python objects to re-use each other's
data buffers intelligently whenever possible. The homogeneous
N-dimensional array interface is a default mechanism for objects to
share N-dimensional array memory and information.  The interface
consists of a Python-side and a C-side using two attributes.  Objects
wishing to be considered an N-dimensional array in application code
should support at least one of these attributes.  Objects wishing to
support an N-dimensional array in application code should look for at
least one of these attributes and use the information provided
appropriately.

This interface describes homogeneous arrays in the sense that each
item of the array has the same "type".  This type can be very simple
or it can be a quite arbitrary and complicated C-like structure.

There are two ways to use the interface: A Python side and a C-side.
Both are separate attributes.

Python side
===========

This approach to the interface consists of the object having an
:data:`__array_interface__` attribute.

.. data:: __array_interface__

   A dictionary of items (3 required and 5 optional).  The optional
   keys in the dictionary have implied defaults if they are not
   provided.

   The keys are:

   **shape** (required)

       Tuple whose elements are the array size in each dimension. Each
       entry is an integer (a Python int or long).  Note that these
       integers could be larger than the platform "int" or "long"
       could hold (a Python int is a C long). It is up to the code
       using this attribute to handle this appropriately; either by
       raising an error when overflow is possible, or by using
       :cdata:`Py_LONG_LONG` as the C type for the shapes.

   **typestr** (required)

       A string providing the basic type of the homogenous array The
       basic string format consists of 3 parts: a character describing
       the byteorder of the data (``<``: little-endian, ``>``:
       big-endian, ``|``: not-relevant), a character code giving the
       basic type of the array, and an integer providing the number of
       bytes the type uses.

       The basic type character codes are:

       =====  ================================================================
       ``t``  Bit field (following integer gives the number of
              bits in the bit field).
       ``b``  Boolean (integer type where all values are only True or False)
       ``i``  Integer
       ``u``  Unsigned integer
       ``f``  Floating point
       ``c``  Complex floating point
       ``O``  Object (i.e. the memory contains a pointer to :ctype:`PyObject`)
       ``S``  String (fixed-length sequence of char)
       ``U``  Unicode (fixed-length sequence of :ctype:`Py_UNICODE`)
       ``V``  Other (void \* -- each item is a fixed-size chunk of memory)
       =====  ================================================================

   **descr** (optional)

       A list of tuples providing a more detailed description of the
       memory layout for each item in the homogeneous array.  Each
       tuple in the list has two or three elements.  Normally, this
       attribute would be used when *typestr* is ``V[0-9]+``, but this is
       not a requirement.  The only requirement is that the number of
       bytes represented in the *typestr* key is the same as the total
       number of bytes represented here.  The idea is to support
       descriptions of C-like structs (records) that make up array
       elements.  The elements of each tuple in the list are

       1.  A string providing a name associated with this portion of
           the record.  This could also be a tuple of ``('full name',
	   'basic_name')`` where basic name would be a valid Python
           variable name representing the full name of the field.

       2. Either a basic-type description string as in *typestr* or
          another list (for nested records)

       3. An optional shape tuple providing how many times this part
          of the record should be repeated.  No repeats are assumed
          if this is not given.  Very complicated structures can be
          described using this generic interface.  Notice, however,
          that each element of the array is still of the same
          data-type.  Some examples of using this interface are given
          below.

       **Default**: ``[('', typestr)]``

   **data** (optional)

       A 2-tuple whose first argument is an integer (a long integer
       if necessary) that points to the data-area storing the array
       contents.  This pointer must point to the first element of
       data (in other words any offset is always ignored in this
       case). The second entry in the tuple is a read-only flag (true
       means the data area is read-only).

       This attribute can also be an object exposing the
       :cfunc:`buffer interface <PyObject_AsCharBuffer>` which
       will be used to share the data. If this key is not present (or
       returns :class:`None`), then memory sharing will be done
       through the buffer interface of the object itself.  In this
       case, the offset key can be used to indicate the start of the
       buffer.  A reference to the object exposing the array interface
       must be stored by the new object if the memory area is to be
       secured.

       **Default**: :const:`None`

   **strides** (optional)

       Either :const:`None` to indicate a C-style contiguous array or
       a Tuple of strides which provides the number of bytes needed
       to jump to the next array element in the corresponding
       dimension. Each entry must be an integer (a Python
       :const:`int` or :const:`long`). As with shape, the values may
       be larger than can be represented by a C "int" or "long"; the
       calling code should handle this appropiately, either by
       raising an error, or by using :ctype:`Py_LONG_LONG` in C. The
       default is :const:`None` which implies a C-style contiguous
       memory buffer.  In this model, the last dimension of the array
       varies the fastest.  For example, the default strides tuple
       for an object whose array entries are 8 bytes long and whose
       shape is (10,20,30) would be (4800, 240, 8)

       **Default**: :const:`None` (C-style contiguous)

   **mask** (optional)

       :const:`None` or an object exposing the array interface.  All
       elements of the mask array should be interpreted only as true
       or not true indicating which elements of this array are valid.
       The shape of this object should be `"broadcastable"
       <arrays.broadcasting.broadcastable>` to the shape of the
       original array.

       **Default**: :const:`None` (All array values are valid)

   **offset** (optional)

       An integer offset into the array data region. This can only be
       used when data is :const:`None` or returns a :class:`buffer`
       object.

       **Default**: 0.

   **version** (required)

       An integer showing the version of the interface (i.e. 3 for
       this version).  Be careful not to use this to invalidate
       objects exposing future versions of the interface.


C-struct access
===============

This approach to the array interface allows for faster access to an
array using only one attribute lookup and a well-defined C-structure.

.. cvar:: __array_struct__

   A :ctype:`PyCObject` whose :cdata:`voidptr` member contains a
   pointer to a filled :ctype:`PyArrayInterface` structure.  Memory
   for the structure is dynamically created and the :ctype:`PyCObject`
   is also created with an appropriate destructor so the retriever of
   this attribute simply has to apply :cfunc:`Py_DECREF()` to the
   object returned by this attribute when it is finished.  Also,
   either the data needs to be copied out, or a reference to the
   object exposing this attribute must be held to ensure the data is
   not freed.  Objects exposing the :obj:`__array_struct__` interface
   must also not reallocate their memory if other objects are
   referencing them.

.. admonition:: New since June 16, 2006:

   In the past most implementations used the "desc" member of the
   :ctype:`PyCObject` itself (do not confuse this with the "descr" member of
   the :ctype:`PyArrayInterface` structure above --- they are two separate
   things) to hold the pointer to the object exposing the interface.
   This is now an explicit part of the interface.  Be sure to own a
   reference to the object when the :ctype:`PyCObject` is created using
   :ctype:`PyCObject_FromVoidPtrAndDesc`.
