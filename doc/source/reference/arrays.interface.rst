.. index::
   pair: array; interface
   pair: array; protocol

.. _arrays.interface:

*******************
The Array Interface
*******************

.. note::

   This page describes the numpy-specific API for accessing the contents of
   a numpy array from other C extensions. :pep:`3118` --
   :c:func:`The Revised Buffer Protocol <PyObject_GetBuffer>` introduces
   similar, standardized API to Python 2.6 and 3.0 for any extension
   module to use. Cython__'s buffer array support
   uses the :pep:`3118` API; see the `Cython numpy
   tutorial`__. Cython provides a way to write code that supports the buffer
   protocol with Python versions older than 2.6 because it has a
   backward-compatible implementation utilizing the array interface
   described here.

__ http://cython.org/
__ http://wiki.cython.org/tutorials/numpy

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
       :c:data:`Py_LONG_LONG` as the C type for the shapes.

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
       ``m``  Timedelta
       ``M``  Datetime
       ``O``  Object (i.e. the memory contains a pointer to :c:type:`PyObject`)
       ``S``  String (fixed-length sequence of char)
       ``U``  Unicode (fixed-length sequence of :c:type:`Py_UNICODE`)
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
       descriptions of C-like structs that make up array
       elements.  The elements of each tuple in the list are

       1.  A string providing a name associated with this portion of
           the datatype.  This could also be a tuple of ``('full name',
	   'basic_name')`` where basic name would be a valid Python
           variable name representing the full name of the field.

       2. Either a basic-type description string as in *typestr* or
          another list (for nested structured types)

       3. An optional shape tuple providing how many times this part
          of the structure should be repeated.  No repeats are assumed
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
       :c:func:`buffer interface <PyObject_AsCharBuffer>` which
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
       calling code should handle this appropriately, either by
       raising an error, or by using :c:type:`Py_LONG_LONG` in C. The
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

.. c:var:: __array_struct__

   A :c:type: `PyCObject` whose :c:data:`voidptr` member contains a
   pointer to a filled :c:type:`PyArrayInterface` structure.  Memory
   for the structure is dynamically created and the :c:type:`PyCObject`
   is also created with an appropriate destructor so the retriever of
   this attribute simply has to apply :c:func:`Py_DECREF()` to the
   object returned by this attribute when it is finished.  Also,
   either the data needs to be copied out, or a reference to the
   object exposing this attribute must be held to ensure the data is
   not freed.  Objects exposing the :obj:`__array_struct__` interface
   must also not reallocate their memory if other objects are
   referencing them.

The PyArrayInterface structure is defined in ``numpy/ndarrayobject.h``
as::

  typedef struct {
    int two;              /* contains the integer 2 -- simple sanity check */
    int nd;               /* number of dimensions */
    char typekind;        /* kind in array --- character code of typestr */
    int itemsize;         /* size of each element */
    int flags;            /* flags indicating how the data should be interpreted */
                          /*   must set ARR_HAS_DESCR bit to validate descr */
    Py_intptr_t *shape;   /* A length-nd array of shape information */
    Py_intptr_t *strides; /* A length-nd array of stride information */
    void *data;           /* A pointer to the first element of the array */
    PyObject *descr;      /* NULL or data-description (same as descr key
                                  of __array_interface__) -- must set ARR_HAS_DESCR
                                  flag or this will be ignored. */
  } PyArrayInterface;

The flags member may consist of 5 bits showing how the data should be
interpreted and one bit showing how the Interface should be
interpreted.  The data-bits are :const:`CONTIGUOUS` (0x1),
:const:`FORTRAN` (0x2), :const:`ALIGNED` (0x100), :const:`NOTSWAPPED`
(0x200), and :const:`WRITEABLE` (0x400).  A final flag
:const:`ARR_HAS_DESCR` (0x800) indicates whether or not this structure
has the arrdescr field.  The field should not be accessed unless this
flag is present.

.. admonition:: New since June 16, 2006:

   In the past most implementations used the "desc" member of the
   :c:type:`PyCObject` itself (do not confuse this with the "descr" member of
   the :c:type:`PyArrayInterface` structure above --- they are two separate
   things) to hold the pointer to the object exposing the interface.
   This is now an explicit part of the interface.  Be sure to own a
   reference to the object when the :c:type:`PyCObject` is created using
   :c:type:`PyCObject_FromVoidPtrAndDesc`.


Type description examples
=========================

For clarity it is useful to provide some examples of the type
description and corresponding :data:`__array_interface__` 'descr'
entries.  Thanks to Scott Gilbert for these examples:

In every case, the 'descr' key is optional, but of course provides
more information which may be important for various applications::

     * Float data
         typestr == '>f4'
         descr == [('','>f4')]

     * Complex double
         typestr == '>c8'
         descr == [('real','>f4'), ('imag','>f4')]

     * RGB Pixel data
         typestr == '|V3'
         descr == [('r','|u1'), ('g','|u1'), ('b','|u1')]

     * Mixed endian (weird but could happen).
         typestr == '|V8' (or '>u8')
         descr == [('big','>i4'), ('little','<i4')]

     * Nested structure
         struct {
             int ival;
             struct {
                 unsigned short sval;
                 unsigned char bval;
                 unsigned char cval;
             } sub;
         }
         typestr == '|V8' (or '<u8' if you want)
         descr == [('ival','<i4'), ('sub', [('sval','<u2'), ('bval','|u1'), ('cval','|u1') ]) ]

     * Nested array
         struct {
             int ival;
             double data[16*4];
         }
         typestr == '|V516'
         descr == [('ival','>i4'), ('data','>f8',(16,4))]

     * Padded structure
         struct {
             int ival;
             double dval;
         }
         typestr == '|V16'
         descr == [('ival','>i4'),('','|V4'),('dval','>f8')]

It should be clear that any structured type could be described using this
interface.

Differences with Array interface (Version 2)
============================================

The version 2 interface was very similar.  The differences were
largely aesthetic.  In particular:

1. The PyArrayInterface structure had no descr member at the end
   (and therefore no flag ARR_HAS_DESCR)

2. The desc member of the PyCObject returned from __array_struct__ was
   not specified.  Usually, it was the object exposing the array (so
   that a reference to it could be kept and destroyed when the
   C-object was destroyed).  Now it must be a tuple whose first
   element is a string with "PyArrayInterface Version #" and whose
   second element is the object exposing the array.

3. The tuple returned from __array_interface__['data'] used to be a
   hex-string (now it is an integer or a long integer).

4. There was no __array_interface__ attribute instead all of the keys
   (except for version) in the __array_interface__ dictionary were
   their own attribute: Thus to obtain the Python-side information you
   had to access separately the attributes:

   * __array_data__
   * __array_shape__
   * __array_strides__
   * __array_typestr__
   * __array_descr__
   * __array_offset__
   * __array_mask__
