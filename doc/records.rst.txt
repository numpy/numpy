
The ndarray supports records intrinsically.  None of the default
descriptors have fields defined, but you can create new descriptors
easily.  The ndarray even supports nested arrays of records inside of
a record.  Any record that the array protocol can describe can be
represented.  The ndarray also supports partial field descriptors.
Not every byte has to be accounted for.

This was done by adding to the established ``PyArray_Descr *`` structure:

1. A PyObject ``*fields`` member which contains a dictionary of "field
   name" : (``PyArray_Descr`` ``*field-type``, ``offset``, [optional field
   title]).  If a title is given, then it is also inserted into the
   dictionary and used to key the same entry.

2. A byteorder member.  By default this is '=' (native), or '|'
   (not-applicable).

3. An additional ``PyArray_ArrDescr`` ``*member`` of the structure which
   contains a simple representation of an array of another base-type.
   types. The ``PyArray_ArrayDescr`` structure has members
   ``PyArray_Descr *``, ``PyObject *``, for holding a reference to
   the base-type and the shape of the sub-array.

4. The ``PyArray_Descr *`` as official Python object that fully describes
   a region of memory for the data


Data type conversions
---------------------

We can support additional data-type
conversions.  The data-type passed in is converted to a
``PyArray_Descr *`` object.

New possibilities for the "data-type"
`````````````````````````````````````

**List [data-type 1, data-type 2, ..., data-type n]**
  Equivalent to  {'names':['f1','f2',...,'fn'],
	        'formats': [data-type 1, data-type 2, ..., data-type n]}

  This is a quick way to specify a record format with default field names.


**Tuple  (flexible type, itemsize) (fixed type, shape)**
  Get converted to a new ``PyArray_Descr *`` object with a flexible
  type. The latter structure also sets the ``PyArray_ArrayDescr`` field of the
  returned ``PyArray_Descr *``.


**Dictionary (keys "names", "titles", and "formats")**
  This will be converted to a ``NPY_VOID`` type with corresponding
  fields parameter (the formats list will be converted to actual
  ``PyArray_Descr *`` objects).


**Objects (anything with an .itemsize and .fields attribute)**
  If its an instance of (a sub-class of) void type, then a new
  ``PyArray_Descr*`` structure is created corresponding to its
  typeobject (and ``NPY_VOID``) typenumber.  If the type is
  registered, then the registered type-number is used.

  Otherwise a new ``NPY_VOID PyArray_Descr*`` structure is created
  and filled ->elsize and ->fields filled in appropriately.

  The itemsize attribute must return a number > 0. The fields
  attribute must return a dictionary with at least "names" and
  "formats" entries.  The "formats" entry will be converted to a
  "proper" descr->fields entry (all generic data-types converted to
  ``PyArray_Descr *`` structure).


Reference counting for ``PyArray_Descr *`` objects.
```````````````````````````````````````````````````

Most functions that take ``PyArary_Descr *`` as arguments and return a
``PyObject *`` steal the reference unless otherwise noted in the code:

Functions that return ``PyArray_Descr *`` objects return a new
reference.

.. tip::

  There is a new function  and a new method of array objects both labelled
  dtypescr which can be used to try out the ``PyArray_DescrConverter``.
