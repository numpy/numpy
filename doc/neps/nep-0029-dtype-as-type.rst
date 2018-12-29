=============================
NEP 29 — Refactor Dtype Class
=============================

:Author: Matti Picus
:Status: Draft
:Type: Standards Track
:Created: 2018-12-27


Abstract
========

NumPy's `dtype <http://www.numpy.org/devdocs/reference/generated/numpy.dtype.html>`_
describes a class used to create a descriptor, which provides the methods
to convert the strided memory in an ndarray into objects that can be
manipulated, printed, and used in loops. In this NEP, **dtype** is used as the
name of the class, and **descriptor** is used as the name of the instance. A
descriptor is what is held by a ndarray object, in addition to a block of
memory and information about the strides and shape of elements in that memory.
Creating an instance of ``dtype`` *i.e.* ``a = np.dtype('int8')`` will result
in a descriptor, which is a python object of type ``dtype``.

The ``dtype`` obect instance ``a`` has attributes, among them ``a.type``, which
is a class object. Instantiating that class object ``a.type(3)`` produces a
Num`py `scalar <http://www.numpy.org/devdocs/reference/arrays.scalars.html>`_.

This NEP proposes a class heirarchy for dtypes. The ``np.dtype`` class will
become an abstrace base class, and a number of new classes will be created with
a heirarchy like scalars. They will support subclassing. A future NEP may
propose unifying the scalar and dtype type systems, but that is not a goal of
this NEP.  The changed dtype will:

- facilitate extending dtypes, typically for things like categoricals, novel
  representations like datetime or IP addresses, or adding attributes like
  units directly on the dtypes.
- Allow replacement of the switch-type code we have in places like arrayprint_,
  PyArray_AdaptFlexibleDtype_, and PyArray_CanCastSafely_ (just a few of many),
  with method-based overrides, simplifying internal logic and allowing
  subclassed dtypes to override the functionality.

.. _arrayprint: https://github.com/numpy/numpy/blob/v1.16.0rc1/numpy/core/arrayprint.py#L418
.. _PyArray_AdaptFlexibleDtype: https://github.com/numpy/numpy/blob/v1.16.0rc1/numpy/core/src/multiarray/convert_datatype.c#L164 
.. _PyArray_CanCastSafely: https://github.com/numpy/numpy/blob/v1.16.0rc1/numpy/core/src/multiarray/convert_datatype.c#L398

Overall Design
--------------

The ``Dtype`` class (and any subclass without ``itemsize``) is by definition an
abstract base class. A metaclass ``DtypeMeta`` is used to add slots for
converting memory chunks into objects and back, to provide datatype-specific
functionality, and to provide the casting functions to convert data.

A prototype without error checking, without options handling, and describing
only ``np.dtype(np.uint8)`` together with an overridable ``get_format_function``
for ``arrayprint`` looks like::

    import numpy as np

    class DtypeMeta(type):
        # Add slot methods to the base Dtype to handle low-level memory
        # conversion to/from char[itemsize] to int/float/utf8/whatever
        # In cython this would look something like
        #cdef int (*unbox_method)(PyObject* self, PyObject* source, char* dest)
        #cdef PyObject* (*box_method)(PyObject* self, char* source)

        def __call__(cls, *args, **kwargs):
            # This is reached for Dtype(np.uint8, ...).
            # Does not yet handle align, copy positional arguments
            if len(args) > 0: 
                obj = args[0]
                if isinstance(obj, int):
                    return dtype_int_dict[obj]
                elif isinstance(obj, type) and issubclass(obj, np.generic):
                    return dtype_scalar_dict[obj]
                else:
                    # Dtype('int8') or Dtype('S10') or record descr
                    return create_new_descr(cls, *args, **kwargs)
            else:
                # At import, when creating Dtype and subclasses
                return type.__call__(cls, *args, **kwargs)

    class Dtype():
        def __new__(cls, *args, **kwargs):
            # Do not allow creating instances of abstract base classes
            if not hasattr(cls, 'itemsize'):
                raise ValueError("cannot create instances of "
                                 f"abstract class {cls!r}")
            return super().__new__(cls, *args, **kwargs)

    class GenericDescr(Dtype, metaclass=DtypeMeta):
        pass

    class IntDescr(GenericDescr):
        def __repr__(self):
            # subclass of IntDescr
            return f"dtype('{_kind_to_stem[self.kind]}{self.itemsize:d}')"

        def get_format_function(self, data, **options):
            # replaces switch on dtype found in _get_format_function
            # (in arrayprint), **options details missing
            from  np.core.arrayprint import IntegerFormat
            return IntegerFormat(data)

    class UInt8Descr(IntDescr):
        kind = 'u'
        itemsize = 8
        type = np.uint8
        # The C-based methods for sort, fill, cast, clip, ... not exposed to
        # Python
        #ArrFuncs = int8_arrayfuncs
        

    dtype_int_dict = {1: UInt8Descr()}
    dtype_scalar_dict = {np.uint8: UInt8Descr()}
    _kind_to_stem = {
        'u': 'uint',
        'i': 'int',
        'c': 'complex',
        'f': 'float',
        'b': 'bool',
        'V': 'void',
        'O': 'object',
        'M': 'datetime',
        'm': 'timedelta',
        'S': 'bytes',
        'U': 'str',
    }

At NumPy startup, as we do today, we would generate the builtin set of
descriptor instances, and fill in ``dtype_int_dict`` and ``dtype_scalar_type``
so that the built-in descriptors would continue to be singletons. ``Void``,
``Byte`` and ``Unicode`` descriptors would be constructed on demand, as is done
today. The magic that returns a singleton or a new descriptor happens in
``DtypeMeta.__call__``. 

All descriptors would inherit from ``Dtype``::

    >>> a = np.dtype(np.uint8)
    >>> type(a).mro()
    [<class 'UInt8Descr'>, <class 'IntDescr'>, <class 'GenericDescr'>,
     <class 'Dtype'>, <class 'object'>]

    >>> isinstance(a, np.dtype):
    True
    
Note that the ``repr`` of ``a`` is compatibility with NumPy::

    >>> repr(a)
    "dtype('uint8')"

Each class will have its own set of ArrFuncs (``clip``, ``fill``,
``cast``). 

Downstream users of NumPy can subclass these type classes. Creating a categorical
dtype would look like this (without error checking for out-of-bounds values)::

    class Colors(Dtype):
        itemsize = 8
        colors = ['red', 'green', 'blue']
        def get_format_function(self, data, **options):
            class Format():
                def __init__(self, data):
                    pass
                def __call__(self, x):
                return self.colors[x]
            return Format(data)

    c = np.array([0, 1, 1, 0, 2], dtype=Colors)    

Additional code would be needed to neutralize the slot functions.

There is a level of indirection between ``Dtype`` and ``IntDescr`` so that
downstream users could create their own duck-descriptors that do not use 
``DtypeMeta.__call__`` at all, but could still answer ``True`` to 
``isintance(mydtype, Dtype)``.

Advantages
==========

It is very difficult today to override dtype behaviour. Internally
descriptor objects are all instances of a generic dtype class and internally
behave as containers more than classes with method overrides. Giving them a
class heirarchy with overrideable methods will reduce explicit branching in
code (at the expense of a dictionary lookup) and allow downstream users to
more easily define new dtypes. We could re-examine interoperability with
pandas_ typesystem.

Disadvantages
=============

The new code will be incompatible with old code and will at least
require a recompile of C-extension modules, including updating cython. We
should continue with `PR 12284`_ to vendor our own numpy.pxd in order to make the
transition less painful. We should not break working dtype-subclasses like
`quaterions`_.

Future Extensions
=================

Note the descriptor holds a ``typeobj`` which is a scalar class. A call like
``np.dtype('int8')(10)`` could theoretically create a scalar object.
This would make the descriptor more like the ``int`` or ``float`` type. However
allowing instantiating scalars from descriptors is not a goal of this NEP.

A further extension would be to refactor ``numpy.datetime64`` to use the new
heirarchy.

Appendix
========

References
----------

- pandas `ExtensionArray interface`_ 
- Dtype `brainstorming session <https://github.com/numpy/numpy/wiki/Dtype-Brainstorming>`_
  from SciPy

.. _pandas: https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/base.py#L148
.. _`ExtensionArray interface`: https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/base.py#L148
.. _`PR 12284`: https://github.com/numpy/numpy/pull/12284
.. _`quaterions`: https://github.com/moble/quaternion

The current interface of dtypes in NumPy
----------------------------------------

.. code-block:: python

    class DescrFlags(IntFlags):
        # The item must be reference counted when it is inserted or extracted.
        ITEM_REFCOUNT   = 0x01
        # Same as needing REFCOUNT
        ITEM_HASOBJECT  = 0x01
        # Convert to list for pickling
        LIST_PICKLE     = 0x02
        # The item is a POINTER 
        ITEM_IS_POINTER = 0x04
        # memory needs to be initialized for this data-type
        NEEDS_INIT      = 0x08
        # operations need Python C-API so don't give-up thread.
        NEEDS_PYAPI     = 0x10
        # Use f.getitem when extracting elements of this data-type
        USE_GETITEM     = 0x20
        # Use f.setitem when setting creating 0-d array from this data-type
        USE_SETITEM     = 0x40
        # A sticky flag specifically for structured arrays
        ALIGNED_STRUCT  = 0x80

    class current_dtype(object):
        itemsize: int
        alignment: int
        
        byteorder: str
        flags: DescrFlags
        metadata: ...  # unknown
        
        # getters
        hasobject: bool
        isalignedstruct: bool
        isbuiltin: bool
        isnative: bool
        
        
        def newbyteorder(self) -> current_dtype: ...
        
        # to move to a structured dtype subclass
        names: Tuple[str]
        fields: Dict[str, Union[
        Tuple[current_dtype, int],
        Tuple[current_dtype, int, Any]
        ]]
        
        # to move to a subarray dtype subclass
        subdtype: Optional[Tuple[dtype, Tuple[int,...]]]
        shape: Tuple[int]
        base: current_dtype
        
        # to deprecate
        type: Type  # merge with cls
        kind: str
        num: int
        str: str
        name: str
        char: str
        descr: List[...]

