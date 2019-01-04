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
Numpy `scalar <http://www.numpy.org/devdocs/reference/arrays.scalars.html>`_.

This NEP proposes a class hierarchy for dtypes. The ``np.dtype`` class will
become an abstract base class, and a number of new classes will be created with
a hierarchy like scalars. They will support subclassing. A future NEP may
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

The ``Dtype`` class (and any subclass without ``itemsize``) is effectively an
abstract base class, as it cannot be used to create instances. A class
hierarchy is used to add datatype-specific functionality such as ``names`` and
``fields`` for structured dtypes. The current behaviours are preserved:

- ``np.dtype(obj, align=False, copy=False)`` calls ``arraydescr_new`` with
  various types of ``obj``:
  - ``int``, ``np.genenric`` scalar classes, list or dict are parsed into
    appropriate dtypes
- singletons are returned where appropriate

Additionally, dtype subclasses are passed through to the subclass ``__new__``

A prototype without error checking, without options handling, and describing
only ``np.dtype(np.uint8)`` together with an overridable ``get_format_function``
for ``arrayprint`` looks like::

    import numpy as np

    class Dtype():
        def __new__(cls, *args, **kwargs):
            if len(args) == 0: 
                # Do not allow creating instances of abstract base classes
                if not hasattr(cls, 'itemsize'):
                    raise ValueError("cannot create instances of "
                                     f"abstract class {cls!r}")
                return super().__new__(cls, *args, **kwargs)
            # This is reached for Dtype(np.uint8, ...).
            # Does not yet handle align, copy positional arguments
            obj = args[0]
            if isinstance(obj, int):
                return dtype_int_dict[obj]
            elif isinstance(obj, type) and issubclass(obj, np.generic):
                return dtype_scalar_dict[obj]
            else:
                # Dtype('int8') or Dtype('S10') or record descr
                return create_new_descr(cls, *args, **kwargs)

    class GenericDtype(Dtype):
        pass

    class IntDtype(GenericDtype):
        def __repr__(self):
            # subclass of IntDescr
            return f"dtype('{_kind_to_stem[self.kind]}{self.itemsize:d}')"

        def get_format_function(self, data, **options):
            # replaces switch on dtype found in _get_format_function
            # (in arrayprint), **options details missing
            from  numpy.core.arrayprint import IntegerFormat
            return IntegerFormat(data)

    class UInt8Dtype(IntDtype):
        kind = 'u'
        itemsize = 8
        type = np.uint8
        # The C-based methods for sort, fill, cast, clip, ... not exposed to
        # Python
        #ArrFuncs = int8_arrayfuncs
        

    dtype_int_dict = {1: UInt8Dtype()}
    dtype_scalar_dict = {np.uint8: dtype_int_dict[1]} 
    _kind_to_stem = {
        'u': 'uint',
        'i': 'int',
    }


At NumPy startup, as we do today, we would generate the builtin set of
descriptor instances, and fill in ``dtype_int_dict`` and ``dtype_scalar_type``
so that the built-in descriptors would continue to be singletons. Some
descriptors would be constructed on demand, as is done today.

All descriptors would inherit from ``Dtype``::

    >>> a = np.dtype(np.uint8)
    >>> type(a).mro()
    [<class 'UInt8Dtype'>, <class 'IntDtype'>, <class 'GenericDtype'>,
     <class 'Dtype'>, <class 'object'>]

    >>> isinstance(a, np.dtype):
    True
    
Note that the ``repr`` of ``a`` is compatibility with NumPy::

    >>> repr(a)
    "dtype('uint8')"

Each class will have its own set of ArrFuncs (``clip``, ``fill``,
``cast``) and attributes appropriate to that class.

Downstream users of NumPy can subclass these type classes. Creating a categorical
dtype would look like this (without error checking for out-of-bounds values)::

    class Plant(Dtype):
        itemsize = 8
        names = ['tree', 'flower', 'grass']
        def get_format_function(self, data, **options):
            class Format():
                def __init__(self, data):
                    pass
                def __call__(self, x):
                    return Plant.names[x]
            return Format(data)

    c = np.array([0, 1, 1, 0, 2], dtype=Plant)    

Additional code would be needed to neutralize the slot functions.

The overall hierarchy is meant to map to the scalar hierarchy.

Now ``arrayprint`` would look something like this (very much simplified, the
actual format details are not the point):

    def arrayformat(data, dtype):
            formatter = dtype.get_format_function(data)
            result = []
            for v in data:
                result.append(formatter(v))
            return 'array[' + ', '.join(result) + ']'

    def arrayprint(data):
        print(arrayformat(data, data.dtype))

    a = np.array([0, 1, 2, 0, 1, 2], dtype='uint8')

    # Create a dtype instance, returns a singleton from dtype_scalar_dict
    uint8 = Dtype(np.uint8)
    
    # Create a user-defined dtype
    garden = Plant()

    # We cannot use ``arrayprint`` just yet, but ``arrayformat`` works
    print(arrayformat(a, uint8))

    array[0, 1, 2, 0, 1, 2]

    print(arrayformat(a, garden))

    array[tree, flower, grass, tree, flower, grass]

Advantages
==========

It is very difficult today to override dtype behaviour. Internally
descriptor objects are all instances of a generic dtype class and internally
behave as containers more than classes with method overrides. Giving them a
class hierarchy with overrideable methods will reduce explicit branching in
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

Code that depends on all dtypes having similar attributes might break. For
instance there is no reason ``int`` dtypes need the ``names`` and ``field``
empty attributes.

Future Extensions
=================

Note the descriptor holds a ``typeobj`` which is a scalar class. A call like
``np.dtype('int8')(10)`` could theoretically create a scalar object.
This would make the descriptor more like the ``int`` or ``float`` type. However
allowing instantiating scalars from descriptors is not a goal of this NEP.

A further extension would be to refactor ``numpy.datetime64`` to use the new
hierarchy.

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

