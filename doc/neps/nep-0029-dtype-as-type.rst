===============================================
NEP 29 — Refactor Dtypes to become Type Objects
===============================================

:Author: Matti Picus
:Status: Draft
:Type: Standards Track
:Created: 2018-12-27


Abstract
========

NumPy's `dtype <http://www.numpy.org/devdocs/reference/generated/numpy.dtype.html>`
is a python class with the simple ``mro`` ``[np.dtype, object]``. Creating an
instance of ``dtype`` *i.e.* ``a = np.dtype('int8')`` will result in a python
object of type ``dtype``. The ``dtype`` obect instance has attributes, among
them ``a.type``, which is a class object. Instantiating that class object
``a.type(3)`` produces a numpy `scalar
<http://www.numpy.org/devdocs/reference/arrays.scalars.html>`.

This NEP proposes a different class heirarchy. Objects of ``np.dtype`` will
become type objects with a heirarchical ``mro`` like scalars. They will support
subclassing. A future NEP may propose that instantiating a dtype type object
will produce a scalar refleting that dtype, but that is not a goal of this NEP.

The changed dtype will:

- facilitate extending dtypes, typically for things like categoricals, novel
  representations like datetime or IP addresses, or adding attributes like
  units.
- Simplify the code around ``__repr__`` and method lookup.

Overall Design
--------------

In pure python (without error checking)::

    import numpy as np

    class Dtype(type):

        def __new__(cls, obj, *args, **kwargs):
            if isinstance(obj, int):
                return dtype_int_dict[obj]
            elif isinstance(obj, type) and issubclass(obj, np.generic):
                return dtype_scalar_dict[obj]
            elif len(args) < 1:
                # Dtype('int8') or Dtype('S10') or record descr
                return create_new_descr(cls, obj, *args, **kwargs)
            else:
                return super().__new__(cls, obj, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self.typeobj(*args, **kwargs)

    class IntDtype(Dtype):
        def __repr__(self):
            if self is IntDescr:
                return type.__repr__(self)
            return 'dtype(%s%d)' %(self.kind, self.itemsize)

    class GenericDescr(type, metaclass=Dtype):
        pass

    class IntDescr(GenericDescr, metaclass=IntDtype):
        def format(value):
            return '%d' % value

    class UInt8Descr(IntDescr):
        kind = 'uint'
        itemsize = 8
        typeobj = np.uint8
        # sort, fill, cast, clip, ...
        ArrFuncs = int8_arrayfuncs

    dtype_int_dict = {1: UInt8Descr}
    dtype_scalar_dict = {np.uint8: UInt8Descr}

At NumPy startup, as we do today, we would generate the builtin set of
descriptor classes, and fill in ``dtype_int_dict`` and ``dtype_scalar_type``
so that the built-in descriptors would continue to be singletons. ``Void``,
``Byte`` and ``Unicode`` descriptors would be constructed on demand, as is done
today.

All dtype instances would inherit from ``GenericDescr`` which inherits from
``type``, making them instances of ``type``::

    >>> a = np.dtype(np.int8)
    >>> a.mro(a)
    [dtype(uint8), <class 'dtype.IntDescr'>, <class 'dtype.GenericDescr'>, \
     <class 'type'>, <class 'object'>]
    
Each descr class will have its own set of ArrFuncs (``clip``, ``fill``,
``cast``), The ``format`` function is what ``array_print`` will call to turn a
memory location into a string.

Downstream users of NumPy could subclass these type classes. Creating a categorical
dtype would look like this (without error checking for out-of-bounds values)::

    class Colors(UInt8Descr):
        colors = ['red', 'green', 'blue']
        def format(value):
            return Colors.colors[value]
        ArrFuncs = null_arrayfuncs

    c = np.array([0, 1, 1, 0, 2], dtype=Colors)    

Additional code would be needed to neutralize the `tp_as_number` slot functions.

Advantages
==========

It is very difficult today to override dtype behaviour, since internally
descriptor objects are not true type instances, rather contianers for the
``ArrayDescrObject`` struct.

Disadvantages
=============

Making descriptors into type objects requires thinking about type classes,
which is more difficult to reason about than object instances. For instance,
note that in the ``Colors`` example, we did not instantiate an object of the
``Colors`` type, rather used that type directly in the ndarray creation. Also
the ``format`` function is not a bound method of a class instance, rather an
unbound function on a type class (no ``self`` argument is used).

Future Extensions
=================

Note the descriptor holds a parallel ``typeobj`` which is a scalar class. A
call like ``np.dtype('int8')(10)`` will now create a scalar object. The next
step will be to replace the scalar classes with the descriptor classes, so
that looking up a scalar's corresponding descriptor type becomes ``type(scalar)``.

We could refactor `numpy.datetime64` to use the new heirarchy, inheriting from
``np.dtype(uint64)``

Alternatives
============

Descriptors as Instances
------------------------

It is confusing that descriptors are classes, not class instances. We could
define them slightly differently as instances (note the call in the value of
``dtype_int_dict`` and that ``_repr__`` is now a bound class method of
``IntDescr``::

    import numpy as np

    class Dtype(type):

        def __new__(cls, obj, *args, **kwargs):
            if isinstance(obj, int):
                return dtype_int_dict[obj]
            elif isinstance(obj, type) and issubclass(obj, np.generic):
                return dtype_scalar_dict[obj]
            elif len(args) < 1:
                # Dtype('int8') or Dtype('S10') or record descr
                return create_new_descr(cls, obj, *args, **kwargs)
            else:
                return super().__new__(cls, obj, *args, **kwargs)

        def __call__(self, args, kwargs):
            return super().__call__(self.__name__, args, kwargs)

    class GenericDescr(type, metaclass=Dtype):
        def __new__(cls, *args, **kwargs):
            import pdb;pdb.set_trace()
            return type.__new__(cls, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self.typeobj(*args, **kwargs)

    class IntDescr(GenericDescr):
        def format(value):
            return '%d' % value
        def __repr__(self):
            return 'dtype(%s%d)' %(self.kind, self.itemsize)


    class UInt8Descr(IntDescr):
        kind = 'uint'
        itemsize = 8
        typeobj = np.uint8
        # sort, fill, cast, clip, ...
        #ArrFuncs = int8_arrayfuncs

    # Create singletons of builtin descriptors via Dtype.__call__
    dtype_int_dict = {1: UInt8Descr()}
    dtype_scalar_dict = {np.uint8: dtype_int_dict[1]}



Appendix
========

References
----------

- pandas `ExtensionArray interface <https://github.com/pandas-dev/pandas/blob/5b0610b875476a6f3727d7e9bedb90d370c669b5/pandas/core/arrays/base.py>`
- Dtype `brainstorming session <https://github.com/numpy/numpy/wiki/Dtype-Brainstorming>`
  from SciPy

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

