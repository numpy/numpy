.. currentmodule:: numpy.ma


.. _numpy.ma.constants:

Constants of the :mod:`numpy.ma` module
=======================================

In addition to the :class:`MaskedArray` class, the :mod:`numpy.ma` module
defines several constants.

.. data:: masked

   The :attr:`masked` constant is a special case of :class:`MaskedArray`,
   with a float datatype and a null shape. It is used to test whether a
   specific entry of a masked array is masked, or to mask one or several
   entries of a masked array::

      >>> x = ma.array([1, 2, 3], mask=[0, 1, 0])
      >>> x[1] is ma.masked
      True
      >>> x[-1] = ma.masked
      >>> x
      masked_array(data = [1 -- --],
                   mask = [False  True  True],
             fill_value = 999999)


.. data:: nomask

   Value indicating that a masked array has no invalid entry.
   :attr:`nomask` is used internally to speed up computations when the mask
   is not needed.


.. data:: masked_print_options

   String used in lieu of missing data when a masked array is printed.
   By default, this string is ``'--'``.




.. _maskedarray.baseclass:

The :class:`MaskedArray` class
==============================


.. class:: MaskedArray

   A subclass of :class:`~numpy.ndarray` designed to manipulate numerical arrays with missing data.



 An instance of :class:`MaskedArray` can be thought as the combination of several elements:

* The :attr:`~MaskedArray.data`, as a regular :class:`numpy.ndarray` of any shape or datatype (the data).
* A boolean :attr:`~numpy.ma.MaskedArray.mask` with the same shape as the data, where a ``True`` value indicates that the corresponding element of the data is invalid.
  The special value :const:`nomask` is also acceptable for arrays without named fields, and indicates that no data is invalid.
* A :attr:`~numpy.ma.MaskedArray.fill_value`, a value that may be used to replace the invalid entries in order to return a standard :class:`numpy.ndarray`.



Attributes and properties of masked arrays
------------------------------------------

.. seealso:: :ref:`Array Attributes <arrays.ndarray.attributes>`


.. attribute:: MaskedArray.data

   Returns the underlying data, as a view of the masked array.
   If the underlying data is a subclass of :class:`numpy.ndarray`, it is
   returned as such.

      >>> x = ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
      >>> x.data
      matrix([[1, 2],
              [3, 4]])

   The type of the data can be accessed through the :attr:`baseclass`
   attribute.

.. attribute:: MaskedArray.mask

   Returns the underlying mask, as an array with the same shape and structure
   as the data, but where all fields are atomically booleans.
   A value of ``True`` indicates an invalid entry.


.. attribute:: MaskedArray.recordmask

   Returns the mask of the array if it has no named fields. For structured
   arrays, returns a ndarray of booleans where entries are ``True`` if **all**
   the fields are masked, ``False`` otherwise::

      >>> x = ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
      ...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
      ...        dtype=[('a', int), ('b', int)])
      >>> x.recordmask
      array([False, False,  True, False, False], dtype=bool)


.. attribute:: MaskedArray.fill_value

   Returns the value used to fill the invalid entries of a masked array.
   The value is either a scalar (if the masked array has no named fields),
   or a 0-D ndarray with the same :attr:`dtype` as the masked array if it has
   named fields.

   The default filling value depends on the datatype of the array:

   ========  ========
   datatype  default
   ========  ========
   bool      True
   int       999999
   float     1.e20
   complex   1.e20+0j
   object    '?'
   string    'N/A'
   ========  ========



.. attribute:: MaskedArray.baseclass

   Returns the class of the underlying data.

      >>> x =  ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 0], [1, 0]])
      >>> x.baseclass
      <class 'numpy.matrixlib.defmatrix.matrix'>


.. attribute:: MaskedArray.sharedmask

   Returns whether the mask of the array is shared between several masked arrays.
   If this is the case, any modification to the mask of one array will be
   propagated to the others.


.. attribute:: MaskedArray.hardmask

   Returns whether the mask is hard (``True``) or soft (``False``).
   When the mask is hard, masked entries cannot be unmasked.


As :class:`MaskedArray` is a subclass of :class:`~numpy.ndarray`, a masked array also inherits all the attributes and properties of a  :class:`~numpy.ndarray` instance.

.. autosummary::
   :toctree: generated/

   MaskedArray.base
   MaskedArray.ctypes
   MaskedArray.dtype
   MaskedArray.flags

   MaskedArray.itemsize
   MaskedArray.nbytes
   MaskedArray.ndim
   MaskedArray.shape
   MaskedArray.size
   MaskedArray.strides

   MaskedArray.imag
   MaskedArray.real

   MaskedArray.flat
   MaskedArray.__array_priority__



:class:`MaskedArray` methods
============================

.. seealso:: :ref:`Array methods <array.ndarray.methods>`


Conversion
----------

.. autosummary::
   :toctree: generated/

   MaskedArray.__float__
   MaskedArray.__hex__
   MaskedArray.__int__
   MaskedArray.__long__
   MaskedArray.__oct__

   MaskedArray.view
   MaskedArray.astype
   MaskedArray.byteswap

   MaskedArray.compressed
   MaskedArray.filled
   MaskedArray.tofile
   MaskedArray.toflex
   MaskedArray.tolist
   MaskedArray.torecords
   MaskedArray.tostring


Shape manipulation
------------------

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

.. autosummary::
   :toctree: generated/

   MaskedArray.flatten
   MaskedArray.ravel
   MaskedArray.reshape
   MaskedArray.resize
   MaskedArray.squeeze
   MaskedArray.swapaxes
   MaskedArray.transpose
   MaskedArray.T


Item selection and manipulation
-------------------------------

For array methods that take an *axis* keyword, it defaults to `None`.
If axis is *None*, then the array is treated as a 1-D array.
Any other value for *axis* represents the dimension along which
the operation should proceed.

.. autosummary::
   :toctree: generated/

   MaskedArray.argmax
   MaskedArray.argmin
   MaskedArray.argsort
   MaskedArray.choose
   MaskedArray.compress
   MaskedArray.diagonal
   MaskedArray.fill
   MaskedArray.item
   MaskedArray.nonzero
   MaskedArray.put
   MaskedArray.repeat
   MaskedArray.searchsorted
   MaskedArray.sort
   MaskedArray.take


Pickling and copy
-----------------

.. autosummary::
   :toctree: generated/

   MaskedArray.copy
   MaskedArray.dump
   MaskedArray.dumps


Calculations
------------

.. autosummary::
   :toctree: generated/

   MaskedArray.all
   MaskedArray.anom
   MaskedArray.any
   MaskedArray.clip
   MaskedArray.conj
   MaskedArray.conjugate
   MaskedArray.cumprod
   MaskedArray.cumsum
   MaskedArray.max
   MaskedArray.mean
   MaskedArray.min
   MaskedArray.prod
   MaskedArray.product
   MaskedArray.ptp
   MaskedArray.round
   MaskedArray.std
   MaskedArray.sum
   MaskedArray.trace
   MaskedArray.var


Arithmetic and comparison operations
------------------------------------

.. index:: comparison, arithmetic, operation, operator

Comparison operators:
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MaskedArray.__lt__
   MaskedArray.__le__
   MaskedArray.__gt__
   MaskedArray.__ge__
   MaskedArray.__eq__
   MaskedArray.__ne__

Truth value of an array (:func:`bool()`):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MaskedArray.__nonzero__


Arithmetic:
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MaskedArray.__abs__
   MaskedArray.__add__
   MaskedArray.__radd__
   MaskedArray.__sub__
   MaskedArray.__rsub__
   MaskedArray.__mul__
   MaskedArray.__rmul__
   MaskedArray.__div__
   MaskedArray.__rdiv__
   MaskedArray.__truediv__
   MaskedArray.__rtruediv__
   MaskedArray.__floordiv__
   MaskedArray.__rfloordiv__
   MaskedArray.__mod__
   MaskedArray.__rmod__
   MaskedArray.__divmod__
   MaskedArray.__rdivmod__
   MaskedArray.__pow__
   MaskedArray.__rpow__
   MaskedArray.__lshift__
   MaskedArray.__rlshift__
   MaskedArray.__rshift__
   MaskedArray.__rrshift__
   MaskedArray.__and__
   MaskedArray.__rand__
   MaskedArray.__or__
   MaskedArray.__ror__
   MaskedArray.__xor__
   MaskedArray.__rxor__


Arithmetic, in-place:
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MaskedArray.__iadd__
   MaskedArray.__isub__
   MaskedArray.__imul__
   MaskedArray.__idiv__
   MaskedArray.__itruediv__
   MaskedArray.__ifloordiv__
   MaskedArray.__imod__
   MaskedArray.__ipow__
   MaskedArray.__ilshift__
   MaskedArray.__irshift__
   MaskedArray.__iand__
   MaskedArray.__ior__
   MaskedArray.__ixor__


Representation
--------------

.. autosummary::
   :toctree: generated/

   MaskedArray.__repr__
   MaskedArray.__str__

   MaskedArray.ids
   MaskedArray.iscontiguous


Special methods
---------------

For standard library functions:

.. autosummary::
   :toctree: generated/

   MaskedArray.__copy__
   MaskedArray.__deepcopy__
   MaskedArray.__getstate__
   MaskedArray.__reduce__
   MaskedArray.__setstate__

Basic customization:

.. autosummary::
   :toctree: generated/

   MaskedArray.__new__
   MaskedArray.__array__
   MaskedArray.__array_wrap__

Container customization: (see :ref:`Indexing <arrays.indexing>`)

.. autosummary::
   :toctree: generated/

   MaskedArray.__len__
   MaskedArray.__getitem__
   MaskedArray.__setitem__
   MaskedArray.__delitem__
   MaskedArray.__getslice__
   MaskedArray.__setslice__
   MaskedArray.__contains__



Specific methods
----------------

Handling the mask
~~~~~~~~~~~~~~~~~

The following methods can be used to access information about the mask or to
manipulate the mask.

.. autosummary::
   :toctree: generated/

   MaskedArray.__setmask__

   MaskedArray.harden_mask
   MaskedArray.soften_mask
   MaskedArray.unshare_mask
   MaskedArray.shrink_mask


Handling the `fill_value`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MaskedArray.get_fill_value
   MaskedArray.set_fill_value



Counting the missing elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MaskedArray.count
