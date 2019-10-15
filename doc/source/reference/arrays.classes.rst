.. _arrays.classes:

#########################
Standard array subclasses
#########################

.. currentmodule:: numpy

.. note::

    Subclassing a ``numpy.ndarray`` is possible but if your goal is to create
    an array with *modified* behavior, as do dask arrays for distributed
    computation and cupy arrays for GPU-based computation, subclassing is
    discouraged. Instead, using numpy's
    :ref:`dispatch mechanism <basics.dispatch>` is recommended.

The :class:`ndarray` can be inherited from (in Python or in C)
if desired. Therefore, it can form a foundation for many useful
classes. Often whether to sub-class the array object or to simply use
the core array component as an internal part of a new class is a
difficult decision, and can be simply a matter of choice. NumPy has
several tools for simplifying how your new object interacts with other
array objects, and so the choice may not be significant in the
end. One way to simplify the question is by asking yourself if the
object you are interested in can be replaced as a single array or does
it really require two or more arrays at its core.

Note that :func:`asarray` always returns the base-class ndarray. If
you are confident that your use of the array object can handle any
subclass of an ndarray, then :func:`asanyarray` can be used to allow
subclasses to propagate more cleanly through your subroutine. In
principal a subclass could redefine any aspect of the array and
therefore, under strict guidelines, :func:`asanyarray` would rarely be
useful. However, most subclasses of the array object will not
redefine certain aspects of the array object such as the buffer
interface, or the attributes of the array. One important example,
however, of why your subroutine may not be able to handle an arbitrary
subclass of an array is that matrices redefine the "*" operator to be
matrix-multiplication, rather than element-by-element multiplication.


Special attributes and methods
==============================

.. seealso:: :ref:`Subclassing ndarray <basics.subclassing>`

NumPy provides several hooks that classes can customize:

.. py:method:: class.__array_ufunc__(ufunc, method, *inputs, **kwargs)

   .. versionadded:: 1.13

   Any class, ndarray subclass or not, can define this method or set it to
   None in order to override the behavior of NumPy's ufuncs. This works
   quite similarly to Python's ``__mul__`` and other binary operation routines.

   - *ufunc* is the ufunc object that was called.
   - *method* is a string indicating which Ufunc method was called
     (one of ``"__call__"``, ``"reduce"``, ``"reduceat"``,
     ``"accumulate"``, ``"outer"``, ``"inner"``).
   - *inputs* is a tuple of the input arguments to the ``ufunc``.
   - *kwargs* is a dictionary containing the optional input arguments
     of the ufunc. If given, any ``out`` arguments, both positional
     and keyword, are passed as a :obj:`tuple` in *kwargs*. See the
     discussion in :ref:`ufuncs` for details.

   The method should return either the result of the operation, or
   :obj:`NotImplemented` if the operation requested is not implemented.

   If one of the input or output arguments has a :func:`__array_ufunc__`
   method, it is executed *instead* of the ufunc.  If more than one of the
   arguments implements :func:`__array_ufunc__`, they are tried in the
   order: subclasses before superclasses, inputs before outputs, otherwise
   left to right. The first routine returning something other than
   :obj:`NotImplemented` determines the result. If all of the
   :func:`__array_ufunc__` operations return :obj:`NotImplemented`, a
   :exc:`TypeError` is raised.

   .. note:: We intend to re-implement numpy functions as (generalized)
       Ufunc, in which case it will become possible for them to be
       overridden by the ``__array_ufunc__`` method.  A prime candidate is
       :func:`~numpy.matmul`, which currently is not a Ufunc, but could be
       relatively easily be rewritten as a (set of) generalized Ufuncs. The
       same may happen with functions such as :func:`~numpy.median`,
       :func:`~numpy.amin`, and :func:`~numpy.argsort`.

   Like with some other special methods in python, such as ``__hash__`` and
   ``__iter__``, it is possible to indicate that your class does *not*
   support ufuncs by setting ``__array_ufunc__ = None``. Ufuncs always raise
   :exc:`TypeError` when called on an object that sets
   ``__array_ufunc__ = None``.

   The presence of :func:`__array_ufunc__` also influences how
   :class:`ndarray` handles binary operations like ``arr + obj`` and ``arr
   < obj`` when ``arr`` is an :class:`ndarray` and ``obj`` is an instance
   of a custom class. There are two possibilities. If
   ``obj.__array_ufunc__`` is present and not None, then
   ``ndarray.__add__`` and friends will delegate to the ufunc machinery,
   meaning that ``arr + obj`` becomes ``np.add(arr, obj)``, and then
   :func:`~numpy.add` invokes ``obj.__array_ufunc__``. This is useful if you
   want to define an object that acts like an array.

   Alternatively, if ``obj.__array_ufunc__`` is set to None, then as a
   special case, special methods like ``ndarray.__add__`` will notice this
   and *unconditionally* raise :exc:`TypeError`. This is useful if you want to
   create objects that interact with arrays via binary operations, but
   are not themselves arrays. For example, a units handling system might have
   an object ``m`` representing the "meters" unit, and want to support the
   syntax ``arr * m`` to represent that the array has units of "meters", but
   not want to otherwise interact with arrays via ufuncs or otherwise. This
   can be done by setting ``__array_ufunc__ = None`` and defining ``__mul__``
   and ``__rmul__`` methods. (Note that this means that writing an
   ``__array_ufunc__`` that always returns :obj:`NotImplemented` is not
   quite the same as setting ``__array_ufunc__ = None``: in the former
   case, ``arr + obj`` will raise :exc:`TypeError`, while in the latter
   case it is possible to define a ``__radd__`` method to prevent this.)

   The above does not hold for in-place operators, for which :class:`ndarray`
   never returns :obj:`NotImplemented`.  Hence, ``arr += obj`` would always
   lead to a :exc:`TypeError`.  This is because for arrays in-place operations
   cannot generically be replaced by a simple reverse operation.  (For
   instance, by default, ``arr += obj`` would be translated to ``arr =
   arr + obj``, i.e., ``arr`` would be replaced, contrary to what is expected
   for in-place array operations.)

   .. note:: If you define ``__array_ufunc__``:

      - If you are not a subclass of :class:`ndarray`, we recommend your
        class define special methods like ``__add__`` and ``__lt__`` that
        delegate to ufuncs just like ndarray does.  An easy way to do this
        is to subclass from :class:`~numpy.lib.mixins.NDArrayOperatorsMixin`.
      - If you subclass :class:`ndarray`, we recommend that you put all your
        override logic in ``__array_ufunc__`` and not also override special
        methods. This ensures the class hierarchy is determined in only one
        place rather than separately by the ufunc machinery and by the binary
        operation rules (which gives preference to special methods of
        subclasses; the alternative way to enforce a one-place only hierarchy,
        of setting :func:`__array_ufunc__` to None, would seem very
        unexpected and thus confusing, as then the subclass would not work at
        all with ufuncs).
      - :class:`ndarray` defines its own :func:`__array_ufunc__`, which,
        evaluates the ufunc if no arguments have overrides, and returns
        :obj:`NotImplemented` otherwise. This may be useful for subclasses
        for which :func:`__array_ufunc__` converts any instances of its own
        class to :class:`ndarray`: it can then pass these on to its
        superclass using ``super().__array_ufunc__(*inputs, **kwargs)``,
        and finally return the results after possible back-conversion. The
        advantage of this practice is that it ensures that it is possible
        to have a hierarchy of subclasses that extend the behaviour. See
        :ref:`Subclassing ndarray <basics.subclassing>` for details.

   .. note:: If a class defines the :func:`__array_ufunc__` method,
      this disables the :func:`__array_wrap__`,
      :func:`__array_prepare__`, :data:`__array_priority__` mechanism
      described below for ufuncs (which may eventually be deprecated).

.. py:method:: class.__array_function__(func, types, args, kwargs)

   .. versionadded:: 1.16

   .. note::

       - In NumPy 1.17, the protocol is enabled by default, but can be disabled
         with ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0``.
       - In NumPy 1.16, you need to set the environment variable
         ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1`` before importing NumPy to use
         NumPy function overrides.
       - Eventually, expect to ``__array_function__`` to always be enabled.

   -  ``func`` is an arbitrary callable exposed by NumPy's public API,
      which was called in the form ``func(*args, **kwargs)``.
   -  ``types`` is a `collection <collections.abc.Collection>`_
      of unique argument types from the original NumPy function call that
      implement ``__array_function__``.
   -  The tuple ``args`` and dict ``kwargs`` are directly passed on from the
      original call.

   As a convenience for ``__array_function__`` implementors, ``types``
   provides all argument types with an ``'__array_function__'`` attribute.
   This allows implementors to quickly identify cases where they should defer
   to ``__array_function__`` implementations on other arguments.
   Implementations should not rely on the iteration order of ``types``.

   Most implementations of ``__array_function__`` will start with two
   checks:

   1.  Is the given function something that we know how to overload?
   2.  Are all arguments of a type that we know how to handle?

   If these conditions hold, ``__array_function__`` should return the result
   from calling its implementation for ``func(*args, **kwargs)``.  Otherwise,
   it should return the sentinel value ``NotImplemented``, indicating that the
   function is not implemented by these types.

   There are no general requirements on the return value from
   ``__array_function__``, although most sensible implementations should
   probably return array(s) with the same type as one of the function's
   arguments.

   It may also be convenient to define a custom decorators (``implements``
   below) for registering ``__array_function__`` implementations.

   .. code:: python

       HANDLED_FUNCTIONS = {}

       class MyArray:
           def __array_function__(self, func, types, args, kwargs):
               if func not in HANDLED_FUNCTIONS:
                   return NotImplemented
               # Note: this allows subclasses that don't override
               # __array_function__ to handle MyArray objects
               if not all(issubclass(t, MyArray) for t in types):
                   return NotImplemented
               return HANDLED_FUNCTIONS[func](*args, **kwargs)

       def implements(numpy_function):
           """Register an __array_function__ implementation for MyArray objects."""
           def decorator(func):
               HANDLED_FUNCTIONS[numpy_function] = func
               return func
           return decorator

       @implements(np.concatenate)
       def concatenate(arrays, axis=0, out=None):
           ...  # implementation of concatenate for MyArray objects

       @implements(np.broadcast_to)
       def broadcast_to(array, shape):
           ...  # implementation of broadcast_to for MyArray objects

   Note that it is not required for ``__array_function__`` implementations to
   include *all* of the corresponding NumPy function's optional arguments
   (e.g., ``broadcast_to`` above omits the irrelevant ``subok`` argument).
   Optional arguments are only passed in to ``__array_function__`` if they
   were explicitly used in the NumPy function call.

   Just like the case for builtin special methods like ``__add__``, properly
   written ``__array_function__`` methods should always return
   ``NotImplemented`` when an unknown type is encountered. Otherwise, it will
   be impossible to correctly override NumPy functions from another object
   if the operation also includes one of your objects.

   For the most part, the rules for dispatch with ``__array_function__``
   match those for ``__array_ufunc__``. In particular:

   -  NumPy will gather implementations of ``__array_function__`` from all
      specified inputs and call them in order: subclasses before
      superclasses, and otherwise left to right. Note that in some edge cases
      involving subclasses, this differs slightly from the
      `current behavior <https://bugs.python.org/issue30140>`_ of Python.
   -  Implementations of ``__array_function__`` indicate that they can
      handle the operation by returning any value other than
      ``NotImplemented``.
   -  If all ``__array_function__`` methods return ``NotImplemented``,
      NumPy will raise ``TypeError``.

   If no ``__array_function__`` methods exists, NumPy will default to calling
   its own implementation, intended for use on NumPy arrays. This case arises,
   for example, when all array-like arguments are Python numbers or lists.
   (NumPy arrays do have a ``__array_function__`` method, given below, but it
   always returns ``NotImplemented`` if any argument other than a NumPy array
   subclass implements ``__array_function__``.)

   One deviation from the current behavior of ``__array_ufunc__`` is that
   NumPy will only call ``__array_function__`` on the *first* argument of each
   unique type. This matches Python's `rule for calling reflected methods
   <https://docs.python.org/3/reference/datamodel.html#object.__ror__>`_, and
   this ensures that checking overloads has acceptable performance even when
   there are a large number of overloaded arguments.

.. py:method:: class.__array_finalize__(obj)

   This method is called whenever the system internally allocates a
   new array from *obj*, where *obj* is a subclass (subtype) of the
   :class:`ndarray`. It can be used to change attributes of *self*
   after construction (so as to ensure a 2-d matrix for example), or
   to update meta-information from the "parent." Subclasses inherit
   a default implementation of this method that does nothing.

.. py:method:: class.__array_prepare__(array, context=None)

   At the beginning of every :ref:`ufunc <ufuncs-output-type>`, this
   method is called on the input object with the highest array
   priority, or the output object if one was specified. The output
   array is passed in and whatever is returned is passed to the ufunc.
   Subclasses inherit a default implementation of this method which
   simply returns the output array unmodified. Subclasses may opt to
   use this method to transform the output array into an instance of
   the subclass and update metadata before returning the array to the
   ufunc for computation.

   .. note:: For ufuncs, it is hoped to eventually deprecate this method in
             favour of :func:`__array_ufunc__`.

.. py:method:: class.__array_wrap__(array, context=None)

   At the end of every :ref:`ufunc <ufuncs-output-type>`, this method
   is called on the input object with the highest array priority, or
   the output object if one was specified. The ufunc-computed array
   is passed in and whatever is returned is passed to the user.
   Subclasses inherit a default implementation of this method, which
   transforms the array into a new instance of the object's class.
   Subclasses may opt to use this method to transform the output array
   into an instance of the subclass and update metadata before
   returning the array to the user.

   .. note:: For ufuncs, it is hoped to eventually deprecate this method in
             favour of :func:`__array_ufunc__`.

.. py:attribute:: class.__array_priority__

   The value of this attribute is used to determine what type of
   object to return in situations where there is more than one
   possibility for the Python type of the returned object. Subclasses
   inherit a default value of 0.0 for this attribute.

   .. note:: For ufuncs, it is hoped to eventually deprecate this method in
             favour of :func:`__array_ufunc__`.

.. py:method:: class.__array__([dtype])

   If a class (ndarray subclass or not) having the :func:`__array__`
   method is used as the output object of an :ref:`ufunc
   <ufuncs-output-type>`, results will be written to the object
   returned by :func:`__array__`. Similar conversion is done on
   input arrays.


Matrix objects
==============

.. index::
   single: matrix

.. note::
   It is strongly advised *not* to use the matrix subclass.  As described
   below, it makes writing functions that deal consistently with matrices
   and regular arrays very difficult. Currently, they are mainly used for
   interacting with ``scipy.sparse``. We hope to provide an alternative
   for this use, however, and eventually remove the ``matrix`` subclass.

:class:`matrix` objects inherit from the ndarray and therefore, they
have the same attributes and methods of ndarrays. There are six
important differences of matrix objects, however, that may lead to
unexpected results when you use matrices but expect them to act like
arrays:

1. Matrix objects can be created using a string notation to allow
   Matlab-style syntax where spaces separate columns and semicolons
   (';') separate rows.

2. Matrix objects are always two-dimensional. This has far-reaching
   implications, in that m.ravel() is still two-dimensional (with a 1
   in the first dimension) and item selection returns two-dimensional
   objects so that sequence behavior is fundamentally different than
   arrays.

3. Matrix objects over-ride multiplication to be
   matrix-multiplication. **Make sure you understand this for
   functions that you may want to receive matrices. Especially in
   light of the fact that asanyarray(m) returns a matrix when m is
   a matrix.**

4. Matrix objects over-ride power to be matrix raised to a power. The
   same warning about using power inside a function that uses
   asanyarray(...) to get an array object holds for this fact.

5. The default __array_priority\__ of matrix objects is 10.0, and
   therefore mixed operations with ndarrays always produce matrices.

6. Matrices have special attributes which make calculations easier.
   These are

   .. autosummary::
      :toctree: generated/

      matrix.T
      matrix.H
      matrix.I
      matrix.A

.. warning::

    Matrix objects over-ride multiplication, '*', and power, '**', to
    be matrix-multiplication and matrix power, respectively. If your
    subroutine can accept sub-classes and you do not convert to base-
    class arrays, then you must use the ufuncs multiply and power to
    be sure that you are performing the correct operation for all
    inputs.

The matrix class is a Python subclass of the ndarray and can be used
as a reference for how to construct your own subclass of the ndarray.
Matrices can be created from other matrices, strings, and anything
else that can be converted to an ``ndarray`` . The name "mat "is an
alias for "matrix "in NumPy.

.. autosummary::
   :toctree: generated/

   matrix
   asmatrix
   bmat

Example 1: Matrix creation from a string

>>> a=mat('1 2 3; 4 5 3')
>>> print (a*a.T).I
[[ 0.2924 -0.1345]
 [-0.1345  0.0819]]

Example 2: Matrix creation from nested sequence

>>> mat([[1,5,10],[1.0,3,4j]])
matrix([[  1.+0.j,   5.+0.j,  10.+0.j],
        [  1.+0.j,   3.+0.j,   0.+4.j]])

Example 3: Matrix creation from an array

>>> mat(random.rand(3,3)).T
matrix([[ 0.7699,  0.7922,  0.3294],
        [ 0.2792,  0.0101,  0.9219],
        [ 0.3398,  0.7571,  0.8197]])

Memory-mapped file arrays
=========================

.. index::
   single: memory maps

.. currentmodule:: numpy

Memory-mapped files are useful for reading and/or modifying small
segments of a large file with regular layout, without reading the
entire file into memory. A simple subclass of the ndarray uses a
memory-mapped file for the data buffer of the array. For small files,
the over-head of reading the entire file into memory is typically not
significant, however for large files using memory mapping can save
considerable resources.

Memory-mapped-file arrays have one additional method (besides those
they inherit from the ndarray): :meth:`.flush() <memmap.flush>` which
must be called manually by the user to ensure that any changes to the
array actually get written to disk.

.. autosummary::
   :toctree: generated/

   memmap
   memmap.flush

Example:

>>> a = memmap('newfile.dat', dtype=float, mode='w+', shape=1000)
>>> a[10] = 10.0
>>> a[30] = 30.0
>>> del a
>>> b = fromfile('newfile.dat', dtype=float)
>>> print b[10], b[30]
10.0 30.0
>>> a = memmap('newfile.dat', dtype=float)
>>> print a[10], a[30]
10.0 30.0


Character arrays (:mod:`numpy.char`)
====================================

.. seealso:: :ref:`routines.array-creation.char`

.. index::
   single: character arrays

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `string_` or `unicode_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

These are enhanced arrays of either :class:`string_` type or
:class:`unicode_` type.  These arrays inherit from the
:class:`ndarray`, but specially-define the operations ``+``, ``*``,
and ``%`` on a (broadcasting) element-by-element basis.  These
operations are not available on the standard :class:`ndarray` of
character type. In addition, the :class:`chararray` has all of the
standard :class:`string <str>` (and :class:`unicode`) methods,
executing them on an element-by-element basis. Perhaps the easiest
way to create a chararray is to use :meth:`self.view(chararray)
<ndarray.view>` where *self* is an ndarray of str or unicode
data-type. However, a chararray can also be created using the
:meth:`numpy.chararray` constructor, or via the
:func:`numpy.char.array <core.defchararray.array>` function:

.. autosummary::
   :toctree: generated/

   chararray
   core.defchararray.array

Another difference with the standard ndarray of str data-type is
that the chararray inherits the feature introduced by Numarray that
white-space at the end of any element in the array will be ignored
on item retrieval and comparison operations.


.. _arrays.classes.rec:

Record arrays (:mod:`numpy.rec`)
================================

.. seealso:: :ref:`routines.array-creation.rec`, :ref:`routines.dtype`,
             :ref:`arrays.dtypes`.

NumPy provides the :class:`recarray` class which allows accessing the
fields of a structured array as attributes, and a corresponding
scalar data type object :class:`record`.

.. currentmodule:: numpy

.. autosummary::
   :toctree: generated/

   recarray
   record

Masked arrays (:mod:`numpy.ma`)
===============================

.. seealso:: :ref:`maskedarray`

Standard container class
========================

.. currentmodule:: numpy

For backward compatibility and as a standard "container "class, the
UserArray from Numeric has been brought over to NumPy and named
:class:`numpy.lib.user_array.container` The container class is a
Python class whose self.array attribute is an ndarray. Multiple
inheritance is probably easier with numpy.lib.user_array.container
than with the ndarray itself and so it is included by default. It is
not documented here beyond mentioning its existence because you are
encouraged to use the ndarray class directly if you can.

.. autosummary::
   :toctree: generated/

   numpy.lib.user_array.container

.. index::
   single: user_array
   single: container class


Array Iterators
===============

.. currentmodule:: numpy

.. index::
   single: array iterator

Iterators are a powerful concept for array processing. Essentially,
iterators implement a generalized for-loop. If *myiter* is an iterator
object, then the Python code::

    for val in myiter:
        ...
        some code involving val
        ...

calls ``val = next(myiter)`` repeatedly until :exc:`StopIteration` is
raised by the iterator. There are several ways to iterate over an
array that may be useful: default iteration, flat iteration, and
:math:`N`-dimensional enumeration.


Default iteration
-----------------

The default iterator of an ndarray object is the default Python
iterator of a sequence type. Thus, when the array object itself is
used as an iterator. The default behavior is equivalent to::

    for i in range(arr.shape[0]):
        val = arr[i]

This default iterator selects a sub-array of dimension :math:`N-1`
from the array. This can be a useful construct for defining recursive
algorithms. To loop over the entire array requires :math:`N` for-loops.

>>> a = arange(24).reshape(3,2,4)+10
>>> for val in a:
...     print 'item:', val
item: [[10 11 12 13]
 [14 15 16 17]]
item: [[18 19 20 21]
 [22 23 24 25]]
item: [[26 27 28 29]
 [30 31 32 33]]


Flat iteration
--------------

.. autosummary::
   :toctree: generated/

   ndarray.flat

As mentioned previously, the flat attribute of ndarray objects returns
an iterator that will cycle over the entire array in C-style
contiguous order.

>>> for i, val in enumerate(a.flat):
...     if i%5 == 0: print i, val
0 10
5 15
10 20
15 25
20 30

Here, I've used the built-in enumerate iterator to return the iterator
index as well as the value.


N-dimensional enumeration
-------------------------

.. autosummary::
   :toctree: generated/

   ndenumerate

Sometimes it may be useful to get the N-dimensional index while
iterating. The ndenumerate iterator can achieve this.

>>> for i, val in ndenumerate(a):
...     if sum(i)%5 == 0: print i, val
(0, 0, 0) 10
(1, 1, 3) 25
(2, 0, 3) 29
(2, 1, 2) 32


Iterator for broadcasting
-------------------------

.. autosummary::
   :toctree: generated/

   broadcast

The general concept of broadcasting is also available from Python
using the :class:`broadcast` iterator. This object takes :math:`N`
objects as inputs and returns an iterator that returns tuples
providing each of the input sequence elements in the broadcasted
result.

>>> for val in broadcast([[1,0],[2,3]],[0,1]):
...     print val
(1, 0)
(0, 1)
(2, 0)
(3, 1)
