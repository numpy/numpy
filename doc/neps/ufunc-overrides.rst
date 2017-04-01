=================================
A Mechanism for Overriding Ufuncs
=================================

.. currentmodule:: numpy

:Author: Blake Griffith
:Contact: blake.g@utexas.edu 
:Date: 2013-07-10

:Author: Pauli Virtanen

:Author: Nathaniel Smith

:Author: Marten van Kerkwijk
:Date: 2017-03-31

Executive summary
=================

NumPy's universal functions (ufuncs) currently have some limited
functionality for operating on user defined subclasses of
:class:`ndarray` using ``__array_prepare__`` and ``__array_wrap__``
[1]_, and there is little to no support for arbitrary
objects. e.g. SciPy's sparse matrices [2]_ [3]_.

Here we propose adding a mechanism to override ufuncs based on the ufunc
checking each of it's arguments for a ``__array_ufunc__`` method.
On discovery of ``__array_ufunc__`` the ufunc will hand off the
operation to the method. 

This covers some of the same ground as Travis Oliphant's proposal to
retro-fit NumPy with multi-methods [4]_, which would solve the same
problem. The mechanism here follows more closely the way Python enables
classes to override ``__mul__`` and other binary operations. It also
specifically addresses how binary operators and ufuncs should interact.

.. note:: In earlier iterations, the override was called
          ``__numpy_ufunc__``. An implementation was made, but had not
          quite the right behaviour, hence the change in name.

.. [1] http://docs.python.org/doc/numpy/user/basics.subclassing.html
.. [2] https://github.com/scipy/scipy/issues/2123
.. [3] https://github.com/scipy/scipy/issues/1569
.. [4] http://technicaldiscovery.blogspot.com/2013/07/thoughts-after-scipy-2013-and-specific.html


Motivation
==========

The current machinery for dispatching Ufuncs is generally agreed to be
insufficient. There have been lengthy discussions and other proposed
solutions [5]_, [6]_.

Using ufuncs with subclasses of :class:`ndarray` is limited to
``__array_prepare__`` and ``__array_wrap__`` to prepare the arguments,
but these don't allow you to for example change the shape or the data of
the arguments. Trying to ufunc things that don't subclass
:class:`ndarray` is even more difficult, as the input arguments tend to
be cast to object arrays, which ends up producing surprising results.

Take this example of ufuncs interoperability with sparse matrices.::

    In [1]: import numpy as np
    import scipy.sparse as sp

    a = np.random.randint(5, size=(3,3))
    b = np.random.randint(5, size=(3,3))

    asp = sp.csr_matrix(a)
    bsp = sp.csr_matrix(b)

    In [2]: a, b
    Out[2]:(array([[0, 4, 4],
                   [1, 3, 2],
                   [1, 3, 1]]),
            array([[0, 1, 0],
                   [0, 0, 1],
                   [4, 0, 1]]))

    In [3]: np.multiply(a, b) # The right answer
    Out[3]: array([[0, 4, 0],
                   [0, 0, 2],
                   [4, 0, 1]])

    In [4]: np.multiply(asp, bsp).todense() # calls __mul__ which does matrix multi
    Out[4]: matrix([[16,  0,  8],
                    [ 8,  1,  5],
                    [ 4,  1,  4]], dtype=int64)
                    
    In [5]: np.multiply(a, bsp) # Returns NotImplemented to user, bad!
    Out[5]: NotImplemted

Returning :obj:`NotImplemented` to user should not happen. Moreover::

    In [6]: np.multiply(asp, b)
    Out[6]: array([[ <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>],
                       [ <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>],
                       [ <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>]], dtype=object)

Here, it appears that the sparse matrix was converted to an object array
scalar, which was then multiplied with all elements of the ``b`` array.
However, this behavior is more confusing than useful, and having a
:exc:`TypeError` would be preferable.

Adding the ``__array_ufunc__`` functionality fixes this and would
deprecate the other ufunc modifying functions.

.. [5] http://mail.python.org/pipermail/numpy-discussion/2011-June/056945.html

.. [6] https://github.com/numpy/numpy/issues/5844

Proposed interface
==================

The standard array class :class:`ndarray` gains an ``__array_ufunc__``
method and objects can override Ufuncs by overriding this method (if
they are :class:`ndarray` subclasses) or defining their own. The method
signature is::

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs)

Here:

- *ufunc* is the ufunc object that was called. 
- *method* is a string indicating which Ufunc method was called
  (one of ``"__call__"``, ``"reduce"``, ``"reduceat"``,
  ``"accumulate"``, ``"outer"``, ``"inner"``). 
- *inputs* is a tuple of the input arguments to the ``ufunc``
- *kwargs* are the keyword arguments passed to the function. The ``out``
  arguments are always contained as a tuple in *kwargs*.

Hence, the arguments are normalized: only the input data (``inputs``)
are passed on as positional arguments, all the others are passed on as a
dict of keyword arguments (``kwargs``). In particular, if there are
output arguments, positional are otherwise, they are passed on as a
tuple in the ``out`` keyword argument.

The function dispatch proceeds as follows:

- If an input argument has a ``__array_ufunc__`` attribute, but its
  value is ``ndarray.__array_ufunc__``, the attribute is considered to
  be absent in what follows.  This happens for instances of `ndarray`
  and those `ndarray` subclasses that did not override their inherited
  ``__array_ufunc__`` implementation.

- If one of the input arguments implements ``__array_ufunc__``, it is
  executed instead of the ufunc.

- If more than one of the input arguments implements ``__array_ufunc__``,
  they are tried in the following order: subclasses before superclasses,
  otherwise left to right.

- The first ``__array_ufunc__`` method returning something else than
  :obj:`NotImplemented` determines the return value of the Ufunc.

- If all ``__array_ufunc__`` methods of the input arguments return
  :obj:`NotImplemented`, a :exc:`TypeError` is raised.

- If a ``__array_ufunc__`` method raises an error, the error is
  propagated immediately.

- If none of the input arguments had an ``__array_ufunc__`` method, the
  execution falls back on the default ufunc behaviour.


Type casting hierarchy
----------------------

Similarly to the Python operator dispatch mechanism, writing ufunc
dispatch methods requires some discipline in order to achieve
predictable results.

In particular, it is useful to maintain a clear idea of what types can
be upcast to others, possibly indirectly (i.e. A->B->C is implemented
but direct A->C not). Moreover, one should make sure the implementations of
``__array_ufunc__``, which implicitly define the type casting hierarchy,
don't contradict this.

The following rules should be followed:

1. The ``__array_ufunc__`` for type A should either return
   `NotImplemented`, or return an output of type A (unless an
   ``out=`` argument was given, in which case ``out`` is returned).

2. For any two different types *A*, *B*, the relation "A can handle B" 
   defined as::

       a.__array_ufunc__(..., b, ...) is not NotImplemented

   for instances *a* and *b* of *A* and *B*, defines the
   edges B->A of a graph.

   This graph must be a directed acyclic graph.

Under these conditions, the transitive closure of the "can handle"
relation defines a strict partial ordering of the types -- that is, the
type casting hierarchy.

In other words, for any given class A, all other classes that define
``__array_ufunc__`` must belong to exactly one of the groups:

- *Above A*: their ``__array_ufunc__`` can handle class A or some
  member of the "above A" classes. In other words, these are the types
  that A can be (indirectly) upcast to in ufuncs.

- *Below A*: they can be handled by the ``__array_ufunc__`` of class A
  or the ``__array_ufunc__`` of some member of the "below A" classes. In
  other words, these are the types that can be (indirectly) upcast to A
  in ufuncs.

- *Incompatible*: neither above nor below A; types for which no
  (indirect) upcasting is possible.

This guarantees that expressions involving ufuncs either raise a
`TypeError`, or the result type is independent of what ufuncs were
called, what order they were called in, and what order their arguments
were in.  Moreover, which ``__array_ufunc__`` payload code runs at each
step is independent of the order of arguments of the ufuncs.

Note also that while converting inputs that don't have
``__array_ufunc__`` to `ndarray` via `np.asarray` is consistent with the
type casting hierarchy, also returning `NotImplemented` is
consistent. However, the numpy ufunc (legacy) behavior is to try to
convert unknown objects to ndarrays.


.. admonition:: Example

   Type casting hierarchy

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> C;
         B -> C;
         D -> B;
         ndarray -> A;
         ndarray -> B;
      }

   The ``__array_ufunc__`` of type A can handle ndarrays, B can handle ndarray and D,
   and C can handle A and B but not ndarrays or D.  The resulting graph is a DAG,
   and defines a type casting hierarchy, with relations ``C > A >
   ndarray``, ``C > B > ndarray``, ``C > B > D``. The type B is incompatible
   relative to A and vice versa, and A and ndarray are incompatible relative to D.
   Ufunc expressions involving these classes produce results of the highest type
   involved or raise a TypeError.


Subclass hierarchies
--------------------

Generally, it is desirable to mirror the class hierarchy in the ufunc
type casting hierarchy. The recommendation is that an
``__array_ufunc__`` implementation of a class should generally return
`NotImplemented` unless the inputs are instances of the same class or
superclasses.  This guarantees that in the type casting hierarchy,
superclasses are below, subclasses above, and other classes are
incompatible.  Exceptions to this need to check they respect the
implicit type casting hierarchy.

Subclasses can be easily constructed if methods consistently use
:func:`super` to pass through the class hierarchy [7]_.  To support
this, :class:`ndarray` has its own ``__array_ufunc__`` method,
equivalent to::

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop('out', None)
        out_tuple = out if out is not None else ()

        # Handle items of type(self), superclasses, and items
        # without __array_ufunc__. Bail out in other cases.
        items = []
        for item in inputs + out_tuple:
            if isinstance(self, type(item)) or not hasattr(item, '__array_ufunc__'):
                # Cast to plain ndarrays
                items.append(np.asarray(item))
            else:
                return NotImplemented

        # Perform ufunc on the underlying ndarrays (no __array_ufunc__ dispatch)
        result = getattr(ufunc, method)(*items, **kwargs)

        # Cast output to type(self), unless `out` specified
        if out is not None:
            return result

        if isinstance(result, tuple):
            return tuple(x.view(type(self)) for x in result)
        else:
            return result.view(type(self))

Note that, as a special case, the ufunc dispatch mechanism does not call
this `ndarray.__array_ufunc__` method, even for `ndarray` subclasses
if they have not overridden the default `ndarray` implementation. As a
consequence, calling `ndarray.__array_ufunc__` will not result to a
nested ufunc dispatch cycle.  Custom implementations of
`__array_ufunc__` should generally avoid nested dispatch cycles.

This should be particularly useful for subclasses of :class:`ndarray`,
which only add an attribute like a unit or mask to a regular
:class:`ndarray`. In their `__array_ufunc__` implementation, such
classes can do possible adjustment of the arguments relevant to their
own class, and pass on to superclass implementation using :func:`super`
until the ufunc is actually done, and then do possible adjustments of
the outputs.

Turning Ufuncs off
------------------

For some classes, Ufuncs make no sense, and, like for other special
methods [8]_, one can indicate Ufuncs are not available by setting
``__array_ufunc__`` to :obj:`None`.  Inside a Ufunc, this is
equivalent to unconditionally returning :obj:`NotImplemented`, and thus
will lead to a :exc:`TypeError` (unless another operand implements
``__array_ufunc__`` and specifically knows how to deal with the class).

In the type casting hierarchy, this makes the type incompatible relative
to `ndarray`.

.. [7] https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

.. [8] https://docs.python.org/3/reference/datamodel.html#specialnames

In combination with Python's binary operations
----------------------------------------------

The ``__array_ufunc__`` mechanism is fully independent of Python's
standard operator override mechanism, and the two do not interact
directly.

They have indirect interactions, however, because NumPy's
:class:`ndarray` type implements its binary operations via Ufuncs.  For
most numerical classes, the easiest way to override binary operations is
thus to define ``__array_ufunc__`` and override the corresponding
Ufunc. The class can then, like :class:`ndarray` itself, define the
binary operators in terms of Ufuncs. Here, one has to take some care.
E.g., the simplest implementation would be::

    class ArrayLike(object):
        ...
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            ...
            return result

        def __mul__(self, other):
            return self.__array_ufunc__(np.multiply, '__call__', self, other)

Suppose now, however, that ``other`` is class that does not know how to
deal with arrays and ufuncs, but does know how to do multiplication::

    class MyObject(object):
        __array_ufunc__ = None
        def __init__(self, value):
            self.value = value
        def __repr__(self):
            return "MyObject({!r})".format(self.value)
        def __mul__(self, other):
            return MyObject(1234)
        def __rmul__(self, other):
            return MyObject(4321)

In this case, standard Python override rules combined with the above
discussion would imply::

    mine = MyObject(0)
    arr = ArrayLike([0])

    mine * arr    # == MyObject(1234)   OK
    arr * mine    # TypeError     surprising

XXX: but it doesn't raise a TypeError, because `__mul__` calls
directly `__array_ufunc__`, which sees the `__array_ufunc__ == None`, and
bails out with `NotImplemented`?

The reason why this would occur is: because ``MyObject`` is not an
``ArrayLike`` subclass, Python resolves the expression ``arr * mine`` by
calling first ``arr.__mul__``. In the above implementation, this would
just call the Ufunc, which would see that ``mine.__array_ufunc__`` is
:obj:`None` and raise a :exc:`TypeError`. (Note that if ``MyObject``
is a subclass of :class:`ndarray`, Python calls ``mine.__rmul__`` first.)

So, a better implementation of the binary operators would check whether
the other class can be dealt with in ``__array_ufunc__`` and, if not,
return :obj:`NotImplemented`::

    class ArrayLike(object):
        ...
        def __mul__(self, other):
            if getattr(other, '__array_ufunc__', False) is None:
                return NotImplemented
            return self.__array_ufunc__(np.multiply, '__call__', self, other)

    arr = ArrayLike([0])

    arr * mine    # == 4321    OK

Indeed, after long discussion about whether it might make more sense to
ask classes like ``ArrayLike`` to implement a full ``__array_ufunc__``
[6]_, the same design as the above was agreed on for :class:`ndarray`
itself.

.. note:: The above holds for regular operators.  For in-place
          operators, :class:`ndarray` never returns
          :obj:`NotImplemented`, i.e., ``ndarr *= mine`` would always
          lead to a :exc:`TypeError`.  This is because for arrays
          in-place operations cannot generically be replaced by a simple
          reverse operation.  For instance, sticking to the above
          example, what would ``ndarr[:] *= mine`` imply? Assuming it
          means ``ndarr[:] = ndarr[:] * mine``, as python does by
          default, is likely to be wrong.

Extension to other numpy functions
----------------------------------

The ``__array_ufunc__`` method is used to override :func:`~numpy.dot`
and :func:`~numpy.matmul` as well, since while these functions are not
(yet) implemented as (generalized) Ufuncs, they are very similar.  For
other functions, such as :func:`~numpy.median`, :func:`~numpy.min`,
etc., implementations as (generalized) Ufuncs may well be possible and
logical as well, in which case it will become possible to override these
as well.

Demo
====

A pull request [8]_ has been made including the changes and revisions
proposed in this NEP.  Here is a demo highlighting the functionality.::

    In [1]: import numpy as np;

    In [2]: a = np.array([1])

    In [3]: class B():
       ...:     def __array_ufunc__(self, func, method, pos, inputs, **kwargs):
       ...:         return "B"
       ...:     

    In [4]: b = B()

    In [5]: np.dot(a, b)
    Out[5]: 'B'

    In [6]: np.multiply(a, b)
    Out[6]: 'B'

As a simple example, one could add the following ``__array_ufunc__`` to
SciPy's sparse matrices (just for ``np.dot`` and ``np.multiply`` as
these are the two most common cases where users would attempt to use
sparse matrices with ufuncs)::

    def __array_ufunc__(self, func, method, pos, inputs, **kwargs):
        """Method for compatibility with NumPy's ufuncs and dot
        functions.
        """

        without_self = list(inputs)
        without_self.pop(self)
        without_self = tuple(without_self)

        if func is np.multiply:
            return self.multiply(*without_self)

        elif func is np.dot:
            if pos == 0:
                return self.__mul__(inputs[1])
            if pos == 1:
                return self.__rmul__(inputs[0])
        else:
            return NotImplemented

So we now get the expected behavior when using ufuncs with sparse matrices.::

        In [1]: import numpy as np; import scipy.sparse as sp

        In [2]: a = np.random.randint(3, size=(3,3))

        In [3]: b = np.random.randint(3, size=(3,3))

        In [4]: asp = sp.csr_matrix(a); bsp = sp.csr_matrix(b)

        In [5]: np.dot(a,b)
        Out[5]: 
        array([[2, 4, 8],
               [2, 4, 8],
                [2, 2, 3]])

        In [6]: np.dot(asp,b)
        Out[6]: 
        array([[2, 4, 8],
               [2, 4, 8],
               [2, 2, 3]], dtype=int64)

        In [7]: np.dot(asp, bsp).A
        Out[7]: 
        array([[2, 4, 8],
               [2, 4, 8],
               [2, 2, 3]], dtype=int64)
                            
.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:

