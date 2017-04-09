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
(Note that in earlier iterations, the override was called
``__numpy_ufunc__``. An implementation was made, but had not quite the
right behaviour, hence the change in name.)

The ``__array_ufunc__`` as described below requires that any
corresponding Python binary operations (``__mul__`` et al.) should be
implemented in a specific way and be compatible with Numpy's ndarray
semantics. Objects that do not satisfy this cannot override any Numpy
ufuncs.  We do not specify a future-compatible path by which this
requirement can be relaxed --- any changes here require corresponding
changes in 3rd party code.

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
``__array_prepare__`` and ``__array_wrap__`` to prepare the output arguments,
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

This proposal will *not* resolve the issue with scipy.sparse matrices,
which have multiplication semantics incompatible with numpy arrays.
However, the aim is to enable writing other custom array types that have
strictly ndarray compatible semantics.

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
- *method* is a string indicating how the Ufunc was called, either
  ``"__call__"`` to indicate it was called directly, or one of its
  :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
  ``"reduceat"``, ``"outer"``, or ``"at"``. 
- *inputs* is a tuple of the input arguments to the ``ufunc``
- *kwargs* contains any optional or keyword arguments passed to the
  function. This includes any ``out`` arguments, which are always
  contained in a tuple.

Hence, the arguments are normalized: only the required input arguments
(``inputs``) are passed on as positional arguments, all the others are
passed on as a dict of keyword arguments (``kwargs``). In particular, if
there are output arguments, positional are otherwise, that are not
:obj:`None`, they are passed on as a tuple in the ``out`` keyword
argument.

The function dispatch proceeds as follows:

- If one of the input or output arguments implements
  ``__array_ufunc__``, it is executed instead of the ufunc.

- If more than one of the input arguments implements ``__array_ufunc__``,
  they are tried in the following order: subclasses before superclasses,
  inputs before outputs, otherwise left to right.

- The first ``__array_ufunc__`` method returning something else than
  :obj:`NotImplemented` determines the return value of the Ufunc.

- If all ``__array_ufunc__`` methods of the input arguments return
  :obj:`NotImplemented`, a :exc:`TypeError` is raised.

- If a ``__array_ufunc__`` method raises an error, the error is
  propagated immediately.

- If none of the input arguments had an ``__array_ufunc__`` method, the
  execution falls back on the default ufunc behaviour.

In the above, there is one proviso: if a class has an
``__array_ufunc__`` attribute but it is identical to
``ndarray.__array_ufunc__``, the attribute is ignored.  This happens for
instances of `ndarray` and for `ndarray` subclasses that did not
override their inherited ``__array_ufunc__`` implementation.


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

It is useful to think of the typecasting hierarchy as a graph (see below
for example graphs that work and that fail because of cyclic
dependencies) in which, for any given class A, all other classes that
define ``__array_ufunc__`` must belong to exactly one of three groups
(making this an directed acyclic graph):

- *Above A*: their ``__array_ufunc__`` can handle class A or some
  member of the "above A" classes. In other words, these are the types
  that A can be (indirectly) upcast to in ufuncs.

- *Below A*: they can be handled by the ``__array_ufunc__`` of class A
  or the ``__array_ufunc__`` of some member of the "below A" classes. In
  other words, these are the types that can be (indirectly) upcast to A
  in ufuncs.

- *Incompatible*: neither above nor below A; types for which no
  (indirect) upcasting is possible.  Neither can handle the other.

Given this grouping, to ensure that expressions involving ufuncs either
raise a :exc:`TypeError`, or have a result type that is independent of
what ufuncs were called, what order they were called in, and what order
their arguments were in, the above implies that ``__array_ufunc__`` for
type A should:

- Return an object of type A if all other arguments are of types below A.

- Return :obj:`NotImplemented` if any argument has a type that is above
  A or with which it is incompatible.

With the above, one can convert relations between types to edges in a
`graph<https://en.wikipedia.org/wiki/Graph_theory>`_ by defining "can
handle" as follows: if for instances ``a`` and ``b`` of types A and B,
``a.__array_ufunc__(..., b, ...)`` returns a result other than
:obj:`NotImplemented` (and does not raise an error), then a can handle
b and B->A is an edge of the graph.
  
Note that there are, as always, exceptions.  For instance, for a
quantity class, the results of most ufuncs should be quantities, but
this is not the case for comparison operators. For those, a quantity
class would return a plain array.

Note also that the legacy behaviour of numpy ufunc is to try to convert
unknown objects to :class:`ndarray` via :func:`np.asarray`.  This is
equivalent to placing :class:`ndarray` at the very top of the graph, and
is thus a consistent type hierarchy (although one that causes the
problems that motivate this NEP...).  By instead letting
:class:`ndarray` return `NotImplemented` if any argument defines
``__array_ufunc__``, we provide the option for other classes to have
:class:`ndarray` at the bottom of the type hierarchy.

.. admonition:: Example

   Type casting hierarchy.

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> C;
         B -> C;
         D -> B;
         ndarray -> A;
         ndarray -> B;
      }

   The ``__array_ufunc__`` of type A can handle ndarrays, B can handle
   ndarray and D, and C can handle A and B but not ndarrays or D.  The
   result is a directed acyclic graph, and defines a type casting
   hierarchy, with relations ``C > A > ndarray``, ``C > B > ndarray``,
   ``C > B > D``. The type B is incompatible relative to A and vice
   versa, and A and ndarray are incompatible relative to D.  Ufunc
   expressions involving these classes should produce results of the
   highest type involved or raise a :exc:`TypeError`.

.. admonition:: Example

   One-cycle in the ``__array_ufunc__`` graph.

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> B;
         B -> A;
      }


   In this case, the ``__array_ufunc__`` relations have a cycle of length 1,
   and a type casting hierarchy does not exist. Binary operations are not
   commutative: ``type(a + b) is A`` but ``type(b + a) is B``.

.. admonition:: Example

   Longer cycle in the ``__array_ufunc__`` graph.

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> B;
         B -> C;
         C -> A;
      }


   In this case, the ``__array_ufunc__`` relations have a longer cycle, and a
   type casting hierarchy does not exist. Binary operations are still
   commutative, but type transitivity is lost: ``type(a + (b + c)) is A`` but
   ``type((a + b) + c) is C``.


Subclass hierarchies
--------------------

Generally, it is desirable to mirror the class hierarchy in the ufunc
type casting hierarchy. The recommendation is that an
``__array_ufunc__`` implementation of a class should generally return
`NotImplemented` unless the inputs are instances of the same class or
superclasses.  This guarantees that in the type casting hierarchy,
superclasses are below, subclasses above, and other classes are
incompatible (sadly, the terminology for graphs and classes has reversed
vertical sense).  Exceptions to this need to check they respect the
implicit type casting hierarchy.

Subclasses can be easily constructed if methods consistently use
:func:`super` to pass through the class hierarchy [7]_.  To support
this, :class:`ndarray` has its own ``__array_ufunc__`` method,
equivalent to::

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Cannot handle items that have __array_ufunc__ (other than our own).
        outputs = kwargs.get('out', ())
        for item in inputs + outputs):
            if (hasattr(item, '__array_ufunc__') and
                    type(item).__array_ufunc__ is not ndarray.__array_ufunc__):
                return NotImplemented

        # If we didn't have to support legacy behaviour (__array_prepare__,
        # __array_wrap__, etc.), we might here convert python floats,
        # lists, etc, to arrays with
        # items = [np.asarray(item) for item in inputs]
        # and then start the right iterator for the given method.
        # However, we do have to support legacy, so call back into the ufunc.
        # Its arguments are now guaranteed not to have __array_ufunc__
        # overrides, and it will do the coercion to array for us.
        return getattr(ufunc, method)(*items, **kwargs)

Note that, as a special case, the ufunc dispatch mechanism does not call
this `ndarray.__array_ufunc__` method, even for `ndarray` subclasses
if they have not overridden the default `ndarray` implementation. As a
consequence, calling `ndarray.__array_ufunc__` will not result to a
nested ufunc dispatch cycle.

The use of :func:`super` should be particularly useful for subclasses of
:class:`ndarray` that only add an attribute like a unit.  In their
`__array_ufunc__` implementation, such classes can do possible
adjustment of the arguments relevant to their own class, and pass on to
the superclass implementation using :func:`super` until the ufunc is
actually done, and then do possible adjustments of the outputs.

In general, custom implementations of `__array_ufunc__` should avoid
nested dispatch cycles.  However, for some subclasses, it may be better
to use ``getattr(ufunc, method)(*items, **kwargs)``. For instance, for a
class like :class:`MaskedArray`, which only cares that whatever
it contains is an :class:`ndarray` subclass, a reimplementation with
``__array_ufunc__`` may well be more easily done by directly applying
the ufunc to its data, and then adjusting the mask.  Indeed, one can
think of this as part of the class determining whether it can handle the
other argument (i.e., where in the type hierarchy it sits). In this
case, one should return :obj:`NotImplemented` if the trial fails.  So,
the implementation would be something like::

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # for simplicity, outputs are ignored here.
        unmasked_items = [item.data if isinstance(item, MaskedArray)
                          else item]
        try:
            unmasked_result = getattr(ufunc, method)(*unmasked_items, **kwargs)
        except TypeError:
            return NotImplemented
        # for simplicity, ignore that unmasked_result could be a tuple
        # or a scalar.
        if not isinstance(unmasked_result, np.ndarray):
            return NotImplemented
        # now combine masks and view as MaskedArray instance
        ...

As a specific example, consider a quantity and a masked array class
which both override ``__array_ufunc__``, with specific instances ``q``
and ``ma``, where the latter contains a regular array. Executing
``np.multiply(q, ma)``, the ufunc will first dispatch to
``q.__array_ufunc__``, which returns :obj:`NotImplemented` (since the
quantity class turns itself into an array and calls :func:`super`, which
passes on to ``ndarray.__array_ufunc__``, which sees the override on
``ma``). Next, ``ma.__array_ufunc__`` gets a chance. It does not know
quantity, and if it were to just return :obj:`NotImplemented` as well,
an :exc:`TypeError` would result. But in our sample implementation, it
uses ``getattr(ufunc, method)`` to, effectively, evaluate
``np.multiply(q, ma.data)``. This again will pass to
``q.__array_ufunc__``, but this time, since ``ma.data`` is a regular
array, it will return a result that is also a quantity. Since this is a
subclass of :class:`ndarray`, ``ma.__array_ufunc__`` can turn this into
a masked array and thus return a result (obviously, if it was not a
array subclass, it could still return :obj:`NotImplemented`).

Note that in the context of the type hierarchy discussed above this is a
somewhat tricky example, since :class:`MaskedArray` has a strange
position: it is above all subclasses of :class:`ndarray`, in that it can
cast them to its own type, but it does not itself know how to interact
with them in ufuncs.


Turning Ufuncs off
------------------

For some classes, Ufuncs make no sense, and, like for some other special
methods such as ``__hash__`` and ``__iter__`` [8]_, one can indicate
Ufuncs are not available by setting ``__array_ufunc__`` to :obj:`None`.
Inside a Ufunc, this is equivalent to unconditionally returning
:obj:`NotImplemented`, and thus will lead to a :exc:`TypeError` (unless
another operand implements ``__array_ufunc__`` and specifically knows
how to deal with the class).

In the type casting hierarchy, this makes it explicit that the type is
incompatible relative to :class:`ndarray`.

.. [7] https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

.. [8] https://docs.python.org/3/reference/datamodel.html#specialnames

In combination with Python's binary operations
----------------------------------------------

The Python operator override mechanism in :class:`ndarray` is coupled to
the ``__array_ufunc__`` mechanism. :class:`ndarray` returns
:obj:`NotImplemented` from ``ndarray.__mul__(self, other)`` and other
binary operation methods if ``other.__array_ufunc__ is None``. If the
``__array_ufunc__`` attribute is absent, :obj:`NotImplemented` is
returned if ``other.__array_priority__ > self.__array_priority__``.  In
other cases, :class:`ndarray` calls the corresponding ufunc. The
resulting behavior can modified by overriding the corresponding ufunc
via implementing ``__array_ufunc__``.

A class wishing to modify the interaction with :class:`ndarray` in
binary operations has two options:

1. Implement ``__array_ufunc__`` and follow Numpy semantics for Python
   binary operations (see below).

2. Set ``__array_ufunc__ = None``, and implement Python binary
   operations freely.  In this case, ufuncs will raise :exc:`TypeError`
   in combination with ndarray inputs.

For most numerical classes, the easiest way to override binary
operations is thus to define ``__array_ufunc__`` and override the
corresponding Ufunc. The class can then, like :class:`ndarray` itself,
define the binary operators in terms of Ufuncs. Here, one has to take
some care to ensure that one allows for other classes to indicate they
are not compatible, i.e., implementations should be something like::

    class ArrayLike(object):
        ...
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            ...
            return result

        # Option 1: call ufunc directly
        def __mul__(self, other):
            if getattr(other, '__array_ufunc__', False) is None:
                return NotImplemented
            return np.multiply(self, other)

        def __rmul__(self, other):
            if getattr(other, '__array_ufunc__', False) is None:
                return NotImplemented
            return np.multiply(other, self)

        def __imul__(self, other):
            return np.multiply(self, other, out=(self,))

        # Option 2: call into one's own __array_ufunc__
        def __mul__(self, other):
            return self.__array_ufunc__(np.multiply, '__call__', self, other)

        def __rmul__(self, other):
            return self.__array_ufunc__(np.multiply, '__call__', other, self)

        def __imul__(self, other):
            result = self.__array_ufunc__(np.multiply, '__call__', self, other,
                                          out=(self,))
            if result is NotImplemented:
                raise TypeError(...)

To see why some care is necessary, consider another class ``other`` that
does not know how to deal with arrays and ufuncs, and thus has set
``__array_ufunc__`` to :obj:`None`, but does know how to do
multiplication::

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

For either option above, we get the expected result::

    mine = MyObject(0)
    arr = ArrayLike([0])

    mine * arr    # -> MyObject(1234)
    mine *= arr   # -> MyObject(1234)
    arr * mine    # -> MyObject(4321)
    arr *= mine   # -> TypeError

Here, in the first and second example, ``mine.__mul__(arr)`` gets called
and the result arrives immediately.  In the third example, first
``arr.__mul__(mine)`` is called. In option (1), the check on
``mine.__array_ufunc__ is None`` will succeed and thus
:obj:`NotImplemented` is returned, which causes ``mine.__rmul__(arg)``
to be executed.  In option (2), it is presumably inside
``arr.__array_ufunc__`` that it becomes clear that the other argument
cannot be dealt with, and again :obj:`NotImplemented` is returned,
causing control to pass to ``mine.__rmul__``.

For the fourth example, with the in-place operators, we have here
followed :class:`ndarray` and ensure we never return
:obj:`NotImplemented`, but rather raise a :exc:`TypeError`. In
option (1) this happens indirectly: we pass to ``np.multiply``, which
calls ``arr.__array_ufunc__``. This, however, will not know what to do
with ``mine`` and will thus return :obj:`NotImplemented`.  Then, the
ufunc turns to ``mine.__array_ufunc__``. But this is :obj:`None`,
equivalent to returning :obj:`NotImplemented`, so a :exc:`TypeError` is
raised.  In option (2), we pass directly to ``arr.__array_ufunc__``,
which will return :obj:`NotImplemted`, which we catch.

.. note :: the reason for not allowing in-place operations to return
   :obj:`NotImplemented` is that these cannot generically be replaced by
   a simple reverse operation: most array operations assume the contents
   of the instance are changed in-place, and do not expect a new
   instance.  Also, what would ``ndarr[:] *= mine`` imply?  Assuming it
   means ``ndarr[:] = ndarr[:] * mine``, as python does by default if
   the ``ndarr.__imul__`` were to return :obj:`NotImplemented`, is
   likely to be wrong.

Now consider what would happen if we had not added checks. For option
(1), the relevant case is if we had not checked whether
``__array_func__`` was set to :obj:`None`.  In the third example,
``arr.__mul__(mine)`` is called, and without the check, this would go to
``np.multiply(arr, mine)``. This tries ``arr.__array_ufunc__``, which
returns :obj:`NotImplemented` and sees that ``mine.__array_ufunc__ is
None``, so a :exc:`TypeError` is raised.

For option (2), the relevant example is the fourth, with ``arr *=
mine``: if we had let the :obj:`NotImplemented` pass, python would have
replaced this with ``arr = mine.__rmul__(arr)``, which is not wanted.

Finally, we note that we had extensive discussion about whether it might
make more sense to ask classes like ``MyObject`` to implement a full
``__array_ufunc__`` [6]_. In the end, allowing classes to opt out was
preferred, and the above reasoning led us to agree on a similar
implementation for :class:`ndarray` itself. To help implement array-like
classes, we will provide a mixin class that provides overrides for all
binary operators.


Future extensions to other functions
------------------------------------

Some numpy functions could be implemented as (generalized) Ufunc, in
which case it would be possible for them to be overridden by the
``__array_ufunc__`` method.  A prime candidate is :func:`~numpy.matmul`,
which currently is not a Ufunc, but could be relatively easily be
rewritten as a (set of) generalized Ufuncs. The same may happen with
functions such as :func:`~numpy.median`, :func:`~numpy.min`, and
:func:`~numpy.argsort`.


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:

