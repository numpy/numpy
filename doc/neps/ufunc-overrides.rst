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

- If one of the input arguments implements ``__array_ufunc__`` it is
  executed instead of the Ufunc.

- If more than one of the input arguments implements ``__array_ufunc__``,
  they are tried in the following order: subclasses before superclasses,
  otherwise left to right.  The first ``__array_ufunc__`` method returning
  something else than :obj:`NotImplemented` determines the return value of
  the Ufunc.

- If all ``__array_ufunc__`` methods of the input arguments return
  :obj:`NotImplemented`, a :exc:`TypeError` is raised.

- If a ``__array_ufunc__`` method raises an error, the error is propagated
  immediately.

If none of the input arguments has an ``__array_ufunc__`` method, the
execution falls back on the default ufunc behaviour.

Subclass hierarchies
--------------------

Hierarchies of such containers (say, a masked quantity), are most easily
constructed if methods consistently use :func:`super` to pass through
the class hierarchy [7]_.  To support this, :class:`ndarray` has its own
``__array_ufunc__`` method (which is equivalent to ``getattr(ufunc,
method)(*inputs, **kwargs)``, i.e., if any of the (adjusted) inputs
still defines ``__array_ufunc__`` that will be called in turn). This
should be particularly useful for container-like subclasses of
:class:`ndarray`, which add an attribute like a unit or mask to a
regular :class:`ndarray`. Such classes can do possible adjustment of the
arguments relevant to their own class, pass on to another class in the
hierarchy using :func:`super` until the Ufunc is actually done, and then
do possible adjustments of the outputs.

Turning Ufuncs off
------------------

For some classes, Ufuncs make no sense, and, like for other special
methods [8]_, one can indicate Ufuncs are not available by setting
``__array_ufunc__`` to :obj:`None`.  Inside a Ufunc, this is
equivalent to unconditionally return :obj:`NotImplemented`, and thus
will lead to a :exc:`TypeError` (unless another operand implements
``__array_ufunc__`` and knows how to deal with the class).

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
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            ...
            return result
        ...
        def __mul__(self, other):
            return self.__array_ufunc__(np.multiply, '__call__', self, other)

Suppose now, however, that ``other`` is class that does not know how to
deal with arrays and ufuncs, but does know how to do multiplication::

    class MyObject(object):
        __array_ufunc__ = None
        def __mul__(self, other):
            return 1234
        def __rmul__(self, other):
            return 4321

In this case, standard Python override rules combined with the above
discussion would imply::

    mine = MyObject()
    arr = ArrayLike([0])

    mine * arr    # == 1234       OK
    arr * mine    # TypeError     surprising

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

