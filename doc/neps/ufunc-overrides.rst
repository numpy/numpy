=================================
A Mechanism for Overriding Ufuncs
=================================

:Author: Blake Griffith
:Contact: blake.g@utexas.edu 
:Date: 2013-07-10

:Author: Pauli Virtanen

:Author: Nathaniel Smith


Executive summary
=================

NumPy's universal functions (ufuncs) currently have some limited
functionality for operating on user defined subclasses of ndarray using
``__array_prepare__`` and ``__array_wrap__`` [1]_, and there is little
to no support for arbitrary objects. e.g. SciPy's sparse matrices [2]_
[3]_.

Here we propose adding a mechanism to override ufuncs based on the ufunc
checking each of it's arguments for a ``__numpy_ufunc__`` method.
On discovery of ``__numpy_ufunc__`` the ufunc will hand off the
operation to the method. 

This covers some of the same ground as Travis Oliphant's proposal to
retro-fit NumPy with multi-methods [4]_, which would solve the same
problem. The mechanism here follows more closely the way Python enables
classes to override ``__mul__`` and other binary operations.

.. [1] http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
.. [2] https://github.com/scipy/scipy/issues/2123
.. [3] https://github.com/scipy/scipy/issues/1569
.. [4] http://technicaldiscovery.blogspot.com/2013/07/thoughts-after-scipy-2013-and-specific.html


Motivation
==========

The current machinery for dispatching Ufuncs is generally agreed to be
insufficient. There have been lengthy discussions and other proposed
solutions [5]_.

Using ufuncs with subclasses of ndarray is limited to ``__array_prepare__`` and
``__array_wrap__`` to prepare the arguments, but these don't allow you to for
example change the shape or the data of the arguments. Trying to ufunc things
that don't subclass ndarray is even more difficult, as the input arguments tend
to be cast to object arrays, which ends up producing surprising results.

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

Returning ``NotImplemented`` to user should not happen. Moreover::

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

Here, it appears that the sparse matrix was converted to a object array
scalar, which was then multiplied with all elements of the ``b`` array.
However, this behavior is more confusing than useful, and having a
``TypeError`` would be preferable.

Adding the ``__numpy_ufunc__`` functionality fixes this and would
deprecate the other ufunc modifying functions.

.. [5] http://mail.scipy.org/pipermail/numpy-discussion/2011-June/056945.html


Proposed interface
==================

Objects that want to override Ufuncs can define a ``__numpy_ufunc__`` method.
The method signature is::

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs)

Here:

- *ufunc* is the ufunc object that was called. 
- *method* is a string indicating which Ufunc method was called
  (one of ``"__call__"``, ``"reduce"``, ``"reduceat"``,
  ``"accumulate"``, ``"outer"``, ``"inner"``). 
- *i* is the index of *self* in *inputs*.
- *inputs* is a tuple of the input arguments to the ``ufunc``
- *kwargs* are the keyword arguments passed to the function. The ``out``
  arguments are always contained in *kwargs*, how positional variables
  are passed is discussed below.

The ufunc's arguments are first normalized into a tuple of input data
(``inputs``), and dict of keyword arguments. If there are output
arguments they are handeled as follows:

- One positional output variable x is passed in the kwargs dict as ``out :
  x``.
- Multiple positional output variables ``x0, x1, ...`` are passed as a tuple
  in the kwargs dict as ``out : (x0, x1, ...)``.
- Keyword output variables like ``out = x`` and ``out = (x0, x1, ...)`` are
  passed unchanged to the kwargs dict like ``out : x`` and ``out : (x0, x1,
  ...)`` respectively.
- Combinations of positional and keyword output variables are not
  supported.

The function dispatch proceeds as follows:

- If one of the input arguments implements ``__numpy_ufunc__`` it is
  executed instead of the Ufunc.

- If more than one of the input arguments implements ``__numpy_ufunc__``,
  they are tried in the following order: subclasses before superclasses,
  otherwise left to right.  The first ``__numpy_ufunc__`` method returning
  something else than ``NotImplemented`` determines the return value of
  the Ufunc.

- If all ``__numpy_ufunc__`` methods of the input arguments return
  ``NotImplemented``, a ``TypeError`` is raised.

- If a ``__numpy_ufunc__`` method raises an error, the error is propagated
  immediately.

If none of the input arguments has a ``__numpy_ufunc__`` method, the
execution falls back on the default ufunc behaviour.


In combination with Python's binary operations
----------------------------------------------

The ``__numpy_ufunc__`` mechanism is fully independent of Python's
standard operator override mechanism, and the two do not interact
directly.

They however have indirect interactions, because NumPy's ``ndarray``
type implements its binary operations via Ufuncs. Effectively, we have::

    class ndarray(object):
        ...
        def __mul__(self, other):
            return np.multiply(self, other)

Suppose now we have a second class::

    class MyObject(object):
        def __numpy_ufunc__(self, *a, **kw):
            return "ufunc"
        def __mul__(self, other):
            return 1234
        def __rmul__(self, other):
            return 4321

In this case, standard Python override rules combined with the above
discussion imply::

    a = MyObject()
    b = np.array([0])

    a * b    # == 1234       OK
    b * a    # == "ufunc"    surprising

This is not what would be naively expected, and is therefore somewhat
undesirable behavior.

The reason why this occurs is: because ``MyObject`` is not an ndarray
subclass, Python resolves the expression ``b * a`` by calling first
``b.__mul__``. Since NumPy implements this via an Ufunc, the call is
forwarded to ``__numpy_ufunc__`` and not to ``__rmul__``.  Note that if
``MyObject`` is a subclass of ``ndarray``, Python calls ``a.__rmul__``
first. The issue is therefore that ``__numpy_ufunc__`` implements
"virtual subclassing" of ndarray behavior, without actual subclassing.

This issue can be resolved by a modification of the binary operation
methods in NumPy::

    class ndarray(object):
        ...
        def __mul__(self, other):
            if (not isinstance(other, self.__class__) 
                    and hasattr(other, '__numpy_ufunc__') 
                    and hasattr(other, '__rmul__')):
                return NotImplemented
            return np.multiply(self, other)

        def __imul__(self, other):
            if (other.__class__ is not self.__class__
                    and hasattr(other, '__numpy_ufunc__') 
                    and hasattr(other, '__rmul__')):
                return NotImplemented
            return np.multiply(self, other, out=self)

    b * a    # == 4321    OK

The rationale here is the following: since the user class explicitly
defines both ``__numpy_ufunc__`` and ``__rmul__``, the implementor has
very likely made sure that the ``__rmul__`` method can process ndarrays.
If not, the special case is simple to deal with (just call
``np.multiply``).

The exclusion of subclasses of self can be made because Python itself
calls the right-hand method first in this case. Moreover, it is
desirable that ndarray subclasses are able to inherit the right-hand
binary operation methods from ndarray.

The same priority shuffling needs to be done also for the in-place
operations, so that ``MyObject.__rmul__`` is prioritized over
``ndarray.__imul__``.


Demo
====

A pull request[6]_ has been made including the changes proposed in this NEP.
Here is a demo highlighting the functionality.::

    In [1]: import numpy as np;

    In [2]: a = np.array([1])

    In [3]: class B():
       ...:     def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
       ...:         return "B"
       ...:     

    In [4]: b = B()

    In [5]: np.dot(a, b)
    Out[5]: 'B'

    In [6]: np.multiply(a, b)
    Out[6]: 'B'

A simple ``__numpy_ufunc__`` has been added to SciPy's sparse matrices
Currently this only handles ``np.dot`` and ``np.multiply`` because it was the 
two most common cases where users would attempt to use sparse matrices with ufuncs.
The method is defined below::

    def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
        """Method for compatibility with NumPy's ufuncs and dot
        functions.
        """

        without_self = list(inputs)
        del without_self[pos]
        without_self = tuple(without_self)

        if func == np.multiply:
            return self.multiply(*without_self)

        elif func == np.dot:
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

