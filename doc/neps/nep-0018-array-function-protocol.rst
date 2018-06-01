==================================================
NEP: Dispatch Mechanism for NumPy's high level API
==================================================

:Author: Stephan Hoyer <shoyer@google.com>
:Author: Matthew Rocklin <mrocklin@gmail.com>
:Status: Draft
:Type: Standards Track
:Created: 2018-05-29

Abstact
-------

We propose a protocol to allow arguments of numpy functions to define
how that function operates on them. This allows other libraries that
implement NumPy's high level API to reuse Numpy functions. This allows
libraries that extend NumPy's high level API to apply to more NumPy-like
libraries.

Detailed description
--------------------

Numpy's high level ndarray API has been implemented several times
outside of NumPy itself for different architectures, such as for GPU
arrays (CuPy), Sparse arrays (scipy.sparse, pydata/sparse) and parallel
arrays (Dask array) as well as various Numpy-like implementations in the
deep learning frameworks, like TensorFlow and PyTorch.

Similarly there are several projects that build on top of the Numpy API
for labeled and indexed arrays (XArray), automatic differentation
(Autograd, Tangent), higher order array factorizations (TensorLy), etc.
that add additional functionality on top of the Numpy API.

We would like to be able to use these libraries together, for example we
would like to be able to place a CuPy array within XArray, or perform
automatic differentiation on Dask array code. This would be easier to
accomplish if code written for NumPy ndarrays could also be used by
other NumPy-like projects.

For example, we would like for the following code example to work
equally well with any Numpy-like array object:

.. code:: python

    def f(x):
        y = np.tensordot(x, x.T)
        return np.mean(np.exp(y))

Some of this is possible today with various protocol mechanisms within
Numpy.

-  The ``np.exp`` function checks the ``__array_ufunc__`` protocol
-  The ``.T`` method works using Python's method dispatch
-  The ``np.mean`` function explicitly checks for a ``.mean`` method on
   the argument

However other functions, like ``np.tensordot`` do not dispatch, and
instead are likely to coerce to a Numpy array (using the ``__array__``)
protocol, or err outright. To achieve enough coverage of the NumPy API
to support downstream projects like XArray and autograd we want to
support *almost all* functions within Numpy, which calls for a more
reaching protocol than just ``__array_ufunc__``. We would like a
protocol that allows arguments of a NumPy function to take control and
divert execution to another function (for example a GPU or parallel
implementation) in a way that is safe and consistent across projects.

Implementation
--------------

We propose adding support for a new protocol in NumPy,
``__array_function__``.

This protocol is intended to be a catch-all for NumPy functionality that
is not covered by existing protocols, like reductions (like ``np.sum``)
or universal functions (like ``np.exp``). The semantics are very similar
to ``__array_ufunc__``, except the operation is specified by an
arbitrary callable object rather than a ufunc instance and method.

The interface
~~~~~~~~~~~~~

We propose the following signature for implementations of
``__array_function__``:

.. code-block:: python

    def __array_function__(self, func, types, args, kwargs)

-  ``func`` is an arbitrary callable exposed by NumPy's public API,
   which was called in the form ``func(*args, **kwargs)``.
-  ``types`` is a list of types for all arguments to the original NumPy
   function call that will be checked for an ``__array_function__``
   implementation.
-  The tuple ``args`` and dict ``**kwargs`` are directly passed on from the
   original call.

Unlike ``__array_ufunc__``, there are no high-level guarantees about the
type of ``func``, or about which of ``args`` and ``kwargs`` may contain objects
implementing the array API. As a convenience for ``__array_function__``
implementors of the NumPy API, the ``types`` keyword contains a list of all
types that implement the ``__array_function__`` protocol.  This allows
downstream implementations to quickly determine if they are likely able to
support the operation.

Still be determined: what guarantees can we offer for ``types``? Should
we promise that types are unique, and appear in the order in which they
are checked?

Example for a project implementing the NumPy API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most implementations of ``__array_function__`` will start with two
checks:

1.  Is the given function something that we know how to overload?
2.  Are all arguments of a type that we know how to handle?

If these conditions hold, ``__array_function__`` should return
the result from calling its implementation for ``func(*args, **kwargs)``.
Otherwise, it should return the sentinel value ``NotImplemented``, indicating
that the function is not implemented by these types.

.. code:: python

    class MyArray:
        def __array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(issubclass(t, MyArray) for t in types):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    HANDLED_FUNCTIONS = {
        np.concatenate: my_concatenate,
        np.broadcast_to: my_broadcast_to,
        np.sum: my_sum,
        ...
    }

Necessary changes within the Numpy codebase itself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will require two changes within the Numpy codebase:

1. A function to inspect available inputs, look for the
   ``__array_function__`` attribute on those inputs, and call those
   methods appropriately until one succeeds.  This needs to be fast in the
   common all-NumPy case.

   This is one additional function of moderate complexity.
2. Calling this function within all relevant Numpy functions.

   This affects many parts of the Numpy codebase, although with very low
   complexity.

Finding and calling the right ``__array_function__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a Numpy function, ``*args`` and ``**kwargs`` inputs, we need to
search through ``*args`` and ``**kwargs`` for all appropriate inputs
that might have the ``__array_function__`` attribute. Then we need to
select among those possible methods and execute the right one.
Negotiating between several possible implementations can be complex.

Finding arguments
'''''''''''''''''

Valid arguments may be directly in the ``*args`` and ``**kwargs``, such
as in the case for ``np.tensordot(left, right, out=out)``, or they may
be nested within lists or dictionaries, such as in the case of
``np.concatenate([x, y, z])``. This can be problematic for two reasons:

1. Some functions are given long lists of values, and traversing them
   might be prohibitively expensive
2. Some function may have arguments that we don't want to inspect, even
   if they have the ``__array_function__`` method

To resolve these we ask the functions to provide an explicit list of
arguments that should be traversed. This is the ``relevant_arguments=``
keyword in the examples below.

Trying ``__array_function__`` methods until the right one works
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Many arguments may implement the ``__array_function__`` protocol. Some
of these may decide that, given the available inputs, they are unable to
determine the correct result. How do we call the right one? If several
are valid then which has precedence?

The rules for dispatch with ``__array_function__`` match those for
``__array_ufunc__`` (see
`NEP-13 <http://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_).
In particular:

-  NumPy will gather implementations of ``__array_function__`` from all
   specified inputs and call them in order: subclasses before
   superclasses, and otherwise left to right. Note that in some edge cases,
   this differs slightly from the
   `current behavior <https://bugs.python.org/issue30140>`_ of Python.
-  Implementations of ``__array_function__`` indicate that they can
   handle the operation by returning any value other than
   ``NotImplemented``.
-  If all ``__array_function__`` methods return ``NotImplemented``,
   NumPy will raise ``TypeError``.

Changes within Numpy functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a function defined above, for now call it
``do_array_function_dance``, we now need to call that function from
within every relevant Numpy function. This is a pervasive change, but of
fairly simple and innocuous code that should complete quickly and
without effect if no arguments implement the ``__array_function__``
protocol. Let us consider a few examples of NumPy functions and how they
might be affected by this change:

.. code:: python

    def broadcast_to(array, shape, subok=False):
        success, value = do_array_function_dance(
            func=broadcast_to,
            relevant_arguments=[array],
            args=(array,),
            kwargs=dict(shape=shape, subok=subok))
        if success:
            return value

        ... # continue with the definition of broadcast_to

    def concatenate(arrays, axis=0, out=None)
        success, value = do_array_function_dance(
            func=concatenate,
            relevant_arguments=[arrays, out],
            args=(arrays,),
            kwargs=dict(axis=axis, out=out))
        if success:
            return value

        ... # continue with the definition of concatenate

The list of objects passed to ``relevant_arguments`` are those that should
be inspected for ``__array_function__`` implementations.

Alternatively, we could write these overloads with a decorator, e.g.,

.. code:: python

    @overload_for_array_function(['array'])
    def broadcast_to(array, shape, subok=False):
        ... # continue with the definition of broadcast_to

    @overload_for_array_function(['arrays', 'out'])
    def concatenate(arrays, axis=0, out=None):
        ... # continue with the definition of concatenate

The decorator ``overload_for_array_function`` would be written in terms
of ``do_array_function_dance``.

The downside of this approach would be a loss of introspection capability
for NumPy functions on Python 2, since this requires the use of
``inspect.Signature`` (only available on Python 3). However, NumPy won't
be supporting Python 2 for `very much longer <http://www.numpy.org/neps/nep-0014-dropping-python2.7-proposal.html>`_.

Use outside of NumPy
~~~~~~~~~~~~~~~~~~~~

Nothing about this protocol that is particular to NumPy itself. Should
we enourage use of the same ``__array_function__`` protocol third-party
libraries for overloading non-NumPy functions, e.g., for making
array-implementation generic functionality in SciPy?

This would offer significant advantages (SciPy wouldn't need to invent
its own dispatch system) and no downsides that we can think of, because
every function that dispatches with ``__array_function__`` already needs
to be explicitly recognized. Libraries like Dask, CuPy, and Autograd
already wrap a limited subset of SciPy functionality (e.g.,
``scipy.linalg``) similarly to how they wrap NumPy.

If we want to do this, we should consider exposing the helper function
``do_array_function_dance()`` above as a public API.

Non-goals
---------

We are aiming for basic strategy that can be relatively mechanistically
applied to almost all functions in NumPy's API in a relatively short
period of time, the development cycle of a single NumPy release.

We hope to get both the ``__array_function__`` protocol and all specific
overloads right on the first try, but our explicit aim here is to get
something that mostly works (and can be iterated upon), rather than to
wait for an optimal implementation. The price of moving fast is that for
now **this protocol should be considered strictly experimental**. We
reserve the right to change the details of this protocol and how
specific NumPy functions use it at any time in the future -- even in
otherwise bug-fix only releases of NumPy.

In particular, we don't plan to write additional NEPs that list all
specific functions to overload, with exactly how they should be
overloaded. We will leave this up to the discretion of committers on
individual pull requests, trusting that they will surface any
controversies for discussion by interested parties.

However, we already know several families of functions that should be
explicitly exclude from ``__array_function__``. These will need their
own protocols:

-  universal functions, which already have their own protocol.
-  ``array`` and ``asarray``, because they are explicitly intended for
   coercion to actual ``numpy.ndarray`` object.
-  dispatch for methods of any kind, e.g., methods on
   ``np.random.RandomState`` objects.

As a concrete example of how we expect to break behavior in the future,
some functions such as ``np.where`` are currently not NumPy universal
functions, but conceivably could become universal functions in the
future. When/if this happens, we will change such overloads from using
``__array_function__`` to the more specialized ``__array_ufunc__``.


Backward compatibility
----------------------

This proposal does not change existing semantics, except for those arguments
that currently have ``__array_function__`` methods, which should be rare.


Alternatives
------------

Specialized protocols
~~~~~~~~~~~~~~~~~~~~~

We could (and should) continue to develop protocols like
``__array_ufunc__`` for cohesive subsets of Numpy functionality.

As mentioned above, if this means that some functions that we overload
with ``__array_function__`` should switch to a new protocol instead,
that is explicitly OK for as long as ``__array_function__`` retains its
experimental status.

Separate namespace
~~~~~~~~~~~~~~~~~~

A separate namespace for overloaded functions is another possibility,
either inside or outside of NumPy.

This has the advantage of alleviating any possible concerns about
backwards compatibility and would provide the maximum freedom for quick
experimentation. In the long term, it would provide a clean abstration
layer, separating NumPy's high level API from default implementations on
``numpy.ndarray`` objects.

The downsides are that this would require an explicit opt-in from all
existing code, e.g., ``import numpy.api as np``, and in the long term
would result in the maintainence of two separate NumPy APIs. Also, many
functions from ``numpy`` itself are already overloaded (but
inadequately), so confusion about high vs. low level APIs in NumPy would
still persist.

Multiple dispatch
~~~~~~~~~~~~~~~~~

An alternative to our suggestion of the ``__array_function__`` protocol
would be implementing NumPy's core functions as
`multi-methods <https://en.wikipedia.org/wiki/Multiple_dispatch>`_.
Although one of us wrote a `multiple dispatch
library <https://github.com/mrocklin/multipledispatch>`_ for Python, we
don't think this approach makes sense for NumPy in the near term.

The main reason is that NumPy already has a well-proven dispatching
mechanism with ``__array_ufunc__``, based on Python's own dispatching
system for arithemtic, and it would be confusing to add another
mechanism that works in a very different way. This would also be more
invasive change to NumPy itself, which would need to gain a multiple
dispatch implementation.

It is possible that multiple dispatch implementation for NumPy's high
level API could make sense in the future. Fortunately,
``__array_function__`` does not preclude this possibility, because it
would be straightforward to write a shim for a default
``__array_function__`` implementation in terms of multiple dispatch.

Implementations in terms of a limited core API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The internal implemenations of some NumPy functions is extremely simple.
For example: - ``np.stack()`` is implemented in only a few lines of code
by combining indexing with ``np.newaxis``, ``np.concatenate`` and the
``shape`` attribute. - ``np.mean()`` is implemented internally in terms
of ``np.sum()``, ``np.divide()``, ``.astype()`` and ``.shape``.

This suggests the possibility of defining a minimal "core" ndarray
interface, and relying upon it internally in NumPy to implement the full
API. This is an attractive option, because it could significantly reduce
the work required for new array implementations.

However, this also comes with several downsides: 1. The details of how
NumPy implements a high-level function in terms of overloaded functions
now becomes an implicit part of NumPy's public API. For example,
refactoring ``stack`` to use ``np.block()`` instead of
``np.concatenate()`` internally would now become a breaking change. 2.
Array libraries may prefer to implement high level functions differently
than NumPy. For example, a library might prefer to implement a
fundamental operations like ``mean()`` directly rather than relying on
``sum()`` followed by division. More generally, it's not clear yet what
exactly qualifies as core functionality, and figuring this out could be
a large project. 3. We don't yet have an overloading system for
attributes and methods on array objects, e.g., for accessing ``.dtype``
and ``.shape``. This should be the subject of a future NEP, but until
then we should be reluctant to rely on these properties.

Given these concerns, we encourage relying on this approach only in
limited cases.

Coersion to a NumPy array as a catch-all fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the current design, classes that implement ``__array_function__``
to overload at least one function implicitly declare an intent to
implement the entire NumPy API. It's not possible to implement *only*
``np.concatenate()`` on a type, but fall back to NumPy's default
behavior of casting with ``np.asarray()`` for all other functions.

This could present a backwards compatibility concern that would
discourage libraries from adopting ``__array_function__`` in an
incremental fashion. For example, currently most numpy functions will
implicitly convert ``pandas.Series`` objects into NumPy arrays, behavior
that assuredly many pandas users rely on. If pandas implemented
``__array_function__`` only for ``np.concatenate``, unrelated NumPy
functions like ``np.nanmean`` would suddenly break on pandas objects by
raising TypeError.

With ``__array_ufunc__``, it's possible to alleviate this concern by
casting all arguments to numpy arrays and re-calling the ufunc, but the
heterogeneous function signatures supported by ``__array_function__``
make it impossible to implement this generic fallback behavior for
``__array_function__``.

We could resolve this issue by change the handling of return values in
``__array_function__`` in either of two possible ways: 1. Change the
meaning of all arguments returning ``NotImplemented`` to indicate that
all arguments should be coerced to NumPy arrays instead. However, many
array libraries (e.g., scipy.sparse) really don't want implicit
conversions to NumPy arrays, and often avoid implementing ``__array__``
for exactly this reason. Implicit conversions can result in silent bugs
and performance degradation. 2. Use another sentinel value of some sort
to indicate that a class implementing part of the higher level array API
is coercible as a fallback, e.g., a return value of
``np.NotImplementedButCoercible`` from ``__array_function__``.

If we take this second approach, we would need to define additional
rules for how coercible array arguments are coerced, e.g., - Would we
try for ``__array_function__`` overloads again after coercing coercible
arguments? - If so, would we coerce coercible arguments one-at-a-time,
or all-at-once?

These are slightly tricky design questions, so for now we propose to
defer this issue. We can always implement
``np.NotImplementedButCoercible`` at some later time if it proves
critical to the numpy community in the future. Importantly, we don't
think this will stop critical libraries that desire to implement most of
the high level NumPy API from adopting this proposal.

NOTE: If you are reading this NEP in its draft state and disagree,
please speak up on the mailing list!

Drawbacks of this approach
--------------------------

Future difficulty extending NumPy's API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One downside of passing on all arguments directly on to
``__array_function__`` is that it makes it hard to extend the signatures
of overloaded NumPy functions with new arguments, because adding even an
optional keyword argument would break existing overloads.

This is not a new problem for NumPy. NumPy has occasionally changed the
signature for functions in the past, including functions like
``numpy.sum`` which support overloads.

For adding new keyword arguments that do not change default behavior, we
would only include these as keyword arguments when they have changed
from default values. This is similar to `what NumPy already has
done <https://github.com/numpy/numpy/blob/v1.14.2/numpy/core/fromnumeric.py#L1865-L1867>`_,
e.g., for the optional ``keepdims`` argument in ``sum``:

.. code:: python

    def sum(array, ..., keepdims=np._NoValue):
        kwargs = {}
        if keepdims is not np._NoValue:
            kwargs['keepdims'] = keepdims
        return array.sum(..., **kwargs)

In other cases, such as deprecated arguments, preserving the existing
behavior of overloaded functions may not be possible. Libraries that use
``__array_function__`` should be aware of this risk: we don't propose to
freeze NumPy's API in stone any more than it already is.

Difficulty adding implementation specific arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some array implementations generally follow NumPy's API, but have
additional optional keyword arguments (e.g., ``dask.array.sum()`` has
``split_every`` and ``tensorflow.reduce_sum()`` has ``name``). A generic
dispatching library could potentially pass on all unrecognized keyword
argument directly to the implementation, but extending ``np.sum()`` to
pass on ``**kwargs`` would entail public facing changes in NumPy.
Customizing the detailed behavior of array libraries will require using
library specific functions, which could be limiting in the case of
libraries that consume the NumPy API such as xarray.


Discussion
----------

Various alternatives to this proposal were discussed in a few Github issues:

1.  `pydata/sparse #1 <https://github.com/pydata/sparse/issues/1>`_
2.  `numpy/numpy #11129 <https://github.com/numpy/numpy/issues/11129>`_

Additionally it was the subject of `a blogpost
<http://matthewrocklin.com/blog/work/2018/05/27/beyond-numpy>`_ Following this
it was discussed at a `NumPy developer sprint
<https://scisprints.github.io/#may-numpy-developer-sprint>`_ at the `UC
Berkeley Institute for Data Science (BIDS) <https://bids.berkeley.edu/>`_.


References and Footnotes
------------------------

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: http://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
