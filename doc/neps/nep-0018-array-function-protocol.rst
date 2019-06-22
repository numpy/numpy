====================================================================
NEP 18 — A dispatch mechanism for NumPy's high level array functions
====================================================================

:Author: Stephan Hoyer <shoyer@google.com>
:Author: Matthew Rocklin <mrocklin@gmail.com>
:Author: Marten van Kerkwijk <mhvk@astro.utoronto.ca>
:Author: Hameer Abbasi <hameerabbasi@yahoo.com>
:Author: Eric Wieser <wieser.eric@gmail.com>
:Status: Provisional
:Type: Standards Track
:Created: 2018-05-29
:Updated: 2019-05-25
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-August/078493.html

Abstact
-------

We propose the ``__array_function__`` protocol, to allow arguments of NumPy
functions to define how that function operates on them. This will allow
using NumPy as a high level API for efficient multi-dimensional array
operations, even with array implementations that differ greatly from
``numpy.ndarray``.

Detailed description
--------------------

NumPy's high level ndarray API has been implemented several times
outside of NumPy itself for different architectures, such as for GPU
arrays (CuPy), Sparse arrays (scipy.sparse, pydata/sparse) and parallel
arrays (Dask array) as well as various NumPy-like implementations in the
deep learning frameworks, like TensorFlow and PyTorch.

Similarly there are many projects that build on top of the NumPy API
for labeled and indexed arrays (XArray), automatic differentiation
(Autograd, Tangent), masked arrays (numpy.ma), physical units (astropy.units,
pint, unyt), etc. that add additional functionality on top of the NumPy API.
Most of these project also implement a close variation of NumPy's level high
API.

We would like to be able to use these libraries together, for example we
would like to be able to place a CuPy array within XArray, or perform
automatic differentiation on Dask array code. This would be easier to
accomplish if code written for NumPy ndarrays could also be used by
other NumPy-like projects.

For example, we would like for the following code example to work
equally well with any NumPy-like array object:

.. code:: python

    def f(x):
        y = np.tensordot(x, x.T)
        return np.mean(np.exp(y))

Some of this is possible today with various protocol mechanisms within
NumPy.

-  The ``np.exp`` function checks the ``__array_ufunc__`` protocol
-  The ``.T`` method works using Python's method dispatch
-  The ``np.mean`` function explicitly checks for a ``.mean`` method on
   the argument

However other functions, like ``np.tensordot`` do not dispatch, and
instead are likely to coerce to a NumPy array (using the ``__array__``)
protocol, or err outright. To achieve enough coverage of the NumPy API
to support downstream projects like XArray and autograd we want to
support *almost all* functions within NumPy, which calls for a more
reaching protocol than just ``__array_ufunc__``. We would like a
protocol that allows arguments of a NumPy function to take control and
divert execution to another function (for example a GPU or parallel
implementation) in a way that is safe and consistent across projects.

Implementation
--------------

We propose adding support for a new protocol in NumPy,
``__array_function__``.

This protocol is intended to be a catch-all for NumPy functionality that
is not covered by the ``__array_ufunc__`` protocol for universal functions
(like ``np.exp``). The semantics are very similar to ``__array_ufunc__``, except
the operation is specified by an arbitrary callable object rather than a ufunc
instance and method.

A prototype implementation can be found in
`this notebook <https://nbviewer.jupyter.org/gist/shoyer/1f0a308a06cd96df20879a1ddb8f0006>`_.

.. warning::

  The ``__array_function__`` protocol, and its use on particular functions,
  is *experimental*. We plan to retain an interface that makes it possible
  to override NumPy functions, but the way to do so for particular functions
  **can and will change** with little warning. If such reduced backwards
  compatibility guarantees are not accepted to you, do not rely upon overrides
  of NumPy functions for non-NumPy arrays. See "Non-goals" below for more
  details.

.. note::

  Dispatch with the ``__array_function__`` protocol has been implemented but is
  not yet enabled by default:

  - In NumPy 1.16, you need to set the environment variable
    ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1`` before importing NumPy to test
    NumPy function overrides.
  - In NumPy 1.17, the protocol will be enabled by default, but can be disabled
    with ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0``.
  - Eventually, expect to ``__array_function__`` to always be enabled.

The interface
~~~~~~~~~~~~~

We propose the following signature for implementations of
``__array_function__``:

.. code-block:: python

    def __array_function__(self, func, types, args, kwargs)

-  ``func`` is an arbitrary callable exposed by NumPy's public API,
   which was called in the form ``func(*args, **kwargs)``.
-  ``types`` is a `collection <https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection>`_
   of unique argument types from the original NumPy function call that
   implement ``__array_function__``.
-  The tuple ``args`` and dict ``kwargs`` are directly passed on from the
   original call.

Unlike ``__array_ufunc__``, there are no high-level guarantees about the
type of ``func``, or about which of ``args`` and ``kwargs`` may contain objects
implementing the array API.

As a convenience for ``__array_function__`` implementors, ``types`` provides all
argument types with an ``'__array_function__'`` attribute. This
allows implementors to quickly identify cases where they should defer to
``__array_function__`` implementations on other arguments.
The type of ``types`` is intentionally vague:
``frozenset`` would most closely match intended use, but we may use ``tuple``
instead for performance reasons. In any case, ``__array_function__``
implementations should not rely on the iteration order of ``types``, which
would violate a well-defined "Type casting hierarchy" (as described in
`NEP-13 <https://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_).

Example for a project implementing the NumPy API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most implementations of ``__array_function__`` will start with two
checks:

1.  Is the given function something that we know how to overload?
2.  Are all arguments of a type that we know how to handle?

If these conditions hold, ``__array_function__`` should return
the result from calling its implementation for ``func(*args, **kwargs)``.
Otherwise, it should return the sentinel value ``NotImplemented``, indicating
that the function is not implemented by these types. This is preferable to
raising ``TypeError`` directly, because it gives *other* arguments the
opportunity to define the operations.

There are no general requirements on the return value from
``__array_function__``, although most sensible implementations should probably
return array(s) with the same type as one of the function's arguments.
If/when Python gains
`typing support for protocols <https://www.python.org/dev/peps/pep-0544/>`_
and NumPy adds static type annotations, the ``@overload`` implementation
for ``SupportsArrayFunction`` will indicate a return type of ``Any``.

It may also be convenient to define a custom decorators (``implements`` below)
for registering ``__array_function__`` implementations.

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

.. note::

    Just like the case for builtin special methods like ``__add__``, properly
    written ``__array_function__`` methods should always return
    ``NotImplemented`` when an unknown type is encountered. Otherwise, it will
    be impossible to correctly override NumPy functions from another object
    if the operation also includes one of your objects.

Necessary changes within the NumPy codebase itself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will require two changes within the NumPy codebase:

1. A function to inspect available inputs, look for the
   ``__array_function__`` attribute on those inputs, and call those
   methods appropriately until one succeeds.  This needs to be fast in the
   common all-NumPy case, and have acceptable performance (no worse than
   linear time) even if the number of overloaded inputs is large (e.g.,
   as might be the case for `np.concatenate`).

   This is one additional function of moderate complexity.
2. Calling this function within all relevant NumPy functions.

   This affects many parts of the NumPy codebase, although with very low
   complexity.

Finding and calling the right ``__array_function__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a NumPy function, ``*args`` and ``**kwargs`` inputs, we need to
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
   might be prohibitively expensive.
2. Some functions may have arguments that we don't want to inspect, even
   if they have the ``__array_function__`` method.

To resolve these issues, NumPy functions should explicitly indicate which
of their arguments may be overloaded, and how these arguments should be
checked. As a rule, this should include all arguments documented as either
``array_like`` or ``ndarray``.

We propose to do so by writing "dispatcher" functions for each overloaded
NumPy function:

- These functions will be called with the exact same arguments that were passed
  into the NumPy function (i.e., ``dispatcher(*args, **kwargs)``), and should
  return an iterable of arguments to check for overrides.
- Dispatcher functions are required to share the exact same positional,
  optional and keyword-only arguments as their corresponding NumPy functions.
  Otherwise, valid invocations of a NumPy function could result in an error when
  calling its dispatcher.
- Because default *values* for keyword arguments do not have
  ``__array_function__`` attributes, by convention we set all default argument
  values to ``None``. This reduces the likelihood of signatures falling out
  of sync, and minimizes extraneous information in the dispatcher.
  The only exception should be cases where the argument value in some way
  effects dispatching, which should be rare.

An example of the dispatcher for ``np.concatenate`` may be instructive:

.. code:: python

    def _concatenate_dispatcher(arrays, axis=None, out=None):
        for array in arrays:
            yield array
        if out is not None:
            yield out

The concatenate dispatcher is written as generator function, which allows it
to potentially include the value of the optional ``out`` argument without
needing to create a new sequence with the (potentially long) list of objects
to be concatenated.

Trying ``__array_function__`` methods until the right one works
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Many arguments may implement the ``__array_function__`` protocol. Some
of these may decide that, given the available inputs, they are unable to
determine the correct result. How do we call the right one? If several
are valid then which has precedence?

For the most part, the rules for dispatch with ``__array_function__``
match those for ``__array_ufunc__`` (see
`NEP-13 <https://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_).
In particular:

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

If no ``__array_function__`` methods exist, NumPy will default to calling its
own implementation, intended for use on NumPy arrays. This case arises, for
example, when all array-like arguments are Python numbers or lists.
(NumPy arrays do have a ``__array_function__`` method, given below, but it
always returns ``NotImplemented`` if any argument other than a NumPy array
subclass implements ``__array_function__``.)

One deviation from the current behavior of ``__array_ufunc__`` is that NumPy
will only call ``__array_function__`` on the *first* argument of each unique
type. This matches Python's
`rule for calling reflected methods <https://docs.python.org/3/reference/datamodel.html#object.__ror__>`_,
and this ensures that checking overloads has acceptable performance even when
there are a large number of overloaded arguments. To avoid long-term divergence
between these two dispatch protocols, we should
`also update <https://github.com/numpy/numpy/issues/11306>`_
``__array_ufunc__`` to match this behavior.

The ``__array_function__`` method on ``numpy.ndarray``
''''''''''''''''''''''''''''''''''''''''''''''''''''''

The use cases for subclasses with ``__array_function__`` are the same as those
with ``__array_ufunc__``, so ``numpy.ndarray`` also defines a
``__array_function__`` method:

.. code:: python

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, ndarray) for t in types):
            # Defer to any non-subclasses that implement __array_function__
            return NotImplemented

        # Use NumPy's private implementation without __array_function__
        # dispatching
        return func._implementation(*args, **kwargs)

This method matches NumPy's dispatching rules, so for most part it is
possible to pretend that ``ndarray.__array_function__`` does not exist.
The private ``_implementation`` attribute, defined below in the
``array_function_dispatch`` decorator, allows us to avoid the special cases for
NumPy arrays that were needed in the ``__array_ufunc__`` protocol.

The ``__array_function__`` protocol always calls subclasses before
superclasses, so if any ``ndarray`` subclasses are involved in an operation,
they will get the chance to override it, just as if any other argument
overrides ``__array_function__``. But the default behavior in an operation
that combines a base NumPy array and a subclass is different: if the subclass
returns ``NotImplemented``, NumPy's implementation of the function will be
called instead of raising an exception. This is appropriate since subclasses
are `expected to be substitutable <https://en.wikipedia.org/wiki/Liskov_substitution_principle>`_.

We still caution authors of subclasses to exercise caution when relying
upon details of NumPy's internal implementations. It is not always possible to
write a perfectly substitutable ndarray subclass, e.g., in cases involving the
creation of new arrays, not least because NumPy makes use of internal
optimizations specialized to base NumPy arrays, e.g., code written in C. Even
if NumPy's implementation happens to work today, it may not work in the future.
In these cases, your recourse is to re-implement top-level NumPy functions via
``__array_function__`` on your subclass.

Changes within NumPy functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a function defining the above behavior, for now call it
``implement_array_function``, we now need to call that
function from within every relevant NumPy function. This is a pervasive change,
but of fairly simple and innocuous code that should complete quickly and
without effect if no arguments implement the ``__array_function__``
protocol.

To achieve this, we define a ``array_function_dispatch`` decorator to rewrite
NumPy functions. The basic implementation is as follows:

.. code:: python

    def array_function_dispatch(dispatcher, module=None):
        """Wrap a function for dispatch with the __array_function__ protocol."""
        def decorator(implementation):
            @functools.wraps(implementation)
            def public_api(*args, **kwargs):
                relevant_args = dispatcher(*args, **kwargs)
                return implement_array_function(
                    implementation, public_api, relevant_args, args, kwargs)
            if module is not None:
                public_api.__module__ = module
            # for ndarray.__array_function__
            public_api._implementation = implementation
            return public_api
        return decorator

    # example usage
    def _broadcast_to_dispatcher(array, shape, subok=None):
        return (array,)

    @array_function_dispatch(_broadcast_to_dispatcher, module='numpy')
    def broadcast_to(array, shape, subok=False):
        ...  # existing definition of np.broadcast_to

Using a decorator is great! We don't need to change the definitions of
existing NumPy functions, and only need to write a few additional lines
for the dispatcher function. We could even reuse a single dispatcher for
families of functions with the same signature (e.g., ``sum`` and ``prod``).
For such functions, the largest change could be adding a few lines to the
docstring to note which arguments are checked for overloads.

It's particularly worth calling out the decorator's use of
``functools.wraps``:

- This ensures that the wrapped function has the same name and docstring as
  the wrapped NumPy function.
- On Python 3, it also ensures that the decorator function copies the original
  function signature, which is important for introspection based tools such as
  auto-complete.
- Finally, it ensures that the wrapped function
  `can be pickled <http://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html>`_.

The example usage illustrates several best practices for writing dispatchers
relevant to NumPy contributors:

- We passed the ``module`` argument, which in turn sets the  ``__module__``
  attribute on the generated function. This is for the benefit of better error
  messages, here for errors raised internally by NumPy when no implementation
  is found, e.g.,
  ``TypeError: no implementation found for 'numpy.broadcast_to'``. Setting
  ``__module__`` to the canonical location in NumPy's public API encourages
  users to use NumPy's public API for identifying functions in
  ``__array_function__``.

- The dispatcher is a function that returns a tuple, rather than an equivalent
  (and equally valid) generator using ``yield``:

  .. code:: python

    # example usage
    def broadcast_to(array, shape, subok=None):
        yield array

  This is no accident: NumPy's implementation of dispatch for
  ``__array_function__`` is fastest when dispatcher functions return a builtin
  sequence type (``tuple`` or ``list``).

  On a related note, it's perfectly fine for dispatchers to return arguments
  even if in some cases you *know* that they cannot have an
  ``__array_function__`` method. This can arise for functions with default
  arguments (e.g., ``None``) or complex signatures. NumPy's dispatching logic
  sorts out these cases very quickly, so it generally is not worth the trouble
  of parsing them on your own.

.. note::

    The code for ``array_function_dispatch`` above has been updated from the
    original version of this NEP to match the actual
    `implementation in NumPy <https://github.com/numpy/numpy/blob/e104f03ac8f65ae5b92a9b413b0fa639f39e6de2/numpy/core/overrides.py>`_.

Extensibility
~~~~~~~~~~~~~

An important virtue of this approach is that it allows for adding new
optional arguments to NumPy functions without breaking code that already
relies on ``__array_function__``.

This is not a theoretical concern. NumPy's older, haphazard implementation of
overrides *within* functions like ``np.sum()`` necessitated some awkward
gymnastics when we decided to add new optional arguments, e.g., the new
``keepdims`` argument is only passed in cases where it is used:

.. code:: python

    def sum(array, ..., keepdims=np._NoValue):
        kwargs = {}
        if keepdims is not np._NoValue:
            kwargs['keepdims'] = keepdims
        return array.sum(..., **kwargs)

For ``__array_function__`` implementors, this also means that it is possible
to implement even existing optional arguments incrementally, and only in cases
where it makes sense. For example, a library implementing immutable arrays
would not be required to explicitly include an unsupported ``out`` argument in
the function signature. This can be somewhat onerous to implement properly,
e.g.,

.. code:: python

    def my_sum(array, ..., out=None):
        if out is not None:
            raise TypeError('out argument is not supported')
        ...

We thus avoid encouraging the tempting shortcut of adding catch-all
``**ignored_kwargs`` to the signatures of functions called by NumPy, which fails
silently for misspelled or ignored arguments.

Performance
~~~~~~~~~~~

Performance is always a concern with NumPy, even though NumPy users have
already prioritized usability over pure speed with their choice of the Python
language itself. It's important that this new ``__array_function__`` protocol
not impose a significant cost in the typical case of NumPy functions acting
on NumPy arrays.

Our `microbenchmark results <https://nbviewer.jupyter.org/gist/shoyer/1f0a308a06cd96df20879a1ddb8f0006>`_
show that a pure Python implementation of the override machinery described
above adds roughly 2-3 microseconds of overhead to each NumPy function call
without any overloaded arguments. For context, typical NumPy functions on small
arrays have a runtime of 1-10 microseconds, mostly determined by what fraction
of the function's logic is written in C. For example, one microsecond is about
the difference in speed between the ``ndarray.sum()`` method (1.6 us) and
``numpy.sum()`` function (2.6 us).

Fortunately, we expect significantly less overhead with a C implementation of
``implement_array_function``, which is where the bulk of the
runtime is. This would leave the ``array_function_dispatch`` decorator and
dispatcher function on their own adding about 0.5 microseconds of overhead,
for perhaps ~1 microsecond of overhead in the typical case.

In our view, this level of overhead is reasonable to accept for code written
in Python. We're pretty sure that the vast majority of NumPy users aren't
concerned about performance differences measured in microsecond(s) on NumPy
functions, because it's difficult to do *anything* in Python in less than a
microsecond.

Use outside of NumPy
~~~~~~~~~~~~~~~~~~~~

Nothing about this protocol that is particular to NumPy itself. Should
we encourage use of the same ``__array_function__`` protocol third-party
libraries for overloading non-NumPy functions, e.g., for making
array-implementation generic functionality in SciPy?

This would offer significant advantages (SciPy wouldn't need to invent
its own dispatch system) and no downsides that we can think of, because
every function that dispatches with ``__array_function__`` already needs
to be explicitly recognized. Libraries like Dask, CuPy, and Autograd
already wrap a limited subset of SciPy functionality (e.g.,
``scipy.linalg``) similarly to how they wrap NumPy.

If we want to do this, we should expose at least the decorator
``array_function_dispatch()`` and possibly also the lower level
``implement_array_function()`` as part of NumPy's public API.

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
otherwise bug-fix only releases of NumPy. In practice, once initial
issues with ``__array_function__`` are worked out, we will use abbreviated
deprecation cycles as short as a single major NumPy release (e.g., as
little as four months).

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

We also expect that the mechanism for overriding specific functions
that will initially use the ``__array_function__`` protocol can and will
change in the future. As a concrete example of how we expect to break
behavior in the future, some functions such as ``np.where`` are currently
not NumPy universal functions, but conceivably could become universal
functions in the future. When/if this happens, we will change such overloads
from using ``__array_function__`` to the more specialized ``__array_ufunc__``.


Backward compatibility
----------------------

This proposal does not change existing semantics, except for those arguments
that currently have ``__array_function__`` attributes, which should be rare.


Alternatives
------------

Specialized protocols
~~~~~~~~~~~~~~~~~~~~~

We could (and should) continue to develop protocols like
``__array_ufunc__`` for cohesive subsets of NumPy functionality.

As mentioned above, if this means that some functions that we overload
with ``__array_function__`` should switch to a new protocol instead,
that is explicitly OK for as long as ``__array_function__`` retains its
experimental status.

Switching to a new protocol should use an abbreviated version of NumPy's
normal deprecation cycle:

- For a single major release, after checking for any new protocols, NumPy
  should still check for ``__array_function__`` methods that implement the
  given function. If any argument returns a value other than
  ``NotImplemented`` from ``__array_function__``, a descriptive
  ``FutureWarning`` should be issued.
- In the next major release, the checks for ``__array_function__`` will be
  removed.

Separate namespace
~~~~~~~~~~~~~~~~~~

A separate namespace for overloaded functions is another possibility,
either inside or outside of NumPy.

This has the advantage of alleviating any possible concerns about
backwards compatibility and would provide the maximum freedom for quick
experimentation. In the long term, it would provide a clean abstraction
layer, separating NumPy's high level API from default implementations on
``numpy.ndarray`` objects.

The downsides are that this would require an explicit opt-in from all
existing code, e.g., ``import numpy.api as np``, and in the long term
would result in the maintenance of two separate NumPy APIs. Also, many
functions from ``numpy`` itself are already overloaded (but
inadequately), so confusion about high vs. low level APIs in NumPy would
still persist.

Alternatively, a separate namespace, e.g., ``numpy.array_only``, could be
created for a non-overloaded version of NumPy's high level API, for cases
where performance with NumPy arrays is a critical concern. This has most
of the same downsides as the separate namespace.

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
system for arithmetic, and it would be confusing to add another
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

The internal implementation of some NumPy functions is extremely simple.
For example:

- ``np.stack()`` is implemented in only a few lines of code by combining
  indexing with ``np.newaxis``, ``np.concatenate`` and the ``shape`` attribute.
- ``np.mean()`` is implemented internally in terms of ``np.sum()``,
  ``np.divide()``, ``.astype()`` and ``.shape``.

This suggests the possibility of defining a minimal "core" ndarray
interface, and relying upon it internally in NumPy to implement the full
API. This is an attractive option, because it could significantly reduce
the work required for new array implementations.

However, this also comes with several downsides:

1. The details of how NumPy implements a high-level function in terms of
   overloaded functions now becomes an implicit part of NumPy's public API. For
   example, refactoring ``stack`` to use ``np.block()`` instead of
   ``np.concatenate()`` internally would now become a breaking change.
2. Array libraries may prefer to implement high level functions differently than
   NumPy. For example, a library might prefer to implement a fundamental
   operations like ``mean()`` directly rather than relying on ``sum()`` followed
   by division. More generally, it's not clear yet what exactly qualifies as
   core functionality, and figuring this out could be a large project.
3. We don't yet have an overloading system for attributes and methods on array
   objects, e.g., for accessing ``.dtype`` and ``.shape``. This should be the
   subject of a future NEP, but until then we should be reluctant to rely on
   these properties.

Given these concerns, we think it's valuable to support explicit overloading of
nearly every public function in NumPy's API. This does not preclude the future
possibility of rewriting NumPy functions in terms of simplified core
functionality with ``__array_function__`` and a protocol and/or base class for
ensuring that arrays expose methods and properties like ``numpy.ndarray``.
However, to work well this would require the possibility of implementing
*some* but not all functions with ``__array_function__``, e.g., as described
in the next section.

Partial implementation of NumPy's API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Even libraries that reimplement most of NumPy's public API sometimes rely upon
using utility functions from NumPy without a wrapper. For example, both CuPy
and JAX simply `use an alias <https://github.com/numpy/numpy/issues/12974>`_ to
``np.result_type``, which already supports duck-types with a ``dtype``
attribute.

With ``__array_ufunc__``, it's possible to alleviate this concern by
casting all arguments to numpy arrays and re-calling the ufunc, but the
heterogeneous function signatures supported by ``__array_function__``
make it impossible to implement this generic fallback behavior for
``__array_function__``.

We considered three possible ways to resolve this issue, but none were
entirely satisfactory:

1. Change the meaning of all arguments returning ``NotImplemented`` from
   ``__array_function__`` to indicate that all arguments should be coerced to
   NumPy arrays and the operation should be retried. However, many array
   libraries (e.g., scipy.sparse) really don't want implicit conversions to
   NumPy arrays, and often avoid implementing ``__array__`` for exactly this
   reason. Implicit conversions can result in silent bugs and performance
   degradation.

   Potentially, we could enable this behavior only for types that implement
   ``__array__``, which would resolve the most problematic cases like
   scipy.sparse. But in practice, a large fraction of classes that present a
   high level API like NumPy arrays already implement ``__array__``. This would
   preclude reliable use of NumPy's high level API on these objects.

2. Use another sentinel value of some sort, e.g.,
   ``np.NotImplementedButCoercible``, to indicate that a class implementing
   part of NumPy's higher level array API is coercible as a fallback. If all
   arguments return ``NotImplementedButCoercible``, arguments would be coerced
   and the operation would be retried.

   Unfortunately, correct behavior after encountering
   ``NotImplementedButCoercible`` is not always obvious. Particularly
   challenging is the "mixed" case where some arguments return
   ``NotImplementedButCoercible`` and others return ``NotImplemented``.
   Would dispatching be retried after only coercing the "coercible" arguments?
   If so, then conceivably we could end up looping through the dispatching
   logic an arbitrary number of times. Either way, the dispatching rules would
   definitely get more complex and harder to reason about.

3. Allow access to NumPy's implementation of functions, e.g., in the form of
   a publicly exposed ``__skip_array_function__`` attribute on the NumPy
   functions. This would allow for falling back to NumPy's implementation by
   using ``func.__skip_array_function__`` inside ``__array_function__``
   methods, and could also potentially be used to be used to avoid the
   overhead of dispatching. However, it runs the risk of potentially exposing
   details of NumPy's implementations for NumPy functions that do not call
   ``np.asarray()`` internally. See
   `this note <https://mail.python.org/pipermail/numpy-discussion/2019-May/079541.html>`_
   for a summary of the full discussion.

These solutions would solve real use cases, but at the cost of additional
complexity. We would like to gain experience with how ``__array_function__`` is
actually used before making decisions that would be difficult to roll back.

A magic decorator that inspects type annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In principle, Python 3 type annotations contain sufficient information to
automatically create most ``dispatcher`` functions. It would be convenient to
use these annotations to dispense with the need for manually writing
dispatchers, e.g.,

.. code:: python

    @array_function_dispatch
    def broadcast_to(array: ArrayLike
                     shape: Tuple[int, ...],
                     subok: bool = False):
        ...  # existing definition of np.broadcast_to

This would require some form of automatic code generation, either at compile or
import time.

We think this is an interesting possible extension to consider in the future. We
don't think it makes sense to do so now, because code generation involves
tradeoffs and NumPy's experience with type annotations is still
`quite limited <https://github.com/numpy/numpy-stubs>`_. Even if NumPy
was Python 3 only (which will happen
`sometime in 2019 <http://www.numpy.org/neps/nep-0014-dropping-python2.7-proposal.html>`_),
we aren't ready to annotate NumPy's codebase directly yet.

Support for implementation-specific arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We could allow ``__array_function__`` implementations to add their own
optional keyword arguments by including ``**ignored_kwargs`` in dispatcher
functions, e.g.,

.. code:: python

    def _concatenate_dispatcher(arrays, axis=None, out=None, **ignored_kwargs):
        ...  # same implementation of _concatenate_dispatcher as above

Implementation-specific arguments are somewhat common in libraries that
otherwise emulate NumPy's higher level API (e.g., ``dask.array.sum()`` adds
``split_every`` and ``tensorflow.reduce_sum()`` adds ``name``). Supporting
them in NumPy would be particularly useful for libraries that implement new
high-level array functions on top of NumPy functions, e.g.,

.. code:: python

    def mean_squared_error(x, y, **kwargs):
        return np.mean((x - y) ** 2, **kwargs)

Otherwise, we would need separate versions of ``mean_squared_error`` for each
array implementation in order to pass implementation-specific arguments to
``mean()``.

We wouldn't allow adding optional positional arguments, because these are
reserved for future use by NumPy itself, but conflicts between keyword arguments
should be relatively rare.

However, this flexibility would come with a cost. In particular, it implicitly
adds ``**kwargs`` to the signature for all wrapped NumPy functions without
actually including it (because we use ``functools.wraps``). This means it is
unlikely to work well with static analysis tools, which could report invalid
arguments. Likewise, there is a price in readability: these optional arguments
won't be included in the docstrings for NumPy functions.

It's not clear that this tradeoff is worth it, so we propose to leave this out
for now. Adding implementation-specific arguments will require using those
libraries directly.

Other possible choices for the protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The array function ``__array_function__`` includes only two arguments, ``func``
and ``types``, that provide information about the context of the function call.

``func`` is part of the protocol because there is no way to avoid it:
implementations need to be able to dispatch by matching a function to NumPy's
public API.

``types`` is included because we can compute it almost for free as part of
collecting ``__array_function__`` implementations to call in
``implement_array_function``. We also think it will be used
by many ``__array_function__`` methods, which otherwise would need to extract
this information themselves. It would be equivalently easy to provide single
instances of each type, but providing only types seemed cleaner.

Taking this even further, it was suggested that ``__array_function__`` should be
a ``classmethod``. We agree that it would be a little cleaner to remove the
redundant ``self`` argument, but feel that this minor clean-up would not be
worth breaking from the precedence of ``__array_ufunc__``.

There are two other arguments that we think *might* be important to pass to
``__array_ufunc__`` implementations:

- Access to the non-dispatched implementation (i.e., before wrapping with
  ``array_function_dispatch``) in ``ndarray.__array_function__`` would allow
  us to drop special case logic for that method from
  ``implement_array_function``.
- Access to the ``dispatcher`` function passed into
  ``array_function_dispatch()`` would allow ``__array_function__``
  implementations to determine the list of "array-like" arguments in a generic
  way by calling ``dispatcher(*args, **kwargs)``. This *could* be useful for
  ``__array_function__`` implementations that dispatch based on the value of an
  array attribute (e.g., ``dtype`` or ``units``) rather than directly on the
  array type.

We have left these out for now, because we don't know that they are necessary.
If we want to include them in the future, the easiest way to do so would be to
update the ``array_function_dispatch`` decorator to add them as function
attributes.

Callable objects generated at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy has some APIs that define callable objects *dynamically*, such as
``vectorize`` and methods on ``random.RandomState`` object. Examples can
also be found in other core libraries in the scientific Python stack, e.g.,
distribution objects in scipy.stats and model objects in scikit-learn. It would
be nice to be able to write overloads for such callables, too. This presents a
challenge for the ``__array_function__`` protocol, because unlike the case for
functions there is no public object in the ``numpy`` namespace to pass into
the ``func`` argument.

We could potentially handle this by establishing an alternative convention
for how the ``func`` argument could be inspected, e.g., by using
``func.__self__`` to obtain the class object and ``func.__func__`` to return
the unbound function object. However, some caution is in order, because
this would immesh what are currently implementation details as a permanent
features of the interface, such as the fact that ``vectorize`` is implemented as a
class rather than closure, or whether a method is implemented directly or using
a descriptor.

Given the complexity and the limited use cases, we are also deferring on this
issue for now, but we are confident that ``__array_function__`` could be
expanded to accommodate these use cases in the future if need be.

Discussion
----------

Various alternatives to this proposal were discussed in a few GitHub issues:

1. `pydata/sparse #1 <https://github.com/pydata/sparse/issues/1>`_
2. `numpy/numpy #11129 <https://github.com/numpy/numpy/issues/11129>`_

Additionally it was the subject of `a blogpost
<http://matthewrocklin.com/blog/work/2018/05/27/beyond-numpy>`_. Following this
it was discussed at a `NumPy developer sprint
<https://scisprints.github.io/#may-numpy-developer-sprint>`_ at the `UC
Berkeley Institute for Data Science (BIDS) <https://bids.berkeley.edu/>`_.

Detailed discussion of this proposal itself can be found on the
`the mailing list <https://mail.python.org/pipermail/numpy-discussion/2018-June/078127.html>`_ and relevant pull requests
(`1 <https://github.com/numpy/numpy/pull/11189>`_,
`2 <https://github.com/numpy/numpy/pull/11303#issuecomment-396638175>`_,
`3 <https://github.com/numpy/numpy/pull/11374>`_)

Copyright
---------

This document has been placed in the public domain.
