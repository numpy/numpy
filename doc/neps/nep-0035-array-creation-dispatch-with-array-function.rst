.. _NEP35:

===========================================================
NEP 35 — Array Creation Dispatching With __array_function__
===========================================================

:Author: Peter Andreas Entschev <pentschev@nvidia.com>
:Status: Draft
:Type: Standards Track
:Created: 2019-10-15
:Updated: 2020-11-06
:Resolution:

Abstract
--------

We propose the introduction of a new keyword argument ``like=`` to all array
creation functions to address one of the shortcomings of ``__array_function__``,
as described by NEP 18 [1]_. The ``like=`` keyword argument will create an
instance of the argument's type, enabling direct creation of non-NumPy arrays.
The target array type must implement the ``__array_function__`` protocol.

Motivation and Scope
--------------------

Many libraries implement the NumPy API, such as Dask for graph
computing, CuPy for GPGPU computing, xarray for N-D labeled arrays, etc. Underneath,
they have adopted the ``__array_function__`` protocol which allows NumPy to understand
and treat downstream objects as if they are the native ``numpy.ndarray`` object.
Hence the community while using various libraries still benefits from a unified
NumPy API. This not only brings great convenience for standardization but also
removes the burden of learning a new API and rewriting code for every new
object. In more technical terms, this mechanism of the protocol is called a
"dispatcher", which is the terminology we use from here onwards when referring
to that.


.. code:: python

    x = dask.array.arange(5)    # Creates dask.array
    np.diff(x)                  # Returns dask.array

Note above how we called Dask's implementation of ``diff`` via the NumPy
namespace by calling ``np.diff``, and the same would apply if we had a CuPy
array or any other array from a library that adopts ``__array_function__``.
This allows writing code that is agnostic to the implementation library, thus
users can write their code once and still be able to use different array
implementations according to their needs.

Obviously, having a protocol in-place is useful if the arrays are created
elsewhere and let NumPy handle them. But still these arrays have to be started
in their native library and brought back. Instead if it was possible to create
these objects through NumPy API then there would be an almost complete
experience, all using NumPy syntax. For example, say we have some CuPy array
``cp_arr``, and want a similar CuPy array with identity matrix. We could still
write the following:

.. code:: python

     x = cupy.identity(3)

Instead, the better way would be using to only use the NumPy API, this could now
be achieved with:

.. code:: python

    x = np.identity(3, like=cp_arr)

As if by magic, ``x`` will also be a CuPy array, as NumPy was capable to infer
that from the type of ``cp_arr``. Note that this last step would not be possible
without ``like=``, as it would be impossible for the NumPy to know the user
expects a CuPy array based only on the integer input.

The new ``like=`` keyword proposed is solely intended to identify the downstream
library where to dispatch and the object is used only as reference, meaning that
no modifications, copies or processing will be performed on that object.

We expect that this functionality will be mostly useful to library developers,
allowing them to create new arrays for internal usage based on arrays passed
by the user, preventing unnecessary creation of NumPy arrays that will
ultimately lead to an additional conversion into a downstream array type.

Support for Python 2.7 has been dropped since NumPy 1.17, therefore we make use
of the keyword-only argument standard described in PEP-3102 [2]_ to implement
``like=``, thus preventing it from being passed by position.

.. _neps.like-kwarg.usage-and-impact:

Usage and Impact
----------------

NumPy users who don't use other arrays from downstream libraries can continue
to use array creation routines without a ``like=`` argument. Using
``like=np.ndarray`` will work as if no array was passed via that argument.
However, this will incur additional checks that will negatively impact
performance.

To understand the intended use for ``like=``, and before we move to more complex
cases, consider the following illustrative example consisting only of NumPy and
CuPy arrays:

.. code:: python

    import numpy as np
    import cupy

    def my_pad(arr, padding):
        padding = np.array(padding, like=arr)
        return np.concatenate((padding, arr, padding))

    my_pad(np.arange(5), [-1, -1])    # Returns np.ndarray
    my_pad(cupy.arange(5), [-1, -1])  # Returns cupy.core.core.ndarray

Note in the ``my_pad`` function above how ``arr`` is used as a reference to
dictate what array type padding should have, before concatenating the arrays to
produce the result. On the other hand, if ``like=`` wasn't used, the NumPy case
would still work, but CuPy wouldn't allow this kind of automatic
conversion, ultimately raising a
``TypeError: Only cupy arrays can be concatenated`` exception.

Now we should look at how a library like Dask could benefit from ``like=``.
Before we understand that, it's important to understand a bit about Dask basics
and how it ensures correctness with ``__array_function__``. Note that Dask can
perform computations on different sorts of objects, like dataframes, bags and
arrays, here we will focus strictly on arrays, which are the objects we can use
``__array_function__`` with.

Dask uses a graph computing model, meaning it breaks down a large problem in
many smaller problems and merges their results to reach the final result. To
break the problem down into smaller ones, Dask also breaks arrays into smaller
arrays that it calls "chunks". A Dask array can thus consist of one or more
chunks and they may be of different types. However, in the context of
``__array_function__``, Dask only allows chunks of the same type; for example,
a Dask array can be formed of several NumPy arrays or several CuPy arrays, but
not a mix of both.

To avoid mismatched types during computation, Dask keeps an attribute ``_meta`` as
part of its array throughout computation: this attribute is used to both predict
the output type at graph creation time, and to create any intermediary arrays
that are necessary within some function's computation. Going back to our
previous example, we can use ``_meta`` information to identify what kind of
array we would use for padding, as seen below:

.. code:: python

    import numpy as np
    import cupy
    import dask.array as da
    from dask.array.utils import meta_from_array

    def my_dask_pad(arr, padding):
        padding = np.array(padding, like=meta_from_array(arr))
        return np.concatenate((padding, arr, padding))

    # Returns dask.array<concatenate, shape=(9,), dtype=int64, chunksize=(5,), chunktype=numpy.ndarray>
    my_dask_pad(da.arange(5), [-1, -1])

    # Returns dask.array<concatenate, shape=(9,), dtype=int64, chunksize=(5,), chunktype=cupy.ndarray>
    my_dask_pad(da.from_array(cupy.arange(5)), [-1, -1])

Note how ``chunktype`` in the return value above changes from
``numpy.ndarray`` in the first ``my_dask_pad`` call to ``cupy.ndarray`` in the
second. We have also renamed the function to ``my_dask_pad`` in this example
with the intent to make it clear that this is how Dask would implement such
functionality, should it need to do so, as it requires Dask's internal tools
that are not of much use elsewhere.

To enable proper identification of the array type we use Dask's utility function
``meta_from_array``, which was introduced as part of the work to support
``__array_function__``, allowing Dask to handle ``_meta`` appropriately. Readers
can think of ``meta_from_array`` as a special function that just returns the
type of the underlying Dask array, for example:

.. code:: python

    np_arr = da.arange(5)
    cp_arr = da.from_array(cupy.arange(5))

    meta_from_array(np_arr)  # Returns a numpy.ndarray
    meta_from_array(cp_arr)  # Returns a cupy.ndarray

Since the value returned by ``meta_from_array`` is a NumPy-like array, we can
just pass that directly into the ``like=`` argument.

The ``meta_from_array`` function is primarily targeted at the library's internal
usage to ensure chunks are created with correct types. Without the ``like=``
argument, it would be impossible to ensure ``my_pad`` creates a padding array
with a type matching that of the input array, which would cause a ``TypeError``
exception to be raised by CuPy, as discussed above would happen to the CuPy case
alone. Combining Dask's internal handling of meta arrays and the proposed
``like=`` argument, it now becomes possible to handle cases involving creation
of non-NumPy arrays, which is likely the heaviest limitation Dask currently
faces from the ``__array_function__`` protocol.

Backward Compatibility
----------------------

This proposal does not raise any backward compatibility issues within NumPy,
given that it only introduces a new keyword argument to existing array creation
functions with a default ``None`` value, thus not changing current behavior.

Detailed description
--------------------

The introduction of the ``__array_function__`` protocol allowed downstream
library developers to use NumPy as a dispatching API. However, the protocol
did not -- and did not intend to -- address the creation of arrays by downstream
libraries, preventing those libraries from using such important functionality in
that context.

The purpose of this NEP is to address that shortcoming in a simple and
straighforward way: introduce a new ``like=`` keyword argument, similar to how
the ``empty_like`` family of functions work. When array creation functions
receive such an argument, they will trigger the ``__array_function__`` protocol,
and call the downstream library's own array creation function implementation.
The ``like=`` argument, as its own name suggests, shall be used solely for the
purpose of identifying where to dispatch.  In contrast to the way
``__array_function__`` has been used so far (the first argument identifies the
target downstream library), and to avoid breaking NumPy's API with regards to
array creation, the new ``like=`` keyword shall be used for the purpose of
dispatching.

Downstream libraries will benefit from the ``like=`` argument without any
changes to their API, given the argument only needs to be implemented by NumPy.
It's still allowed that downstream libraries include the ``like=`` argument,
as it can be useful in some cases, please refer to
:ref:`neps.like-kwarg.implementation` for details on those cases. It will still
be required that downstream libraries implement the ``__array_function__``
protocol, as described by NEP 18 [1]_, and appropriately introduce the argument
to their calls to NumPy array creation functions, as exemplified in
:ref:`neps.like-kwarg.usage-and-impact`.

Related work
------------

Other NEPs have been written to address parts of ``__array_function__``
protocol's limitation, such as the introduction of the ``__duckarray__``
protocol in NEP 30 [3]_, and the introduction of an overriding mechanism called
``uarray`` by NEP 31 [4]_.

.. _neps.like-kwarg.implementation:

Implementation
--------------

The implementation requires introducing a new ``like=`` keyword to all existing
array creation functions of NumPy. As examples of functions that would add this
new argument (but not limited to) we can cite those taking array-like objects
such as ``array`` and ``asarray``, functions that create arrays based on
numerical inputs such as ``range`` and ``identity``, as well as the ``empty``
family of functions, even though that may be redundant, since specializations
for those already exist with the naming format ``empty_like``. As of the
writing of this NEP, a complete list of array creation functions can be
found in [5]_.

This newly proposed keyword shall be removed by the ``__array_function__``
mechanism from the keyword dictionary before dispatching. The purpose for this
is twofold:

1. Simplifies adoption of array creation by those libraries already opting-in
   to implement the ``__array_function__`` protocol, thus removing the
   requirement to explicitly opt-in for all array creation functions; and
2. Most downstream libraries will have no use for the keyword argument, and
   those that do may accomplish so by capturing ``self`` from
   ``__array_function__``.

Downstream libraries thus do not require to include the ``like=`` keyword to
their array creation APIs. In some cases (e.g., Dask), having the ``like=``
keyword can be useful, as it would allow the implementation to identify
array internals. As an example, Dask could benefit from the reference array
to identify its chunk type (e.g., NumPy, CuPy, Sparse), and thus create a new
Dask array backed by the same chunk type, something that's not possible unless
Dask can read the reference array's attributes.

Function Dispatching
~~~~~~~~~~~~~~~~~~~~

There are two different cases to dispatch: Python functions, and C functions.
To permit ``__array_function__`` dispatching, one possible implementation is to
decorate Python functions with ``overrides.array_function_dispatch``, but C
functions have a different requirement, which we shall describe shortly.

The example below shows a suggestion on how the ``asarray`` could be decorated
with ``overrides.array_function_dispatch``:

.. code:: python

    def _asarray_decorator(a, dtype=None, order=None, *, like=None):
        return (like,)

    @set_module('numpy')
    @array_function_dispatch(_asarray_decorator)
    def asarray(a, dtype=None, order=None, *, like=None):
        return array(a, dtype, copy=False, order=order)

Note in the example above that the implementation remains unchanged, the only
difference is the decoration, which uses the new ``_asarray_decorator`` function
to instruct the ``__array_function__`` protocol to dispatch if ``like`` is not
``None``.

We will now look at a C function example, and since ``asarray`` is anyway a
specialization of ``array``, we will use the latter as an example now. As
``array`` is a C function, currently all NumPy does regarding its Python source
is to import the function and adjust its ``__module__`` to ``numpy``. The
function will now be decorated with a specialization of
``overrides.array_function_from_dispatcher``, which shall take care of adjusting
the module too.

.. code:: python

    array_function_nodocs_from_c_func_and_dispatcher = functools.partial(
        overrides.array_function_from_dispatcher,
        module='numpy', docs_from_dispatcher=False, verify=False)

    @array_function_nodocs_from_c_func_and_dispatcher(_multiarray_umath.array)
    def array(a, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
              like=None):
        return (like,)

There are two downsides to the implementation above for C functions:

1.  It creates another Python function call; and
2.  To follow current implementation standards, documentation should be attached
    directly to the Python source code.

The first version of this proposal suggested the implementation above as one
viable solution for NumPy functions implemented in C. However, due to the
downsides pointed out above we have decided to discard any changes on the Python
side and resolve those issues with a pure-C implementation. Please refer to
[7]_ for details.

Reading the Reference Array Downstream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As stated in the beginning of :ref:`neps.like-kwarg.implementation` section,
``like=`` is not propagated to the downstream library, nevertheless, it's still
possible to access it. This requires some changes in the downstream library's
``__array_function__`` definition, where the ``self`` attribute is in practice
that passed via ``like=``. This is the case because we use ``like=`` as the
dispatching array, unlike other compute functions covered by NEP-18 that usually
dispatch on the first positional argument.

An example of such use is to create a new Dask array while preserving its
backend type:

.. code:: python

    # Returns dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=cupy.ndarray>
    np.asarray([1, 2, 3], like=da.array(cp.array(())))

    # Returns a cupy.ndarray
    type(np.asarray([1, 2, 3], like=da.array(cp.array(()))).compute())

Note how above the array is backed by ``chunktype=cupy.ndarray``, and the
resulting array after computing it is also a ``cupy.ndarray``. If Dask did
not use the ``like=`` argument via the ``self`` attribute from
``__array_function__``, the example above would be backed by ``numpy.ndarray``
instead:

.. code:: python

    # Returns dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.ndarray>
    np.asarray([1, 2, 3], like=da.array(cp.array(())))

    # Returns a numpy.ndarray
    type(np.asarray([1, 2, 3], like=da.array(cp.array(()))).compute())

Given the library would need to rely on ``self`` attribute from
``__array_function__`` to dispatch the function with the correct reference
array, we suggest one of two alternatives:

1. Introduce a list of functions in the downstream library that do support the
   ``like=`` argument and pass ``like=self`` when calling the function; or
2. Inspect whether the function's signature and verify whether it includes the
   ``like=`` argument. Note that this may incur in a higher performance penalty
   and assumes introspection is possible, which may not be if the function is
   a C function.

To make things clearer, let's take a look at how suggestion 2 could be
implemented in Dask. The current relevant part of ``__array_function__``
definition in Dask is seen below:

.. code:: python

    def __array_function__(self, func, types, args, kwargs):
        # Code not relevant for this example here

        # Dispatch ``da_func`` (da.asarray, for example) with *args and **kwargs
        da_func(*args, **kwargs)

And this is how the updated code would look like:

.. code:: python

    def __array_function__(self, func, types, args, kwargs):
        # Code not relevant for this example here

        # Inspect ``da_func``'s  signature and store keyword-only arguments
        import inspect
        kwonlyargs = inspect.getfullargspec(da_func).kwonlyargs

        # If ``like`` is contained in ``da_func``'s signature, add ``like=self``
        # to the kwargs dictionary.
        if 'like' in kwonlyargs:
            kwargs['like'] = self

        # Dispatch ``da_func`` (da.asarray, for example) with args and kwargs.
        # Here, kwargs contain ``like=self`` if the function's signature does too.
        da_func(*args, **kwargs)

Alternatives
------------

Recently a new protocol to replace ``__array_function__`` entirely was proposed
by NEP 37 [6]_, which would require considerable rework by downstream libraries
that adopt ``__array_function__`` already, because of that we still believe the
``like=`` argument is beneficial for NumPy and downstream libraries. However,
that proposal wouldn't necessarily be considered a direct alternative to the
present NEP, as it would replace NEP 18 entirely, upon which this builds.
Discussion on details about this new proposal and why that would require rework
by downstream libraries is beyond the scope of the present proposal.

Discussion
----------

- `Further discussion on implementation and the NEP's content <https://mail.python.org/pipermail/numpy-discussion/2020-August/080919.html>`_
- `Decision to release an experimental implementation in NumPy 1.20.0 <https://mail.python.org/pipermail/numpy-discussion/2020-November/081193.html>`__


References
----------

.. [1] `NEP 18 - A dispatch mechanism for NumPy's high level array functions <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_.

.. [2] `PEP 3102 — Keyword-Only Arguments <https://www.python.org/dev/peps/pep-3102/>`_.

.. [3] `NEP 30 — Duck Typing for NumPy Arrays - Implementation <https://numpy.org/neps/nep-0030-duck-array-protocol.html>`_.

.. [4] `NEP 31 — Context-local and global overrides of the NumPy API <https://github.com/numpy/numpy/pull/14389>`_.

.. [5] `Array creation routines <https://docs.scipy.org/doc/numpy-1.17.0/reference/routines.array-creation.html>`_.

.. [6] `NEP 37 — A dispatch protocol for NumPy-like modules <https://numpy.org/neps/nep-0037-array-module.html>`_.

.. [7] `Implementation's pull request on GitHub <https://github.com/numpy/numpy/pull/16935>`_

Copyright
---------

This document has been placed in the public domain.
