.. _NEP35:

===========================================================
NEP 35 — Array Creation Dispatching With __array_function__
===========================================================

:Author: Peter Andreas Entschev <pentschev@nvidia.com>
:Status: Draft
:Type: Standards Track
:Created: 2019-10-15
:Updated: 2020-08-17
:Resolution:

Abstract
--------

We propose the introduction of a new keyword argument ``like=`` to all array
creation functions, this argument permits the creation of an array based on
a non-NumPy reference array passed via that argument, resulting in an array
defined by the downstream library implementing that type, which also implements
the ``__array_function__`` protocol. With this we address one of that
protocol's shortcomings, as described by NEP 18 [1]_.

Motivation and Scope
--------------------

Many are the libraries implementing the NumPy API, such as Dask for graph
computing, CuPy for GPGPU computing, xarray for N-D labeled arrays, etc. All
the libraries mentioned have yet another thing in common: they have also adopted
the ``__array_function__`` protocol. The protocol defines a mechanism allowing a
user to directly use the NumPy API as a dispatcher based on the input array
type. In essence, dispatching means users are able to pass a downstream array,
such as a Dask array, directly to one of NumPy's compute functions, and NumPy
will be able to automatically recognize that and send the work back to Dask's
implementation of that function, which will define the return value. For
example:

.. code:: python

    x = dask.array.arange(5)    # Creates dask.array
    np.sum(a)                   # Returns dask.array

Note above how we called Dask's implementation of ``sum`` via the NumPy
namespace by calling ``np.sum``, and the same would apply if we had a CuPy
array or any other array from a library that adopts ``__array_function__``.
This allows writing code that is agnostic to the implementation library, thus
users can write their code once and still be able to use different array
implementations according to their needs.

Unfortunately, ``__array_function__`` has limitations, one of them being array
creation functions. In the example above, NumPy was able to call Dask's
implementation because the input array was a Dask array. The same is not true
for array creation functions, in the example the input of ``arange`` is simply
the integer ``5``, not providing any information of the array type that should
be the result, that's where a reference array passed by the ``like=`` argument
proposed here can be of help, as it provides NumPy with the information
required to create the expected type of array.

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
case would still work, but CuPy wouldn't allow this kind of automatic
conversion, ultimately raising a
``TypeError: Only cupy arrays can be concatenated`` exception.

Now we should look at how a library like Dask could benefit from ``like=``.
Before we understand that, it's important to understand a bit about Dask basics
and ensures correctness with ``__array_function__``. Note that Dask can compute
different sorts of objects, like dataframes, bags and arrays, here we will focus
strictly on arrays, which are the objects we can use ``__array_function__``
with.

Dask uses a graph computing model, meaning it breaks down a large problem in
many smaller problems and merge their results to reach the final result. To
break the problem down into smaller ones, Dask also breaks arrays into smaller
arrays, that it calls "chunks". A Dask array can thus consist of one or more
chunks and they may be of different types. However, in the context of
``__array_function__``, Dask only allows chunks of the same type, for example,
a Dask array can be formed of several NumPy arrays or several CuPy arrays, but
not a mix of both.

To avoid mismatched types during compute, Dask keeps an attribute ``_meta`` as
part of its array throughout computation, this attribute is used to both predict
the output type at graph creation time and to create any intermediary arrays
that are necessary within some function's computation. Going back to our
previous example, we can use ``_meta`` information to identify what kind of
array we would use for padding, as seen below:

.. code:: python

    import numpy as np
    import cupy
    import dask.array as da
    from dask.array.utils import meta_from_array

    def my_pad(arr, padding):
        padding = np.array(padding, like=meta_from_array(arr))
        return np.concatenate((padding, arr, padding))

    # Returns dask.array<concatenate, shape=(9,), dtype=int64, chunksize=(5,), chunktype=numpy.ndarray>
    my_pad(da.arange(5), [-1, -1])

    # Returns dask.array<concatenate, shape=(9,), dtype=int64, chunksize=(5,), chunktype=cupy.ndarray>
    my_pad(da.from_array(cupy.arange(5)), [-1, -1])

Note how ``chunktype`` in the return value above changes from
``numpy.ndarray`` in the first ``my_pad`` call to ``cupy.ndarray`` in the
second.

To enable proper identification of the array type we use Dask's utility function
``meta_from_array``, which was introduced as part of the work to support
``__array_function__``, allowing Dask to handle ``_meta`` appropriately. That
function is primarily targeted at the library's internal usage to ensure chunks
are created with correct types. Without the ``like=`` argument, it would be
impossible to ensure ``my_pad`` creates a padding array with a type matching
that of the input array, which would cause cause a ``TypeError`` exception to
be raised by CuPy, as discussed above would happen to the CuPy case alone.

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
changes to their API, given the argument is of exclusive implementation in
NumPy. It will still be required that downstream libraries implement the
``__array_function__`` protocol, as described by NEP 18 [1]_, and appropriately
introduce the argument to their calls to NumPy array creation functions, as
exemplified in :ref:`neps.like-kwarg.usage-and-impact`.

Related work
------------

Other NEPs have been written to address parts of ``__array_function__``
protocol's limitation, such as the introduction of the ``__duckarray__``
protocol in NEP 30 [3]_, and the introduction of an overriding mechanism called
``uarray`` by NEP 31 [4]_.

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

1. The object will have no use in the downstream library's implementation; and
2. Simplifies adoption of array creation by those libraries already opting-in
   to implement the ``__array_function__`` protocol, thus removing the
   requirement to explicitly opt-in for all array creation functions.

Downstream libraries thus shall _NOT_ include the ``like=`` keyword to their
array creation APIs, which is a NumPy-exclusive keyword.

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

The first version of this proposal suggested the C implementation above as one
viable solution. However, due to the downsides pointed above we have decided to
implement that entirely in C. Please refer to [implementation]_ for details.

Alternatives
------------

Recently a new protocol to replace ``__array_function__`` entirely was proposed
by NEP 37 [6]_, which would require considerable rework by downstream libraries
that adopt ``__array_function__`` already, because of that we still believe the
``like=`` argument is beneficial for NumPy and downstream libraries. However,
that proposal wouldn't necessarily be considered a direct alternative to the
present NEP, as it would replace NEP 18 entirely, on which this builds upon.
Discussion on details about this new proposal and why that would require rework
by downstream libraries is beyond the scopy of the present proposal.

Discussion
----------

.. [implementation] `Implementation's pull request on GitHub <https://github.com/numpy/numpy/pull/16935>`_
.. [discussion] `Further discussion on implementation and the NEP's content <https://mail.python.org/pipermail/numpy-discussion/2020-August/080919.html>`_

References
----------

.. [1] `NEP 18 - A dispatch mechanism for NumPy's high level array functions <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_.

.. [2] `PEP 3102 — Keyword-Only Arguments <https://www.python.org/dev/peps/pep-3102/>`_.

.. [3] `NEP 30 — Duck Typing for NumPy Arrays - Implementation <https://numpy.org/neps/nep-0030-duck-array-protocol.html>`_.

.. [4] `NEP 31 — Context-local and global overrides of the NumPy API <https://github.com/numpy/numpy/pull/14389>`_.

.. [5] `Array creation routines <https://docs.scipy.org/doc/numpy-1.17.0/reference/routines.array-creation.html>`_.

.. [6] `NEP 37 — A dispatch protocol for NumPy-like modules <https://numpy.org/neps/nep-0037-array-module.html>`_.

Copyright
---------

This document has been placed in the public domain.
