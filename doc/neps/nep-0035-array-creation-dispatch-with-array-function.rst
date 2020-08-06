.. _NEP35:

===========================================================
NEP 35 — Array Creation Dispatching With __array_function__
===========================================================

:Author: Peter Andreas Entschev <pentschev@nvidia.com>
:Status: Draft
:Type: Standards Track
:Created: 2019-10-15
:Updated: 2020-08-06
:Resolution:

Abstract
--------

We propose the introduction of a new keyword argument ``like=`` to all array
creation functions to permit dispatching of such functions by the
``__array_function__`` protocol, addressing one of the protocol shortcomings,
as described by NEP-18 [1]_.

Detailed description
--------------------

The introduction of the ``__array_function__`` protocol allowed downstream
library developers to use NumPy as a dispatching API. However, the protocol
did not -- and did not intend to -- address the creation of arrays by downstream
libraries, preventing those libraries from using such important functionality in
that context.

Other NEPs have been written to address parts of that limitation, such as the
introduction of the ``__duckarray__`` protocol in NEP-30 [2]_, and the
introduction of an overriding mechanism called ``uarray`` by NEP-31 [3]_.

The purpose of this NEP is to address that shortcoming in a simple and
straighforward way: introduce a new ``like=`` keyword argument, similar to how
the ``empty_like`` family of functions work. When array creation functions
receive such an argument, they will trigger the ``__array_function__`` protocol,
and call the downstream library's own array creation function implementation.
The ``like=`` argument, as its own name suggests, shall be used solely for the
purpose of identifying where to dispatch.  In contrast to the way
``__array_function__`` has been used so far (the first argument identifies where
to dispatch), and to avoid breaking NumPy's API with regards to array creation,
the new ``like=`` keyword shall be used for the purpose of dispatching.

Usage Guidance
~~~~~~~~~~~~~~

The new ``like=`` keyword is solely intended to identify the downstream library
where to dispatch and the object is used only as reference, meaning that no
modifications, copies or processing will be performed on that object.

We expect that this functionality will be mostly useful to library developers,
allowing them to create new arrays for internal usage based on arrays passed
by the user, preventing unnecessary creation of NumPy arrays that will
ultimately lead to an additional conversion into a downstream array type.

Support for Python 2.7 has been dropped since NumPy 1.17, therefore we should
make use of the keyword-only argument standard described in PEP-3102 [4]_ to
implement the ``like=``, thus preventing it from being passed by position.

Implementation
--------------

The implementation requires introducing a new ``like=`` keyword to all existing
array creation functions of NumPy. As examples of functions that would add this
new argument (but not limited to) we can cite those taking array-like objects
such as ``array`` and ``asarray``, functions that create arrays based on
numerical ranges such as ``range`` and ``linspace``, as well as the ``empty``
family of functions, even though that may be redundant, since there exists
already specializations for those with the naming format ``empty_like``. As of
the writing of this NEP, a complete list of array creation functions can be
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

Alternatively for C functions, the implementation of ``like=`` could be moved
into the C implementation itself. This is not the primary suggestion here due
to its inherent complexity which would be difficult too long to describe in its
entirety here, and too tedious for the reader. However, we leave that as an
option open for discussion.

Usage
-----

The purpose of this NEP is to keep things simple. Similarly, we can exemplify
the usage of ``like=`` in a simple way. Imagine you have an array of ones
created by a downstream library, such as CuPy. What you need now is a new array
that can be created using the NumPy API, but that will in fact be created by
the downstream library, a simple way to achieve that is shown below.

.. code:: python

    x = cupy.ones(2)
    np.array([1, 3, 5], like=x)     # Returns cupy.ndarray

As a second example, we could also create an array of evenly spaced numbers
using a Dask identity matrix as reference:

.. code:: python

    x = dask.array.eye(3)
    np.linspace(0, 2, like=x)       # Returns dask.array


Compatibility
-------------

This proposal does not raise any backward compatibility issues within NumPy,
given that it only introduces a new keyword argument to existing array creation
functions.

Downstream libraries will benefit from the ``like=`` argument automatically,
that is, without any explicit changes in their codebase. The only requirement
is that they already implement the ``__array_function__`` protocol, as
described by NEP-18 [2]_.

References and Footnotes
------------------------

.. [1] `NEP-18 - A dispatch mechanism for NumPy's high level array functions <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_.

.. [2] `NEP 30 — Duck Typing for NumPy Arrays - Implementation <https://numpy.org/neps/nep-0030-duck-array-protocol.html>`_.

.. [3] `NEP 31 — Context-local and global overrides of the NumPy API <https://github.com/numpy/numpy/pull/14389>`_.

.. [4] `PEP 3102 — Keyword-Only Arguments <https://www.python.org/dev/peps/pep-3102/>`_.

.. [5] `Array creation routines <https://docs.scipy.org/doc/numpy-1.17.0/reference/routines.array-creation.html>`_.

Copyright
---------

This document has been placed in the public domain.
