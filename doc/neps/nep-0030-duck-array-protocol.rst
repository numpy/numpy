======================================================
NEP 30 — Duck Typing for NumPy Arrays - Implementation
======================================================

:Author: Peter Andreas Entschev <pentschev@nvidia.com>
:Author: Stephan Hoyer <shoyer@google.com>
:Status: Draft
:Type: Standards Track
:Created: 2019-07-31
:Updated: 2019-07-31
:Resolution:

Abstract
--------

We propose the ``__duckarray__`` protocol, following the high-level overview
described in NEP 22, allowing downstream libraries to return arrays of their
defined types, in contrast to ``np.asarray``, that coerces those ``array_like``
objects to NumPy arrays.

Detailed description
--------------------

NumPy's API, including array definitions, is implemented and mimicked in
countless other projects. By definition, many of those arrays are fairly
similar in how they operate to the NumPy standard. The introduction of
``__array_function__`` allowed dispathing of functions implemented by several
of these projects directly via NumPy's API. This introduces a new requirement,
returning the NumPy-like array itself, rather than forcing a coercion into a
pure NumPy array.

For the purpose above, NEP 22 introduced the concept of duck typing to NumPy
arrays. The suggested solution described in the NEP allows libraries to avoid
coercion of a NumPy-like array to a pure NumPy array where necessary, while
still allowing that NumPy-like array libraries that do not wish to implement
the protocol to coerce arrays to a pure Numpy array via ``np.asarray``.

Usage Guidance
~~~~~~~~~~~~~~

Code that uses np.duckarray is meant for supporting other ndarray-like objects
that "follow the NumPy API". That is an ill-defined concept at the moment --
every known library implements the NumPy API only partly, and many deviate
intentionally in at least some minor ways. This cannot be easily remedied, so
for users of ``__duckarray__`` we recommend the following strategy: check if the
NumPy functionality used by the code that follows your use of ``__duckarray__``
is present in Dask, CuPy and Sparse. If so, it's reasonable to expect any duck
array to work here. If not, we suggest you indicate in your docstring what kinds
of duck arrays are accepted, or what properties they need to have.

To exemplify the usage of duck arrays, suppose one wants to take the ``mean()``
of an array-like object ``arr``. Using NumPy to achieve that, one could write
``np.asarray(arr).mean()`` to achieve the intended result. However, libraries
may expect ``arr`` to be a NumPy-like array, and at the same time, the array may
or may not be an object compliant to the NumPy API (either in full or partially)
such as a CuPy, Sparse or a Dask array. In the case where ``arr`` is already an
object compliant to the NumPy API, we would simply return it (and prevent it
from being coerced into a pure NumPy array), otherwise, it would then be coerced
into a NumPy array.

Implementation
--------------

The implementation idea is fairly straightforward, requiring a new function
``duckarray`` to be introduced in NumPy, and a new method ``__duckarray__`` in
NumPy-like array classes. The new ``__duckarray__`` method shall return the
downstream array-like object itself, such as the ``self`` object. If appropriate,
an ``__array__`` method may be implemented that returns a NumPy array or possibly
raise a ``TypeError`` with a helpful message.

The new NumPy ``duckarray`` function can be implemented as follows:

.. code:: python

    def duckarray(array_like):
        if hasattr(array_like, '__duckarray__'):
            return array_like.__duckarray__()
        return np.asarray(array_like)

Example for a project implementing NumPy-like arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now consider a library that implements a NumPy-compatible array class called
``NumPyLikeArray``, this class shall implement the methods described above, and
a complete implementation would look like the following:

.. code:: python

    class NumPyLikeArray:
        def __duckarray__(self):
            return self

        def __array__(self):
            return TypeError("NumPyLikeArray can not be converted to a numpy array. "
                             "You may want to use np.duckarray.")

The implementation above exemplifies the simplest case, but the overall idea
is that libraries will implement a ``__duckarray__`` method that returns the
original object, and an ``__array__`` method that either creates and returns an
appropriate NumPy array, or raises a``TypeError`` to prevent unintentional use
as an object in a NumPy array (if ``np.asarray`` is called on an arbitrary
object that does not implement ``__array__``, it will create a NumPy array
scalar).

In case of existing libraries that don't already implement ``__array__`` but
would like to use duck array typing, it is advised that they introduce
both ``__array__`` and``__duckarray__`` methods.

Usage
-----

An example of how the ``__duckarray__`` protocol could be used to write a
``stack`` function based on ``concatenate``, and its produced outcome, can be
seen below. The example here was chosen not only to demonstrate the usage of
the ``duckarray`` function, but also to demonstrate its dependency on the NumPy
API, demonstrated by checks on the array's ``shape`` attribute. Note that the
example is merely a simplified version of NumPy's actually implementation of
``stack`` working on the first axis, and it is assumed that Dask has implemented
the ``__duckarray__`` method.

.. code:: python

    def duckarray_stack(arrays):
        arrays = [np.duckarray(arr) for arr in arrays]

        shapes = {arr.shape for arr in arrays}
        if len(shapes) != 1:
            raise ValueError('all input arrays must have the same shape')

        expanded_arrays = [arr[np.newaxis, ...] for arr in arrays]
        return np.concatenate(expanded_arrays, axis=0)

    dask_arr = dask.array.arange(10)
    np_arr = np.arange(10)
    np_like = list(range(10))

    duckarray_stack((dask_arr, dask_arr))   # Returns dask.array
    duckarray_stack((dask_arr, np_arr))     # Returns dask.array
    duckarray_stack((dask_arr, np_like))    # Returns dask.array

In contrast, using only ``np.asarray`` (at the time of writing of this NEP, this
is the usual method employed by library developers to ensure arrays are
NumPy-like) has a different outcome:

.. code:: python

    def asarray_stack(arrays):
        arrays = [np.asanyarray(arr) for arr in arrays]

        # The remaining implementation is the same as that of
        # ``duckarray_stack`` above

    asarray_stack((dask_arr, dask_arr))     # Returns np.ndarray
    asarray_stack((dask_arr, np_arr))       # Returns np.ndarray
    asarray_stack((dask_arr, np_like))      # Returns np.ndarray

Backward compatibility
----------------------

This proposal does not raise any backward compatibility issues within NumPy,
given that it only introduces a new function. However, downstream libraries
that opt to introduce the ``__duckarray__`` protocol may choose to remove the
ability of coercing arrays back to a NumPy array via ``np.array`` or
``np.asarray`` functions, preventing unintended effects of coercion of such
arrays back to a pure NumPy array (as some libraries already do, such as CuPy
and Sparse), but still leaving libraries not implementing the protocol with the
choice of utilizing ``np.duckarray`` to promote ``array_like`` objects to pure
NumPy arrays.

Previous proposals and discussion
---------------------------------

The duck typing protocol proposed here was described in a high level in
`NEP 22 <https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html>`_.

Additionally, longer discussions about the protocol and related proposals
took place in
`numpy/numpy #13831 <https://github.com/numpy/numpy/issues/13831>`_

Copyright
---------

This document has been placed in the public domain.
