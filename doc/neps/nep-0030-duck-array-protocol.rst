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
to NumPy arrays.

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

Implementation
--------------

The implementation idea is fairly straightforward, requiring a new function
``duckarray`` to be introduced in NumPy, and a new method ``__duckarray__`` in
NumPy-like array classes. The new ``__duckarray__`` method shall return the
downstream array-like object itself, such as the ``self`` object, while the
``__array__`` method returns ``TypeError``.

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
            return TypeError

The implementation above exemplifies the simplest case, but the overall idea
is that libraries will implement a ``__duckarray__`` method that returns the
original object, and ``__array__`` solely for the purpose of raising a
``TypeError``, thus preventing unintentional NumPy-coercion. In case of existing
libraries that don't already implement ``__array__`` but would like to use duck
array typing, it is advised that they they introduce both ``__array__`` and
``__duckarray__`` methods.

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
