============================================================
NEP 28 — Context-local and global overrides of the NumPy API
============================================================

:Author: Hameer Abbasi <einstein.edison@gmail.com>
:Status: Draft
:Type: Standards Track
:Created: 2019-07-31


Abstract
--------

This NEP proposes to make all of NumPy's public API overridable via a backend
mechanism, using a library called `uarray <https://uarray.readthedocs.io>`_.

``uarray`` provides global and context-local overrides, as well as a dispatch
mechanism similar to NEP-18 `[2]`_. This NEP proposes to supercede NEP-18,
and is intended as a comprehensive resolution to NEP-22 `[3]`_.

Motivation and Scope
--------------------

The motivation behind this library is manifold: First, there have been several attempts to allow
dispatch of parts of the NumPy API, including (most prominently), the ``__array_ufunc__`` protocol
in NEP-13 `[4]`_, and the ``__array_function__`` protocol in NEP-18 `[2]`_, but this has shown the
need for further protocols to be developed, including a protocol for coercion. `[5]`_. The reasons
these overrides are needed have been extensively discussed in the references, and this NEP will not
attempt to go into the details of why these are needed.

This NEP takes a more holistic approach: It assumes that there are parts of the API that need to be
overridable, and that these will grow over time. It provides a general framework and a mechanism to
avoid a design of a new protocol each time this is required.

The second is to ease the creation of new duck-arrays, 

The second is the existence of actual, third party dtype packages, and
their desire to blend into the NumPy ecosystem. `[6]`_. This is a separate
issue compared to the C-level dtype redesign proposed in `[7]`_, it's about
allowing third-party dtype implementations to work with NumPy, much like third-party array
implementations.

This NEP proposes the following:

* A path to adopting ``uarray`` `[1]`_ as the de-facto override mechanism the NumPy API.
* A path to the deprecation of ``__array_function__`` `[2]`_ and ``__array_ufunc__`` `[3]`_,
  and thus, NEPs 18 and 13 being superceded.
* The exact specifics of how to use ``uarray`` to override the NumPy API.


Detailed description
--------------------

This section will not attempt to explain the specifics or the mechanism of ``uarray``,
that is explained in the ``uarray`` documentation. `[1]`_ However, the NumPy community
will have input into the design of ``uarray``, and any backward-incompatible changes
will be discussed on the mailing list.

The first goal of this NEP is as follows: To complete an overridable version of NumPy,
called ``unumpy`` `[8]`_, the implementation of which is already underway. Again, ``unumpy``
will not be explained here, the reader should refer to its documentation for this purpose.

The only change this NEP proposes at its acceptance, is to make ``unumpy`` the officially recommended
way to override NumPy. ``unumpy`` will remain a separate repository/package, and will be developed
primarily with the input of duck-array authors and secondarily, custom dtype authors, via the usual
GitHub workflow. There are a few reasons for this:
 
* Faster iteration in the case of bugs or issues.
* Faster design changes, in the case of needed functionality.
* Lower likelihood to be stuck with a bad design decision.
* The user and library author opt-in to the override process,
  rather than breakages happening when it is least expected.
  In simple terms, bugs in ``unumpy`` mean that ``numpy`` remains
  unaffected.
* The upgrade pathway to NumPy 2.0 becomes simpler, requiring just
  a backend change, and allowing both to exist side-by-side.

Once maturity is achieved, ``unumpy`` be moved into the NumPy organization,
and NumPy will become the reference implementation for ``unumpy``.

Related Work
------------

Previous override mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* NEP-18, the ``__array_function__`` protocol. `[2]`_
* NEP-13, the ``__array_ufunc__`` protocol. `[3]`_

Existing NumPy-like array implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Dask: https://dask.org/
* CuPy: https://cupy.chainer.org/
* PyData/Sparse: https://sparse.pydata.org/
* Xnd: https://xnd.readthedocs.io/

Existing and potential consumers of alternative arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Dask: https://dask.org/
* scikit-learn: https://scikit-learn.org/
* XArray: https://xarray.pydata.org/
* TensorLy: http://tensorly.org/

Implementation
--------------

The implementation of this NEP will require the following steps:

* Implementation of ``uarray`` multimethods corresponding to the
  NumPy API, including classes for overriding ``dtype``, ``ufunc``
  and ``array`` objects, in the ``unumpy`` repository.
* Moving backends from ``unumpy`` into the respective array libraries.

Backward compatibility
----------------------

This NEP proposed a deprecation path for ``__array_function__`` and ``__array_ufunc__``.
After the maturity of the ``unumpy`` project, as decided by the status of this NEP,
``__array_function__`` and ``__array_ufunc__`` will be deprecated and subsequently
removed.


Alternatives
------------

The current alternative to this problem, already implemented, is a
combination of NEP-18 and NEP-13.


Discussion
----------

* The discussion section of NEP-18: https://numpy.org/neps/nep-0018-array-function-protocol.html#discussion
* NEP-22: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
* Dask issue #4462: https://github.com/dask/dask/issues/4462
* PR #13046: https://github.com/numpy/numpy/pull/13046
* Dask issue #4883: https://github.com/dask/dask/issues/4883
* Issue #13831: https://github.com/numpy/numpy/issues/13831


References and Footnotes
------------------------

.. _[1]:

[1] uarray, A general dispatch mechanism for Python: https://uarray.readthedocs.io

.. _[2]:

[2] NEP 18 — A dispatch mechanism for NumPy’s high level array functions: https://numpy.org/neps/nep-0018-array-function-protocol.html

.. _[3]:

[3] NEP 22 — Duck typing for NumPy arrays – high level overview: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html

.. _[4]:

[4] NEP 13 — A Mechanism for Overriding Ufuncs: https://numpy.org/neps/nep-0013-ufunc-overrides.html

.. _[5]:

[5] Reply to Adding to the non-dispatched implementation of NumPy methods: http://numpy-discussion.10968.n7.nabble.com/Adding-to-the-non-dispatched-implementation-of-NumPy-methods-tp46816p46874.html

.. _[6]:

[6] Custom Dtype/Units discussion: http://numpy-discussion.10968.n7.nabble.com/Custom-Dtype-Units-discussion-td43262.html

.. _[7]:

[7] The epic dtype cleanup plan: https://github.com/numpy/numpy/issues/2899

.. _[8]:

[8] unumpy: NumPy, but implementation-independent: https://unumpy.readthedocs.io

Copyright
---------

This document has been placed in the public domain.
