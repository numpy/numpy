============================================================
NEP 31 — Context-local and global overrides of the NumPy API
============================================================

:Author: Hameer Abbasi <habbasi@quansight.com>
:Author: Ralf Gommers <rgommers@quansight.com>
:Author: Peter Bell <peterbell10@live.co.uk>
:Status: Draft
:Type: Standards Track
:Created: 2019-08-22


Abstract
--------

This NEP proposes to make all of NumPy's public API overridable via a backend
mechanism, using a library called ``uarray`` `[1]`_

``uarray`` provides global and context-local overrides, as well as a dispatch
mechanism similar to NEP-18 `[2]`_. First experiences with ``__array_function__``
show that it is necessary to be able to override NumPy functions that
*do not take an array-like argument*, and hence aren't overridable via
``__array_function__``. The most pressing need is array creation and coercion
functions - see e.g. NEP-30 `[9]`_.

This NEP proposes to allow, in an opt-in fashion, overriding any part of the NumPy API.
It is intended as a comprehensive resolution to NEP-22 `[3]`_, and obviates the need to
add an ever-growing list of new protocols for each new type of function or object that needs
to become overridable.

Motivation and Scope
--------------------

The motivation behind ``uarray`` is manyfold: First, there have been several attempts to allow
dispatch of parts of the NumPy API, including (most prominently), the ``__array_ufunc__`` protocol
in NEP-13 `[4]`_, and the ``__array_function__`` protocol in NEP-18 `[2]`_, but this has shown the
need for further protocols to be developed, including a protocol for coercion (see `[5]`_). The reasons
these overrides are needed have been extensively discussed in the references, and this NEP will not
attempt to go into the details of why these are needed. Another pain point requiring yet another
protocol is the duck-array protocol (see `[9]`_).

This NEP takes a more holistic approach: It assumes that there are parts of the API that need to be
overridable, and that these will grow over time. It provides a general framework and a mechanism to
avoid a design of a new protocol each time this is required.

This NEP proposes the following: That ``unumpy`` `[8]`_  becomes the recommended override mechanism
for the parts of the NumPy API not yet covered by ``__array_function__`` or ``__array_ufunc__``,
and that ``uarray`` is vendored into a new namespace within NumPy to give users and downstream dependencies
access to these overrides.  This vendoring mechanism is similar to what SciPy decided to do for
making ``scipy.fft`` overridable (see `[10]`_).


Detailed description
--------------------

**Note:** *This section will not attempt to explain the specifics or the mechanism of ``uarray``,
that is explained in the ``uarray`` documentation.* `[1]`_ *However, the NumPy community
will have input into the design of ``uarray``, and any backward-incompatible changes
will be discussed on the mailing list.*

The way we propose the overrides will be used by end users is::

    import numpy.overridable as np
    with np.set_backend(backend):
        x = np.asarray(my_array, dtype=dtype)

And a library that implements a NumPy-like API will use it in the following manner (as an example)::

    import numpy.overridable as np
    _ua_implementations = {}

    __ua_domain__ = "numpy"

    def __ua_function__(func, args, kwargs):
        return _ua_implementations[func](*args, **kwargs)

    def implements(ua_func):
        def inner(func):
            _ua_implementations[ua_func] = func
            return func

        return inner

    @implements(np.asarray)
    def asarray(a, dtype=None, order=None):
        # Code here
        # Either this method or __ua_convert__ must
        # return NotImplemented for unsupported types,
        # Or they shouldn't be marked as dispatchable.

    # Provides a default implementation for ones and zeros.
    @implements(np.full)
    def full(shape, fill_value, dtype=None, order='C'):
        # Code here

The only change this NEP proposes at its acceptance, is to make ``unumpy`` the officially recommended
way to override NumPy. ``unumpy`` will remain a separate repository/package (which we propose to vendor
to avoid a hard dependency, and use the separate ``unumpy`` package only if it is installed)
rather than depend on for the time being), and will be developed
primarily with the input of duck-array authors and secondarily, custom dtype authors, via the usual
GitHub workflow. There are a few reasons for this:

* Faster iteration in the case of bugs or issues.
* Faster design changes, in the case of needed functionality.
* ``unumpy`` will work with older versions of NumPy as well.
* The user and library author opt-in to the override process,
  rather than breakages happening when it is least expected.
  In simple terms, bugs in ``unumpy`` mean that ``numpy`` remains
  unaffected.

Advantanges of ``unumpy`` over other solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``unumpy`` offers a number of advantanges over the approach of defining a new protocol for every
problem encountered: Whenever there is something requiring an override, ``unumpy`` will be able to
offer a unified API with very minor changes. For example:

* ``ufunc`` objects can be overridden via their ``__call__``, ``reduce`` and other methods.
* ``dtype`` objects can be overridden via the dispatch/backend mechanism, going as far as to allow
  ``np.float32`` et. al. to be overridden by overriding ``__get__``.
* Other functions can be overridden in a similar fashion.
* ``np.asduckarray`` goes away, and becomes ``np.asarray`` with a backend set.
* The same holds for array creation functions such as ``np.zeros``, ``np.empty`` and so on.

This also holds for the future: Making something overridable would require only minor changes to ``unumpy``.

Another promise ``unumpy`` holds is one of default implementations. Default implementations can be provided for
any multimethod, in terms of others. This allows one to override a large part of the NumPy API by defining
only a small part of it. This is to ease the creation of new duck-arrays, by providing default implementations of many
functions that can be easily expressed in terms of others, as well as a repository of utility functions
that help in the implementation of duck-arrays that most duck-arrays would require.

The last benefit is a clear way to coerce to a given backend, and a protocol for coercing not only arrays,
but also ``dtype`` objects and ``ufunc`` objects with similar ones from other libraries. This is due to the existence of
actual, third party dtype packages, and their desire to blend into the NumPy ecosystem (see `[6]`_). This is a separate
issue compared to the C-level dtype redesign proposed in `[7]`_, it's about allowing third-party dtype implementations to
work with NumPy, much like third-party array implementations.

Mixing NumPy and ``unumpy`` in the same file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Normally, one would only want to import only one of ``unumpy`` or ``numpy``, you would import it as ``np`` for
familiarity. However, there may be situations where one wishes to mix NumPy and the overrides, and there are
a few ways to do this, depending on the user's style::

    import numpy.overridable as unumpy
    import numpy as np

or::

    import numpy as np

    # Use unumpy via np.overridable

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
* Astropy's Quantity: https://docs.astropy.org/en/stable/units/

Existing and potential consumers of alternative arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Dask: https://dask.org/
* scikit-learn: https://scikit-learn.org/
* Xarray: https://xarray.pydata.org/
* TensorLy: http://tensorly.org/

Existing alternate dtype implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ndtypes``: https://ndtypes.readthedocs.io/en/latest/
* Datashape: https://datashape.readthedocs.io
* Plum: https://plum-py.readthedocs.io/

Implementation
--------------

The implementation of this NEP will require the following steps:

* Implementation of ``uarray`` multimethods corresponding to the
  NumPy API, including classes for overriding ``dtype``, ``ufunc``
  and ``array`` objects, in the ``unumpy`` repository.
* Moving backends from ``unumpy`` into the respective array libraries.

Backward compatibility
----------------------

There are no backward incompatible changes proposed in this NEP.


Alternatives
------------

The current alternative to this problem is NEP-30 plus adding more protocols
(not yet specified) in addition to it.  Even then, some parts of the NumPy
API will remain non-overridable, so it's a partial alternative.

The main alternative to vendoring ``unumpy`` is to simply move it into NumPy
completely and not distribute it as a separate package. This would also achieve
the proposed goals, however we prefer to keep it a separate package for now,
for reasons already stated above.


Discussion
----------

* The discussion section of NEP-18: https://numpy.org/neps/nep-0018-array-function-protocol.html#discussion
* NEP-22: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
* Dask issue #4462: https://github.com/dask/dask/issues/4462
* PR #13046: https://github.com/numpy/numpy/pull/13046
* Dask issue #4883: https://github.com/dask/dask/issues/4883
* Issue #13831: https://github.com/numpy/numpy/issues/13831
* Discussion PR 1: https://github.com/hameerabbasi/numpy/pull/3
* Discussion PR 2: https://github.com/hameerabbasi/numpy/pull/4


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

.. _[9]:

[9] NEP 30 — Duck Typing for NumPy Arrays - Implementation: https://www.numpy.org/neps/nep-0030-duck-array-protocol.html

.. _[10]:

[10] http://scipy.github.io/devdocs/fft.html#backend-control


Copyright
---------

This document has been placed in the public domain.
