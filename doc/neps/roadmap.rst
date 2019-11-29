=============
NumPy Roadmap
=============

This is a live snapshot of tasks and features we will be investing resources
in. It may be used to encourage and inspire developers and to search for
funding.

Interoperability
----------------

We aim to make it easier to interoperate with NumPy. There are many NumPy-like
packages that add interesting new capabilities to the Python ecosystem, as well
as many libraries that extend NumPy's model in various ways.  Work in NumPy to
facilitate interoperability with all such packages, and the code that uses them,
may include (among other things) interoperability protocols, better duck typing
support and ndarray subclass handling.

- The ``__array_function__`` protocol is currently experimental and needs to be
  matured. See `NEP 18`_ for details.
- New protocols for overriding other functionality in NumPy may be needed.
- Array duck typing, or handling "duck arrays", needs improvements.  See
  `NEP 22`_ for details.

Extensibility
-------------

We aim to make it much easier to extend NumPy. The primary topic here is to
improve the dtype system.

- Easier custom dtypes:

  - Simplify and/or wrap the current C-API
  - More consistent support for dtype metadata
  - Support for writing a dtype in Python

- New string dtype(s):

  - Encoded strings with fixed-width storage (utf8, latin1, ...) and/or
  - Variable length strings (could share implementation with dtype=object,
    but are explicitly type-checked)
  - One of these should probably be the default for text data. The current
    behavior on Python 3 is neither efficient nor user friendly.

- ``np.dtype(int)`` should not be platform dependent
- Better coercion for string + number

Performance
-----------

We want to further improve NumPy's performance, through:

- Better use of SIMD instructions, also on platforms other than x86.
- Reducing ufunc overhead.
- Optimizations in individual functions.

Furthermore we would like to improve the benchmarking system, in terms of coverage,
easy of use, and publication of the results (now
`here <https://pv.github.io/numpy-bench>`__) as part of the docs or website.

Website and documentation
-------------------------

Our website (https://numpy.org) is in very poor shape and needs to be rewritten
completely.

The NumPy `documentation <https://www.numpy.org/devdocs/user/index.html>`__ is
of varying quality - in particular the User Guide needs major improvements.

Random number generation policy & rewrite
-----------------------------------------

A new random number generation framework with higher performance generators is
close to completion, see `NEP 19`_ and `PR 13163`_.

Indexing
--------

We intend to add new indexing modes for "vectorized indexing" and "outer indexing",
see `NEP 21`_.

Continuous Integration
----------------------

We depend on CI to discover problems as we continue to develop NumPy before the
code reaches downstream users.

- CI for more exotic platforms (if available as a service).
- Multi-package testing
- Add an official channel for numpy dev builds for CI usage by other projects so
  they may confirm new builds do not break their package.

Other functionality
-------------------

- ``MaskedArray`` needs to be improved, ideas include:

  - Rewrite masked arrays to not be a ndarray subclass -- maybe in a separate project?
  - MaskedArray as a duck-array type, and/or
  - dtypes that support missing values

- A backend system for ``numpy.fft`` (so that e.g. ``fft-mkl`` doesn't need to monkeypatch numpy)
- Write a strategy on how to deal with overlap between NumPy and SciPy for ``linalg``
  and ``fft`` (and implement it).
- Deprecate ``np.matrix`` (very slowly)


.. _`NEP 19`: https://www.numpy.org/neps/nep-0019-rng-policy.html
.. _`NEP 22`: http://www.numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
.. _`NEP 18`: https://www.numpy.org/neps/nep-0018-array-function-protocol.html
.. _implementation: https://gist.github.com/shoyer/1f0a308a06cd96df20879a1ddb8f0006
.. _`reference implementation`: https://github.com/bashtage/randomgen
.. _`NEP 21`: https://www.numpy.org/neps/nep-0021-advanced-indexing.html
.. _`PR 13163`: https://github.com/numpy/numpy/pull/13163
