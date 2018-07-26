=============
NumPy Roadmap
=============

This is a live snapshot of tasks and features we will be investing resources
in. It may be used to encourage and inspire developers and to search for
funding.

Interoperability protocols & duck typing
----------------------------------------

- `__array_function__`

  See `NEP 18`_ and a sample implementation_

- Array Duck-Typing

  `NEP 22`_    `np.asduckarray()`

- Mixins like `NDArrayOperatorsMixin`:

  - for mutable arrays
  - for reduction methods implemented as ufuncs

Better dtypes
-------------

- Easier custom dtypes
  - Simplify and/or wrap the current C-API
  - More consistent support for dtype metadata
  - Support for writing a dtype in Python
- New string dtype(s):
  - Encoded strings with fixed-width storage (utf8, latin1, ...) and/or
  - Variable length strings (could share implementation with dtype=object, but are explicitly type-checked)
  - One of these should probably be the default for text data. The current behavior on Python 3 is neither efficient nor user friendly.
- `np.int` should not be platform dependent
- better coercion for string + number

Random number generation policy & rewrite
-----------------------------------------

`NEP 19`_ and a `reference implementation`_

Indexing
--------

vindex/oindex `NEP 21`_

Infrastructure
--------------

NumPy is much more than just the code base itself, we also maintain
docs, CI, benchmarks, etc.

- Rewrite numpy.org
- Benchmarking: improve the extent of the existing suite, and run & render
  the results as part of the docs or website.

  - Hardware: find a machine that can reliably run serial benchmarks
  - ASV produces graphs, could we set up a site? Currently at
    https://pv.github.io/numpy-bench/, should that become a community resource?

Functionality outside core
--------------------------

Some things inside NumPy do not actually match the `Scope of NumPy`.

- A backend system for `numpy.fft` (so that e.g. `fft-mkl` doesn't need to monkeypatch numpy)

- Rewrite masked arrays to not be a ndarray subclass -- maybe in a separate project?
- MaskedArray as a duck-array type, and/or
- dtypes that support missing values

- Write a strategy on how to deal with overlap between numpy and scipy for `linalg` and `fft` (and implement it).

- Deprecate `np.matrix`

Continuous Integration
----------------------

We depend on CI to discover problems as we continue to develop NumPy before the
code reaches downstream users.

- CI for more exotic platforms (e.g. ARM is now available from
  http://www.shippable.com/, but it is not free).
- Multi-package testing
- Add an official channel for numpy dev builds for CI usage by other projects so
  they may confirm new builds do not break their package.

Typing
------

Python type annotation syntax should support ndarrays and dtypes.

- Type annotations for NumPy: github.com/numpy/numpy-stubs
- Support for typing shape and dtype in multi-dimensional arrays in Python more generally

NumPy scalars
-------------

Numpy has both scalars and zero-dimensional arrays.

- The current implementation adds a large maintenance burden -- can we remove
  scalars and/or simplify it internally?
- Zero dimensional arrays get converted into scalars by most NumPy
  functions (i.e., output of `np.sin(x)` depends on whether `x` is
  zero-dimensional or not).  This inconsistency should be addressed,
  so that one could, e.g., write sane type annotations.

.. _`NEP 19`: https://www.numpy.org/neps/nep-0019-rng-policy.html
.. _`NEP 22`: http://www.numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
.. _`NEP 18`: https://www.numpy.org/neps/nep-0018-array-function-protocol.html
.. _implementation: https://gist.github.com/shoyer/1f0a308a06cd96df20879a1ddb8f0006
.. _`reference implementation`: https://github.com/bashtage/randomgen
.. _`NEP 21`: https://www.numpy.org/neps/nep-0021-advanced-indexing.html
