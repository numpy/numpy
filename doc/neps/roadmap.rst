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

The key goal is: *make it easy for code written for NumPy to also work with
other NumPy-like projects.* This will enable GPU support via, e.g, CuPy or JAX,
distributed array support via Dask, and writing special-purpose arrays (either
from scratch, or as a ``numpy.ndarray`` subclass) that work well with SciPy,
scikit-learn and other such packages.

The ``__array_ufunc__`` and ``__array_function__`` protocols are stable, but
do not cover the whole API.  New protocols for overriding other functionality
in NumPy are needed. Work in this area aims to bring to completion one or more
of the following proposals:

- :ref:`NEP30`
- :ref:`NEP31`
- :ref:`NEP35`
- :ref:`NEP37`

In addition we aim to provide ways to make it easier for other libraries to
implement a NumPy-compatible API. This may include defining consistent subsets
of the API, as discussed in `this section of NEP 37
<https://numpy.org/neps/nep-0037-array-module.html#requesting-restricted-subsets-of-numpy-s-api>`__.


Performance
-----------

Improvements to NumPy's performance are important to many users. We have
focused this effort on Universal SIMD (see :ref:`NEP38`) intrinsics which
provide nice improvements across various hardware platforms via an abstraction
layer.  The infrastructure is in place, and we welcome follow-on PRs to add
SIMD support across all relevant NumPy functions.

Other performance improvement ideas include:

- A better story around parallel execution.
- Optimizations in individual functions.
- Reducing ufunc and ``__array_function__`` overhead.

Furthermore we would like to improve the benchmarking system, in terms of coverage,
easy of use, and publication of the results (now
`here <https://pv.github.io/numpy-bench>`__) as part of the docs or website.


Documentation and website
-------------------------

The NumPy `documentation <https://www.numpy.org/devdocs>`__ is of varying
quality. The API documentation is in good shape; tutorials and high-level
documentation on many topics are missing or outdated. See :ref:`NEP44` for
planned improvements. Adding more tutorials is underway in the
`numpy-tutorials repo <https://github.com/numpy/numpy-tutorials>`__.

Our website (https://numpy.org) was completely redesigned recently. We aim to
further improve it by adding translations, more case studies and other
high-level content, and more (see `this tracking issue <https://github.com/numpy/numpy.org/issues/266>`__).


Extensibility
-------------

We aim to make it much easier to extend NumPy. The primary topic here is to
improve the dtype system - see :ref:`NEP41` and related NEPs linked from it.
Concrete goals for the dtype system rewrite are:

- Easier custom dtypes:

  - Simplify and/or wrap the current C-API
  - More consistent support for dtype metadata
  - Support for writing a dtype in Python

- Allow adding (a) new string dtype(s). This could be encoded strings with
  fixed-width storage (e.g., ``utf8`` or ``latin1``), and/or a variable length
  string dtype. The latter could share an implementation with ``dtype=object``,
  but be explicitly type-checked.
  One of these should probably be the default for text data. The current
  string dtype support is neither efficient nor user friendly.


User experience
---------------

Type annotations
````````````````
NumPy 1.20 adds type annotations for most NumPy functionality, so users can use
tools like `mypy`_ to type check their code and IDEs can improve their support
for NumPy. Improving those type annotations, for example to support annotating
array shapes and dtypes, is ongoing.

Platform support
````````````````
We aim to increase our support for different hardware architectures. This
includes adding CI coverage when CI services are available, providing wheels on
PyPI for POWER8/9 (``ppc64le``), providing better build and install
documentation, and resolving build issues on other platforms like AIX.


Maintenance
-----------

- ``MaskedArray`` needs to be improved, ideas include:

  - Rewrite masked arrays to not be a ndarray subclass -- maybe in a separate project?
  - MaskedArray as a duck-array type, and/or
  - dtypes that support missing values

- Fortran integration via ``numpy.f2py`` requires a number of improvements, see
  `this tracking issue <https://github.com/numpy/numpy/issues/14938>`__.
- A backend system for ``numpy.fft`` (so that e.g. ``fft-mkl`` doesn't need to monkeypatch numpy).
- Write a strategy on how to deal with overlap between NumPy and SciPy for ``linalg``.
- Deprecate ``np.matrix`` (very slowly).
- Add new indexing modes for "vectorized indexing" and "outer indexing" (see :ref:`NEP21`).
- Make the polynomial API easier to use.
- Integrate an improved text file loader.
- Ufunc and gufunc improvements, see `gh-8892 <https://github.com/numpy/numpy/issues/8892>`__
  and `gh-11492 <https://github.com/numpy/numpy/issues/11492>`__.


.. _`mypy`: https://mypy.readthedocs.io
