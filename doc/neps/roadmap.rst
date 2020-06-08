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


Performance
-----------

Improvements to NumPy's performance are important to many users. The primary
topic at the moment is better use of SIMD instructions, also on platforms other
than x86 - see :ref:`NEP38`.

Other performance improvement ideas include:

- Reducing ufunc and ``__array_function__`` overhead.
- Optimizations in individual functions.
- A better story around parallel execution.

Furthermore we would like to improve the benchmarking system, in terms of coverage,
easy of use, and publication of the results (now
`here <https://pv.github.io/numpy-bench>`__) as part of the docs or website.


Website and documentation
-------------------------

The NumPy `documentation <https://www.numpy.org/devdocs>`__ is of varying
quality. The API documentation is in good shape; tutorials and high-level
documentation on many topics are missing or outdated. See :ref:`NEP44` for
planned improvements.

Our website (https://numpy.org) was completely redesigned recently. We aim to
further improve it by adding translations, better Hugo-Sphinx integration via a
new Sphinx theme, and more (see `this tracking issue <https://github.com/numpy/numpy.org/issues/266>`__).


User experience
---------------

Type annotations
````````````````
We aim to add type annotations for all NumPy functionality, so users can use
tools like `mypy`_ to type check their code and IDEs can improve their support
for NumPy. The existing type stubs in the `numpy-stubs`_ repository are being
improved and will be moved into the NumPy code base.

Platform support
````````````````
We aim to increase our support for different hardware architectures. This
includes adding CI coverage when CI services are available, providing wheels on
PyPI for ARM64 (``aarch64``) and POWER8/9 (``ppc64le``), providing better
build and install documentation, and resolving build issues on other platforms
like AIX.


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
.. _`numpy-stubs`: https://github.com/numpy/numpy-stubs
