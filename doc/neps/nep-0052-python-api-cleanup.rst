.. _NEP52:

=========================================
NEP 52 — Python API cleanup for NumPy 2.0
=========================================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Author: Stéfan van der Walt <stefanv@berkeley.edu>
:Status: Draft
:Type: Standards Track
:Created: 2023-03-28
:Resolution:


Abstract
--------

We propose to clean up NumPy's Python API for the NumPy 2.0 release.
This includes a more clearly defined split between what is public and what is
private, reducing the size of the main namespace by removing many aliases and
other functionality which has better alternatives.


Motivation and Scope
--------------------

NumPy has a very large, and not very well-defined, API surface:

.. code:: python

   >>> objects_in_api = [s for s in dir(np) if not s.startswith('_')]
   >>> len(objects_in_api)
   562
   >>> modules = [s for s in objects_in_api if inspect.ismodule(eval(f'np.{s}'))]
   >>> modules
   ['char', 'compat', 'ctypeslib', 'emath', 'fft', 'lib', 'linalg', 'ma', 'math', 'polynomial', 'random', 'rec', 'testing', 'version']
   >>> len(modules)
   14

The above doesn't even include items that look public but are not included in
``__dir__``. A particularly unhelpful module in that category is ``np.core``,
which is private but in practice heavily used. For a full overview of what's
consider public, private or a bit in between, see
`numpy/tests/test_public_api.py <https://github.com/numpy/numpy/blob/main/numpy/tests/test_public_api.py>`__.

The size and lack of clear boundaries of this API surface has costs:

- For users it is difficult to choose between similar functions or know what
  the recommended way of implementing something is. Looking for functions with
  tab completion in IPython or a notebook or IDE is a challenge (e.g., type
  ``np.<TAB>`` and look at the first six items offered - there's two ufuncs
  (``abs``, ``add``), one alias (``absolute``), and three functions that are
  not useful to them (``add_docstring``, ``add_newdoc``, ``add_newdoc_ufunc``).
  As a result, the learning curve for NumPy is steeper than it has to be.
- For maintainers of other libraries which aim to implement a NumPy-compatible
  API - Dask, CuPy, JAX, PyTorch, TensorFlow, cuNumeric, etc. - or that aim to
  support NumPy in some other way - Numba, Pythran - there is an implementation
  cost to every extra object in the namespace. In practice, no other library
  has full support for everything in NumPy, and making decisions about niche or
  legacy objects in NumPy is a time-consuming effort.
- Teaching the NumPy API to others or making it accessible in a
  learner-oriented GUI has similar costs.

Link discussion about restructuring namespaces! (e.g., find the thread with the
GUI explorer person)

The scope of this NEP includes:

- xxx

Out of scope for this NEP are:

- xxx




Usage and Impact
----------------


A key principle of this API refactor is to ensure that, when code has been
adapted to the changes and is 2.0-compatible, that code then *also* works with
NumPy ``1.2x.x``. This keeps the burden on users and downstream library
maintainers low by not having to carry duplicate code which switches on the
NumPy major version number.


Backward compatibility
----------------------

There is a backwards compatibility impact for users of deprecated or removed
functionality, as well as for users and libraries that were relying on private
NumPy APIs that will move due to the clearer namespacing.

In order to make it easier to adapt to the changes in this NEP, we will:

1. Ensure that NumPy 2.0 will come with guidance for each removed API, pointing
   to the preferred alternative API to use,
2. Provide a script to automate the migration wherever possible. This will be
   similar to ``tools/replace_old_macros.sed`` (which adapts code for a
   previous C API naming scheme change).


Detailed description
--------------------

Cleaning up the main namespace
``````````````````````````````

We expect to reduce the main namespace by a large number of entries - O(100)
probably. Here is a representative set of examples:

- ``np.inf`` and ``np.nan`` have 8 aliases between them, most can be removed.
- A collection of random and undocumented functions (e.g., ``byte_bounds``, ``disp``,
  ``safe_eval``, ``who``) listed in
  `gh-12385 <https://github.com/numpy/numpy/issues/12385>`__
  can be deprecated and removed.
- All ``*sctype`` functions can be deprecated and removed, they (see
  `gh-17325 <https://github.com/numpy/numpy/issues/17325>`__,
  `gh-12334 <https://github.com/numpy/numpy/issues/12334>`__,
  and other issues for ``maximum_sctype`` and related functions).
- Business day functionality can likely be removed (unclear if it needs
  splitting out like was done for ``np.financial``).
- The ``np.compat`` namespace can be removed.
- There are lots of one-off functions that can be deprecated and removed, such as
  ``real_if_close`` (see `gh-11375 <https://github.com/numpy/numpy/issues/11375>`__).
  These can be identified via triaging the issue tracker and/or via going
  through the main namespace manually.

There are new namespaces for warnings/exceptions (``np.exceptions``) and for dtype-related
functionality (``np.types``). NumPy 2.0 is a good opportunity to move things
there from the main namespace.

Functionality that gets a fair amount of usage but has a preferred alternative
may be hidden by not including it in ``__dir__``, rather than deprecating it. A
``.. legacy::`` directory may be used to mark such functionality in the
documentation.

A more comprehensive test will be added to the test suite to ensure that we won't get
any more accidental additions to any namespace - every new entry will need to be
allow-listed.

Cleaning up the submodule structure
```````````````````````````````````

Let's reorganize the API reference guide along main and submodule namespaces,
and only within the main namespace use the current subdivision along
functionality groupings. Also by "mainstream" and special-purpose namespaces.
Details TBD, something like::

    # Regular/recommended user-facing namespaces for general use
    numpy
    numpy.exceptions
    numpy.fft
    numpy.linalg
    numpy.ma
    numpy.polynomial
    numpy.random
    numpy.testing
    numpy.typing

    # Special-purpose
    numpy.array_api
    numpy.ctypeslib
    numpy.emath
    numpy.f2py
    numpy.math
    numpy.lib.stride_tricks
    numpy.rec
    numpy.types

    # Legacy (prefer not to use)
    numpy.char
    numpy.distutils
    numpy.matrixlib

    # To remove
    numpy.compat
    numpy.core?
    numpy.doc
    numpy.matlib
    numpy.version
    
    # To clean out or somehow deal with: everything in `numpy.lib`

Reducing the number of ways to select dtypes
````````````````````````````````````````````

The many dtype classes, instances, aliases and ways to select them are one of
the larger usability problems in the NumPy API. E.g.:

.. code:: python

   >>> # np.intp is different, but compares equal too
   >>> np.int64 == np.int_ == np.dtype('i8') == np.sctypeDict['i8']
   True
   >>> np.float64 == np.double == np.float_ == np.dtype('f8') == np.sctypeDict['f8']
   True
   ### Really?
   >>> np.clongdouble == np.clongfloat == np.longcomplex == np.complex256
   True

These aliases can go: https://numpy.org/devdocs/reference/arrays.scalars.html#other-aliases

To discuss:

- move *all* dtype-related classes to ``np.types``?
- mark one-character type code strings and related routines like ``mintypecode`` as legacy?
- canonical way to compare/select dtypes: ``np.isdtype`` (new, xref array API
  NEP), leaving ``np.issubdtype`` for the more niche use of numpy's dtype class
  hierarchy, and hide most other stuff.
- possibly remove ``float96``/``float128``? they're aliases that may not exist,
  and are too easy to shoot yourself in the foot with.


Related Work
------------

A clear split between public and private API was fairly recently done in SciPy
for 1.8.0 (2021), see `tracking issue scipy#14360 <https://github.com/scipy/scipy/issues/14360>`__).
The results of that were beneficial, and the impact relatively modest.


Implementation
--------------

The full implementation will be split over many different PRs, each touching on
a single API or a set of related APIs. To illustrate what those PRs will look
like, we will link here to a representative set of example PRs:

Deprecating non-preferred aliases and scheduling them for removal in 2.0:

- `gh-23302: deprecate np.round_; add round/min/max to the docs <https://github.com/numpy/numpy/pull/23302>`__
- `gh-23314: deprecate product/cumproduct/sometrue/alltrue <https://github.com/numpy/numpy/pull/23314>`__

Hiding or removing objects that are accidentally made public or not even NumPy objects at all:

- `gh-21403: remove some names from main numpy namespace <https://github.com/numpy/numpy/pull/21403>`__

Restructuring of public submodules:

- `gh-18447: hide internals of np.lib to only show submodules <https://github.com/numpy/numpy/pull/18447>`__

Create new namespaces to make it easier to navigate the module structure:

- `gh-22644: Add new np.exceptions namespace for errors and warnings <https://github.com/numpy/numpy/pull/22644>`__


Alternatives
------------



Discussion
----------


References and Footnotes
------------------------


Copyright
---------

This document has been placed in the public domain.
