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
private, reducing the size of the main namespace by removing aliases
and functions that have better alternatives.


Motivation and Scope
--------------------

NumPy has a large API surface that evolved organically over many
years:

.. code:: python

   >>> objects_in_api = [s for s in dir(np) if not s.startswith('_')]
   >>> len(objects_in_api)
   562
   >>> modules = [s for s in objects_in_api if inspect.ismodule(eval(f'np.{s}'))]
   >>> modules
   ['char', 'compat', 'ctypeslib', 'emath', 'fft', 'lib', 'linalg', 'ma', 'math', 'polynomial', 'random', 'rec', 'testing', 'version']
   >>> len(modules)
   14

The above doesn't even include items that are public but have been
been hidden from ``__dir__``.
A particularly problematic example of that is ``np.core``,
which is technically private but heavily used in practice.
For a full overview of what's consider public, private or a bit in between, see
`<https://github.com/numpy/numpy/blob/main/numpy/tests/test_public_api.py>`__.

The size of the API and the lacking definition of its boundaries
incur significant costs:

- **Users find it hard to disambiguate between similarly named
  functions.**

  Looking for functions with
  tab completion in IPython, a notebook, or an IDE is a challenge. E.g., type
  ``np.<TAB>`` and look at the first six items offered: two ufuncs
  (``abs``, ``add``), one alias (``absolute``), and three functions that are
  not intended for end-users (``add_docstring``, ``add_newdoc``, ``add_newdoc_ufunc``).
  As a result, the learning curve for NumPy is steeper than it has to be.

- **Libraries that mimic the NumPy API face significant implementation barriers.**

  For maintainers of NumPy API-compatible array libraries (Dask, CuPy, JAX, PyTorch,
  TensorFlow, cuNumeric, etc.) and compilers/transpilers (Numba, Pythran,
  Cython, etc.) there is an implementation cost to each object in the
  namespace. In practice, no other library has full support for the entire
  NumPy API, partly because it is so hard to know what to include when faced
  with a slew of aliases and legacy objects.

- **Teaching NumPy is more complicated than it needs to be.**

  Similarly, a larger API is confusing to learners, who not only have
  to *find* functions but have to choose *which* functions to use.

- **Developers are hesitant to grow the API surface.**

  This happens even when the changes are warranted, because they are
  aware of the above concerns.

.. R: Link discussion about restructuring namespaces! (e.g., find the thread
   with the GUI explorer person)

.. S: I first thought you were talking about Manim,
   but looks like it's something different.

.. S: Aaron's post re: array API and NumPy 2.0:
   https://mail.python.org/archives/list/numpy-discussion@python.org/thread/TTZEUKXUICDHGTCX5EMR6DQTYOSDGRV7/#YKBWQ2AP76WYWAP6GFRYMPHZCKTC43KM

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

As mentioned above, while the new (or cleaned up, NumPy 2.0) API should be
backward compatible, there is no guarantee of forward compatibility from 1.25.X
to 2.0. Code will have to be updated to account for deprecated, moved, or
removed functions/classes, as well as for more strictly enforced private APIs.

In order to make it easier to adapt to the changes in this NEP, we will:

1. Provide a transition guide that lists each API change and its replacement.
2. Provide a script to automate the migration wherever possible. This will be
   similar to ``tools/replace_old_macros.sed`` (which adapts code for a
   previous C API naming scheme change). This will be ``sed`` (or equivalent)
   based rather than attempting AST analysis, so it won't cover everything.


Detailed description
--------------------

Cleaning up the main namespace
``````````````````````````````

We expect to reduce the main namespace by a large number of entries,
on the order of 100.
Here is a representative set of examples:

- ``np.inf`` and ``np.nan`` have 8 aliases between them, of which most can be removed.
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
- The ``np.compat`` namespace, used during the Python 2 to 3 transition, will be removed.
- Functions that are narrow in scope, with very few public use-cases,
  will be removed.  See, e.g.
  ``real_if_close`` (`gh-11375 <https://github.com/numpy/numpy/issues/11375>`__).
  These will have to be identified manually and by issue triage.

New namespaces are introduced for warnings/exceptions (``np.exceptions``) and
for dtype-related functionality (``np.types``). NumPy 2.0 is a good opportunity
to populate these submodules from the main namespace.

.. S: Has the ``np.types`` name been fixed? Wonder if we're going to
   create confusion with that name.

Functionality that is widely used but has a preferred alternative may either be
deprecated (with the deprecation message pointing out what to use instead) or
be hidden by not including it in ``__dir__``. In case of hiding, a ``..
legacy::`` directory may be used to mark such functionality in the
documentation.

A test will be added to ensure limited future growth of all namespaces; i.e.,
every new entry will need to be explicitly added to an allow-list.


Cleaning up the submodule structure
```````````````````````````````````

We will clean up the NumPy submodule structure, so it is easier to navigate.
When this was discussed before (see
`MAINT: Hide internals of np.lib to only show submodules <https://github.com/numpy/numpy/pull/18447>`__)
there was already rough consensus on that - however it was hard to pull off in
a minor release.

We will reorganize the API reference guide along main and submodule namespaces,
and only within the main namespace use the current subdivision along
functionality groupings. Also by "mainstream" and special-purpose namespaces.
Details TBD, something like:

.. S: not sure what to call these submodules; made something up, but
   should be improved

::

    # `numpy.util`: Regular/recommended user-facing namespaces for general use
    numpy
    numpy.exceptions
    numpy.testing
    numpy.typing
    numpy.lib.stride_tricks
    numpy.types

    # `numpy.algorithms`: special purpose computation algorithms
    numpy.emath
    numpy.math
    numpy.fft
    numpy.linalg
    numpy.polynomial
    numpy.random

    # `numpy.ffi`: Special-purpose
    numpy.array_api
    numpy.ctypeslib
    numpy.f2py

    # `numpy.containers`:
    numpy.ma
    numpy.rec

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

.. S: Are you thinking that even math.* will disappear out of the name
   mainspace? That's quite a big change.  I like the principle you
   proposed on Sebastian's PR above: one function, one home.

.. S: Will we preserve `np.lib` as per the above discussion?

We will make all submodules available lazily, so that users don't have to type
``import numpy.xxx`` but can use ``import numpy as np; np.xxx.*``, while at the
same time not negatively impacting the overhead of ``import numpy``. This has
been very helpful for teaching scikit-image and SciPy.


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


.. S: consider `np.dtypes`.


Related Work
------------

A clear split between public and private API was recently established
as part of SciPy 1.8.0 (2021),
see `tracking issue scipy#14360 <https://github.com/scipy/scipy/issues/14360>`__.
The results were beneficial, and the impact on users relatively modest.


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
