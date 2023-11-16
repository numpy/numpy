.. _NEP52:

=========================================
NEP 52 — Python API cleanup for NumPy 2.0
=========================================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Author: Stéfan van der Walt <stefanv@berkeley.edu>
:Author: Nathan Goldbaum <ngoldbaum@quansight.com>
:Author: Mateusz Sokół <msokol@quansight.com>
:Status: Accepted
:Type: Standards Track
:Created: 2023-03-28
:Resolution: https://mail.python.org/archives/list/numpy-discussion@python.org/thread/QLMPFTWA67DXE3JCUQT2RIRLQ44INS4F/

Abstract
--------

We propose to clean up NumPy's Python API for the NumPy 2.0 release.
This includes a more clearly defined split between what is public and what is
private, and reducing the size of the main namespace by removing aliases
and functions that have better alternatives. Furthermore, each function is meant
to be accessible from only one place, so all duplicates also need to be dropped.


Motivation and scope
--------------------

NumPy has a large API surface that evolved organically over many years:

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
For a full overview of what's considered public, private or a bit in between, see
`<https://github.com/numpy/numpy/blob/main/numpy/tests/test_public_api.py>`__.

The size of the API and the lacking definition of its boundaries
incur significant costs:

- **Users find it hard to disambiguate between similarly named
  functions.**

  Looking for functions with tab completion in IPython, a notebook, or an IDE
  is a challenge. E.g., type ``np.<TAB>`` and look at the first six items
  offered: two ufuncs (``abs``, ``add``), one alias (``absolute``), and three
  functions that are not intended for end-users (``add_docstring``,
  ``add_newdoc``, ``add_newdoc_ufunc``). As a result, the learning curve for
  NumPy is steeper than it has to be.

- **Libraries that mimic the NumPy API face significant implementation barriers.**

  For maintainers of NumPy API-compatible array libraries (Dask, CuPy, JAX,
  PyTorch, TensorFlow, cuNumeric, etc.) and compilers/transpilers (Numba,
  Pythran, Cython, etc.) there is an implementation cost to each object in the
  namespace. In practice, no other library has full support for the entire
  NumPy API, partly because it is so hard to know what to include when faced
  with a slew of aliases and legacy objects.

- **Teaching NumPy is more complicated than it needs to be.**

  Similarly, a larger API is confusing to learners, who not only have to *find*
  functions but have to choose *which* functions to use.

- **Developers are hesitant to grow the API surface.**

  This happens even when the changes are warranted, because they are aware of
  the above concerns.

.. R: TODO: find and link discussion about restructuring namespaces! (e.g.,
   find the thread with the GUI explorer person)

.. S: Aaron's post re: array API and NumPy 2.0:
   https://mail.python.org/archives/list/numpy-discussion@python.org/thread/TTZEUKXUICDHGTCX5EMR6DQTYOSDGRV7/#YKBWQ2AP76WYWAP6GFRYMPHZCKTC43KM

The scope of this NEP includes:

- Deprecating or removing functionality that is too niche for NumPy, not
  well-designed, superseded by better alternatives, an unnecessary alias,
  or otherwise a candidate for removal.
- Clearly separating public from private NumPy API by use of underscores.
- Restructuring the NumPy namespaces to be easier to understand and navigate.

Out of scope for this NEP are:

- Introducing new functionality or performance enhancements.


Usage and impact
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

In order to make it easier to adopt the changes in this NEP, we will:

1. Provide a transition guide that lists each API change and its replacement.
2. Explicitly flag all expired attributes with a meaningful ``AttributeError``
   that points out to the new place or recommends an alternative.
3. Provide a script to automate the migration wherever possible. This will be
   similar to ``tools/replace_old_macros.sed`` (which adapts code for a
   previous C API naming scheme change). This will be ``sed`` (or equivalent)
   based rather than attempting AST analysis, so it won't cover everything.


Detailed description
--------------------

Cleaning up the main namespace
``````````````````````````````

We expect to reduce the main namespace by a large number of entries, on the
order of 100. Here is a representative set of examples:

- ``np.inf`` and ``np.nan`` have 8 aliases between them, of which most can be removed.
- A collection of random and undocumented functions (e.g., ``byte_bounds``, ``disp``,
  ``safe_eval``, ``who``) listed in
  `gh-12385 <https://github.com/numpy/numpy/issues/12385>`__
  can be deprecated and removed.
- All ``*sctype`` functions can be deprecated and removed, they (see
  `gh-17325 <https://github.com/numpy/numpy/issues/17325>`__,
  `gh-12334 <https://github.com/numpy/numpy/issues/12334>`__,
  and other issues for ``maximum_sctype`` and related functions).
- The ``np.compat`` namespace, used during the Python 2 to 3 transition, will be removed.
- Functions that are narrow in scope, with very few public use-cases,
  will be removed. These will have to be identified manually and by issue triage.

New namespaces are introduced for warnings/exceptions (``np.exceptions``) and
for dtype-related functionality (``np.dtypes``). NumPy 2.0 is a good opportunity
to populate these submodules from the main namespace.

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

A basic principle we will adhere to is "one function, one location". Functions
that are exposed in more than one namespace (e.g., many functions are present
in ``numpy`` and ``numpy.lib``) need to find a single home.

We will reorganize the API reference guide along main and submodule namespaces,
and only within the main namespace use the current subdivision along
functionality groupings. Also by "mainstream" and special-purpose namespaces:

::

    # Regular/recommended user-facing namespaces for general use. Present these
    # as the primary set of namespaces to the users.
    numpy
    numpy.exceptions
    numpy.fft
    numpy.linalg
    numpy.polynomial
    numpy.random
    numpy.testing
    numpy.typing

    # Special-purpose namespaces. Keep these, but document them in a separate
    # grouping in the reference guide and explain their purpose.
    numpy.array_api
    numpy.ctypeslib
    numpy.emath
    numpy.f2py  # only a couple of public functions, like `compile` and `get_include`
    numpy.lib.stride_tricks
    numpy.lib.npyio
    numpy.rec
    numpy.dtypes
    numpy.array_utils

    # Legacy (prefer not to use, there are better alternatives and/or this code
    # is deprecated or isn't reliable). This will be a third grouping in the
    # reference guide; it's still there, but de-emphasized and the problems
    # with it or better alternatives are explained in the docs.
    numpy.char
    numpy.distutils
    numpy.ma
    numpy.matlib

    # To remove
    numpy.compat
    numpy.core  # rename to _core
    numpy.doc
    numpy.math
    numpy.version  # rename to _version
    numpy.matrixlib

    # To clean out or somehow deal with: everything in `numpy.lib`

.. note::

    TBD: will we preserve ``np.lib`` or not? It only has a couple of unique
    functions/objects, like ``Arrayterator`` (a candidate for removal), ``NumPyVersion``,
    and the ``stride_tricks``, ``mixins`` and ``format`` subsubmodules.
    ``numpy.lib`` itself is not a coherent namespace, and does not even have a
    reference guide page.

We will make all submodules available lazily, so that users don't have to type
``import numpy.xxx`` but can use ``import numpy as np; np.xxx.*``, while at the
same time not negatively impacting the overhead of ``import numpy``. This has
been very helpful for teaching scikit-image and SciPy, and it resolves a
potential issue for Spyder users because Spyder already makes all submodules
available - so code using the above import pattern then works in Spyder but not
outside it.


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

All one-character type code strings and related routines like ``mintypecode``
will be marked as legacy.

To discuss:

- move *all* dtype-related classes to ``np.dtypes``?
- canonical way to compare/select dtypes: ``np.isdtype`` (new, xref array API
  NEP), leaving ``np.issubdtype`` for the more niche use of numpy's dtype class
  hierarchy, and hide most other stuff.
- possibly remove ``float96``/``float128``? they're aliases that may not exist,
  and are too easy to shoot yourself in the foot with.


Cleaning up the niche methods on ``numpy.ndarray``
``````````````````````````````````````````````````

The ``ndarray`` object has a lot of attributes and  methods, some of which are
too niche to be that prominent, all that does is distract the average user.
E.g.:

- ``.itemset`` (already discouraged)
- ``.newbyteorder`` (too niche)
- ``.ptp`` (niche, use ``np.ptp`` function instead)


API changes considered and rejected
-----------------------------------

For some functions and submodules it turned out that removing them would cause
too much disruption or would require an amount of work disproportional to the
actual gain. We arrived at this conclusion for such items:

- Removing business day functions: ``np.busday_count``, ``np.busday_offset``, ``np.busdaycalendar``.
- Removing ``np.nan*`` functions and introducing new ``nan_mode`` argument to the related base functions.
- Hiding histogram functions in the ``np.histograms`` submodule.
- Hiding ``c_``, ``r_`` and ``s_`` in the ``np.lib.index_tricks`` submodule.
- Functions that looked niche but are present in the Array API (for example ``np.can_cast``).
- Removing ``.repeat`` and ``.ctypes`` from ``ndarray`` object.


Related work
------------

A clear split between public and private API was recently established
as part of SciPy 1.8.0 (2021), see
`tracking issue scipy#14360 <https://github.com/scipy/scipy/issues/14360>`__.
The results were beneficial, and the impact on users relatively modest.


Implementation
--------------

The implementation has been split over many different PRs, each touching on
a single API or a set of related APIs. Here's a sample of the most impactful PRs:

- `gh-24634: Rename numpy/core to numpy/_core <https://github.com/numpy/numpy/pull/24634>`__
- `gh-24357: Cleaning numpy/__init__.py and main namespace - Part 2 <https://github.com/numpy/numpy/pull/24357>`__
- `gh-24376: Cleaning numpy/__init__.py and main namespace - Part 3 <https://github.com/numpy/numpy/pull/24376>`__

The complete list of cleanup work done in the 2.0 release can be found by searching a dedicated label:

- `Numpy 2.0 API Changes: <https://github.com/numpy/numpy/labels/Numpy%202.0%20API%20Changes>`__

Some PRs has already been merged and shipped with the `1.25.0` release.
For example, deprecating non-preferred aliases:

- `gh-23302: deprecate np.round_; add round/min/max to the docs <https://github.com/numpy/numpy/pull/23302>`__
- `gh-23314: deprecate product/cumproduct/sometrue/alltrue <https://github.com/numpy/numpy/pull/23314>`__

Hiding or removing objects that are accidentally made public or not even NumPy objects at all:

- `gh-21403: remove some names from main numpy namespace <https://github.com/numpy/numpy/pull/21403>`__

Creation of new namespaces to make it easier to navigate the module structure:

- `gh-22644: Add new np.exceptions namespace for errors and warnings <https://github.com/numpy/numpy/pull/22644>`__


Alternatives
------------



Discussion
----------

- `gh-23999: Tracking issue for the NEP 52 <https://github.com/numpy/numpy/issues/23999>`__

- `gh-24306: Overhaul of the main namespace <https://github.com/numpy/numpy/issues/24306>`__

- `gh-24507: Overhaul of the np.lib namespace <https://github.com/numpy/numpy/issues/24507>`__

References and footnotes
------------------------


Copyright
---------

This document has been placed in the public domain.
