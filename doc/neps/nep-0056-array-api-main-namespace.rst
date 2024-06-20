.. _NEP56:

=============================================================
NEP 56 — Array API standard support in NumPy's main namespace
=============================================================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Author: Mateusz Sokół <msokol@quansight.com>
:Author: Nathan Goldbaum <ngoldbaum@quansight.com>
:Status: Final
:Replaces: :ref:`NEP30`, :ref:`NEP31`, :ref:`NEP37`, :ref:`NEP47`
:Type: Standards Track
:Created: 2023-12-19
:Resolution: https://mail.python.org/archives/list/numpy-discussion@python.org/message/Z6AA5CL47NHBNEPTFWYOTSUVSRDGHYPN/


Abstract
--------

This NEP proposes adding nearly full support for the 2022.12 version of the
array API standard in NumPy's main namespace for the 2.0 release.

Adoption in the main namespace has a number of advantages; most importantly for
libraries that depend on NumPy and want to start supporting other array
libraries. SciPy and scikit-learn are two prominent libraries already moving
along this path. The need to support the array API standard in the main
namespace draws from lessons learned by those libraries and the experimental
``numpy.array_api`` implementation with a different array object.
There will also be benefits for other array libraries, JIT compilers like Numba,
and for end users who may have an easier time switching between different array
libraries.

Motivation and scope
--------------------

.. note::

    The main changes proposed in this NEP were presented in the NumPy 2.0
    Developer Meeting in April 2023 (see `here
    <https://github.com/numpy/archive/blob/main/2.0_developer_meeting/NumPy_2.0_devmeeting_array_API_adoption.pdf>`__
    for presentations from that meeting) and given a thumbs up there. The
    majority of the implementation work for NumPy 2.0 has already been merged.
    For the rest, PRs are ready - those are mainly the items that are specific
    to array API support and we'd probably not consider for inclusion in NumPy
    without that context. This NEP will focus on those APIs and PRs in a bit
    more detail.

:ref:`NEP47` contains the motivation for adding array API support to NumPy.
This NEP expands on and supersedes NEP 47. The main reason NEP 47 aimed for a
separate ``numpy.array_api`` submodule rather than the main namespace is that
casting rules differed too much. With value-based casting being removed
(:ref:`NEP50`), that will be resolved in NumPy 2.0. Having NumPy be a superset
of the array API standard will be a significant improvement for code
portability to other libraries (CuPy, JAX, PyTorch, etc.) and thereby address
one of the top user requests from the 2020 NumPy user survey [4]_ (GPU support).
See `the numpy.array_api API docs (1.26.x) <https://numpy.org/doc/1.26/reference/array_api.html#table-of-differences-between-numpy-array-api-and-numpy>`__
for an overview of differences between it and the main namespace (note that the
"strictness" ones are not applicable).

Experiences with ``numpy.array_api``, which is still marked as experimental,
have shown that the separate strict implementation and separate array object
are mostly good for testing purposes, but not for regular usage in downstream
libraries. Having support in the main namespace resolves this issue. Hence this
NEP supersedes NEP 47. The ``numpy.array_api`` module will be moved to a
standalone package, to facilitate easier updates not tied to a NumPy release
cycle.

Some of the key design rules from the array API standard (e.g., output dtypes
predictable from input dtypes, no polymorphic APIs with varying number of
returns controlled by keywords) will also be applied to NumPy functions that
are not part of the array API standard, because those design rules are now
understood to be good practice in general. Those two design rules in particular
make it easier for Numba and other JIT compilers to support NumPy or
NumPy-compatible APIs. We'll note that making existing arguments
positional-only and keyword-only is a good idea for functions added to NumPy in
the future, but will not be done for existing functions since each such change
is a backwards compatibility break and it's not necessary for writing code that
is portable across libraries supporting the standard. An additional reason to
apply those design rules to all functions in the main namespace now is that it
then becomes much easier to deal with potential standardization of new
functions already present in NumPy - those could otherwise be blocked or forced
to use alternative function names due to the need for backwards compatibility.

It is important that new functions added to the main namespace integrate well
with the rest of NumPy. So they should for example follow broadcasting and
other rules as expected, and work with all NumPy's dtypes rather than only the
ones in the standard. The same goes for backwards-incompatible changes (e.g.,
linear algebra functions need to all support batching in the same way, and
consider the last two axes as matrices). As a result, NumPy should become more
rather than less consistent.

Here are what we see as the main expected benefits and costs of the complete
set of proposed changes:

Benefits:

- It will enable array-consuming libraries (the likes of SciPy and
  scikit-learn, as well as smaller libraries higher up the stack) to implement
  support for multiple array libraries,
- It will remove the "having to make a choice between the NumPy API and the
  array API standard" issue for other array libraries when choosing what API
  to implement,
- Easier for CuPy, JAX, PyTorch, Dask, Numba, and other such libraries and
  compilers to match or support NumPy, through providing a more well-defined
  and minimal API surface to target, as well as through resolving some
  differences that were caused by Numpy semantics that were hard to support in
  JIT compilers,
- A few new features that have benefits independent of the standard: adding
  ``matrix_transpose`` and ``ndarray.mT``, adding ``vecdot``, introducing
  ``matrix_norm``/``vector_norm`` (they can be made gufuncs, vecdot already has
  a PR making it one),
- Closer correspondence between the APIs of NumPy and other array libraries
  will lower the learning curve for end users when they switch from one array
  library to another one,
- The array API standard tends to have more consistent behavior than NumPy
  itself has (in cases where there are differences between the two, see for
  example the `linear algebra design principles <https://data-apis.org/array-api/2022.12/extensions/linear_algebra_functions.html#design-principles>`__
  and `data-dependent output shapes page <https://data-apis.org/array-api/2022.12/design_topics/data_dependent_output_shapes.html>`__
  in the standard),

Costs:

- A number of backwards compatibility breaks (mostly minor, see the Backwards
  compatibility section further down),
- Expanding the size of the main namespace with about ~20 aliases (e.g.,
  ``acos`` & co. with C99 names aliasing ``arccos`` & co.).

Overall we believe that the benefits significantly outweigh the costs - and are
permanent, while the costs are largely temporary. In particular, the benefits
to array libraries and compilers that want to achieve compatibility with NumPy
are significant. And as a result, the long-term benefits for the PyData (or
scientific Python) ecosystem as a whole - because of downstream libraries being
able to support multiple array libraries much more easily - are
significant too. The number of breaking changes needed is fairly limited, and
the impact of those changes seems modest. Not painless, but we believe the
impact is smaller than the impact of other breaking changes in NumPy 2.0, and a
price worth paying.

In scope for this NEP are:

- Changes to NumPy's Python API needed to support the 2022.12 version of the
  array API standard, in the main namespace as well as ``numpy.linalg`` and
  ``numpy.fft``,
- Changes in the behavior of existing NumPy functions not (or not yet) present
  in the array API standard, to align with key design principles of the
  standard.

Out of scope for this NEP are:

- Other changes to NumPy's Python API unrelated to the array API standard,
- Changes to NumPy's C API.

This NEP will supersede the following NEPs:

- :ref:`NEP30` (never implemented)
- :ref:`NEP31` (never implemented)
- :ref:`NEP37` (never implemented; the ``__array_module__`` idea is basically
  the same as ``__array_namespace__``)
- :ref:`NEP47` (implemented with an experimental label in ``numpy.array_api``,
  will be removed)


Usage and impact
----------------

We have several different types of users in mind: end users writing numerical
code, downstream packages that depend on NumPy who want to start supporting
multiple array libraries, and other array libraries and tools which aim to
implement NumPy-like or NumPy-compatible APIs.

The most prominent users who will benefit from array API support are probably
downstream libraries that want to start supporting CuPy, PyTorch, JAX, Dask, or
other such libraries. SciPy and scikit-learn are already fairly far along the
way of doing just that, and successfully support CuPy arrays and PyTorch
tensors in a small part of their own APIs (that support is still marked as
experimental).

The main principle they use is that they replace the regular
``import numpy as np`` with a utility function to retrieve the array library
namespace from the input array. They call it ``xp``, which is effectively an
alias to ``np`` if the input is a NumPy array, ``cupy`` for a CuPy array,
``torch`` for a PyTorch tensor. This ``xp`` then allows writing code that works
for all these libraries - because the array API standard is the common
denominator. As a concrete example, this code is taken from ``scipy.cluster``:

.. code:: python

    def vq_py(obs, code_book, check_finite=True):
        """Python version of vq algorithm"""
        xp = array_namespace(obs, code_book)
        obs = as_xparray(obs, xp=xp, check_finite=check_finite)
        code_book = as_xparray(code_book, xp=xp, check_finite=check_finite)

        if obs.ndim != code_book.ndim:
            raise ValueError("Observation and code_book should have the same rank")

        if obs.ndim == 1:
            obs = obs[:, xp.newaxis]
            code_book = code_book[:, xp.newaxis]

        # Once `cdist` has array API support, this `xp.asarray` call can be removed
        dist = xp.asarray(cdist(obs, code_book))
        code = xp.argmin(dist, axis=1)
        min_dist = xp.min(dist, axis=1)
        return code, min_dist

It mostly looks like normal NumPy code, but will run with for example PyTorch
tensors as input and then return PyTorch tensors. There is a lot more to this
story of course then this basic example. These blog posts on scikit-learn [1]_
and SciPy's [2]_ experiences and impact (large performance gains in some cases
- ``LinearDiscriminantAnalysis.fit`` showed ~28x gain with PyTorch on GPU vs.
NumPy) paint a more complete picture.

For end users who are using NumPy directly, little changes aside from there
being fewer differences between NumPy and other libraries they may want to use
as well. This shortens their learning curve and makes it easier to switch
between NumPy and PyTorch/JAX/CuPy. In addition, they should benefit from
array-consuming libraries starting to support multiple array libraries, making
their experience of using a stack of Python packages for scientific computing
or data science more seamless.

Finally, for authors of other array libraries as well as tools like Numba,
API improvements which align NumPy with the array API standard will also save
them time. The design rules ([3]_), and in some cases new APIs like the
``unique_*`` ones, are easier to implement on GPU and for JIT compilers as a
result of more predictable behavior.


Backward compatibility
----------------------

The changes that have a backwards compatibility impact fall into these
categories:

1. Raising errors for consistency/strictness in some places where NumPy now
   allows more flexible behavior,
2. Dtypes of returned arrays for some element-wise functions and reductions,
3. Numerical behavior for a few tolerance keywords,
4. Functions moved to ``numpy.linalg`` and supporting stacking/batching,
5. The semantics of the ``copy`` keyword in ``asarray`` and ``array``,
6. Changes to ``numpy.fft`` functionality.

**Raising errors for consistency/strictness includes**:

1. Making ``.T`` error for >2 dimensions,
2. Making ``cross`` error on size-2 vectors (only size-3 vectors are supported),
3. Making ``solve`` error on ambiguous input (only accept ``x2`` as vector if ``x2.ndim == 1``),
4. ``outer`` raises rather than flattens on >1-D inputs,

*We expect the impact of this category of changes to be small.*

**Dtypes of returned arrays for some element-wise functions and reductions**
includes functions where dtypes need to be preserved: ``ceil``, ``floor``, and
``trunc`` will start returning arrays with the same integer dtypes if the input
has an integer dtype.

*We expect the impact of this category of changes to be small.*

**Changes in numerical behavior** include:

- The ``rtol`` default value for ``pinv`` changes from ``1e-15`` to a
  dtype-dependent default value of ``None``, interpreted as ``max(M, N) *
  finfo(result_dtype).eps``,
- The ``tol`` keyword to ``matrix_rank`` changes to ``rtol`` with a different
  interpretation. In addition, ``matrix_rank`` will no longer support 1-D array
  input,

Raising a ``FutureWarning`` for these tolerance changes doesn't seem reasonable;
they'd be spurious warnings for the vast majority of users, and it would force
users to hardcode a tolerance value to avoid the warning. Changes in numerical
results are in principle undesirable, so while we expect the impact to be small
it would be good to do this in a major release.

*We expect the impact of this category of changes to be medium. It is the only
category of changes that does not result in clear exceptions or warnings, and
hence if it does matter (e.g., downstream tests start failing or users notice
a change in behavior) it may require more work from users to track down the problem.
This should happen infrequently - one month after the PR implementing this change
was merged (see* `gh-25437 <https://github.com/numpy/numpy/pull/25437>`__),
*the impact reported so far is a single test failure in AstroPy.*

**Functions moved to numpy.linalg and supporting stacking/batching** are
the ``diagonal`` and ``trace`` functions. They part of the ``linalg`` submodule
in the standard, rather than the main namespace. Hence they will be introduced
in ``numpy.linalg``. They will operate on the last two rather than first two
axes. This is done for consistency, since this is now other NumPy functions
work, and to support "stacking" (or "batching" in more commonly used
terminology in other libraries). Hence the ``linalg`` and main namespace
functions of the same names will differ. This is technically not breaking, but
potentially confusing because of the different behavior for functions with the
same name. We may deprecate ``np.trace`` and ``np.diagonal`` to resolve it, but
preferably not immediately to avoid users having to write ``if-2.0-else``
conditional code.

*We expect the impact of this category of changes to be small.*

**The semantics of the copy keyword in asarray and array** for
``copy=False`` will change from "copy if needed" to "never copy". there are now
three types of behavior rather than two - ``copy=None`` means "copy if needed".

*We expect the impact of this category of changes to be medium. In case users get
an exception because they use* ``copy=False`` *explicitly in their copy but a
copy was previously made anyway, they have to inspect their code and determine
whether the intent of the code was the old or the new semantics (both seem
roughly equally likely), and adapt the code as appropriate. We expect most cases
to be* ``np.array(..., copy=False)``, *because until a few years ago that had
lower overhead than* ``np.asarray(...)``. *This was solved though, and*
``np.asarray(...)`` *is idiomatic NumPy usage.*

**Changes to numpy.fft**: all functions in the ``numpy.fft`` submodule need to
preserve precision for 32-bit input dtypes rather than upcast to
``float64``/``complex128``. This is a desirable change, consistent with the design
of NumPy as a whole - but it's possible that the lower precision or the dtype of
the returned arrays from calls to functions in this module may affect users.
This change was made by via a new gufunc-based implementation and vendoring of the
C++ version of PocketFFT in (`gh-25711 <https://github.com/numpy/numpy/pull/25711>`__).

A smaller backwards-incompatible change to ``numpy.fft`` is to make the
behavior of the ``s`` and ``axes`` arguments in n-D transforms easier to
understand by disallowing ``None`` values in ``s`` and requiring that if ``s``
is used, ``axes`` must be specified as well (see
`gh-25495 <https://github.com/numpy/numpy/pull/25495>`__).

*We expect the impact of this category of changes to be small.*


Adapting to the changes & tooling support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some parts of the array API have already been implemented as part of the general
Python API cleanup for NumPy 2.0 (see NEP 52), such as:

- establishing one and way for naming ``inf`` and ``nan`` that is array API
  compatible.
- removing cryptic dtype names and establishing (array API compatible)
  canonical names for each dtype.

All instructions for migrating to a NEP 52 compatible codebase are available in
the `NumPy 2.0 Migration Guide
<https://numpy.org/devdocs/numpy_2_0_migration_guide.html>`__ . 

Additionally, a new ``ruff`` rule was implemented for an automatic migration of
Python API changes. It's worth pointing out that the new rule NP201 is only to
adhere to the NEP 52 changes, and does not cover using new functions that are
part of the array API standard nor APIs with some types of backwards
incompatible changes discussed above.

For an automated migration to an array API compatible codebase, a new rule is
being implemented (see issue `ruff#8615 <https://github.com/astral-sh/ruff/issues/8615>`__
and PR `ruff#8910 <https://github.com/astral-sh/ruff/pull/8910>`__).

With both rules in place a downstream user should be able to update their
project, to the extent that is possible with automation, to a library
agnostic codebase that can benefit from different array libraries and devices.

Backwards incompatible changes that cannot be handled automatically (e.g., a
change in ``rtol`` defaults for a linear algebra function) will be handled the
in same way as any other backwards incompatible change in NumPy 2.0 -
through documentation, release notes, API migrations and deprecations over
several releases.


Detailed description
--------------------

In this section we'll focus on specific API additions and functionality that we
would not consider introducing into NumPy if the standard did not exist and
we didn't have to think/worry about its main goal: writing code that is
portable across multiple array libraries and their supported features like GPUs
and other hardware accelerators or JIT compilers.

``device`` support
^^^^^^^^^^^^^^^^^^

Device support is perhaps the most obvious example. NumPy is and will remain a
CPU-only library, so why bother introducing a ``ndarray.device`` attribute or
``device=`` keywords in several functions? This one feature is purely meant to
make it easier to write code that is portable across libraries. The ``.device``
attribute will return an object representing CPU, and that object will be
accepted as an input to ``device=`` keywords. For example:

.. code::

    # Should work when `xp` is `np` and `x1` a numpy array
    x2 = xp.asarray([0, 1, 2, 3], dtype=xp.float64, device=x1.device)

This will work as expected for NumPy, creating a 1-D numpy array from the input
list. It will also work for CuPy & co, where it may create a new array on a GPU
or other supported device.


``isdtype``
^^^^^^^^^^^

The array API standard introduced a new function ``isdtype`` for introspection
of dtypes, because there was no suitable alternative in NumPy. The closest one
is ``np.issubdtype``, however that assumes a complex class hierarchy which
other array libraries don't have, isn't the most ergonomic API, and required a
larger API surface (``np.floating`` and friends). ``isdtype`` will be the new
and canonical way to introspect dtypes. All it requires from a dtype is that
``__eq__`` is implemented and has the expected behavior when compared with other
dtypes from the same library.

Note that as part of the effort on NEP 52, some dtype aliases were removed and
canonical Python and C names documented. See also `gh-17325
<https://github.com/numpy/numpy/issues/17325>`__ covering issues with NumPy's
lack of a good API for this.


``copy`` keyword semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``copy`` keyword in ``asarray`` and ``array`` will now support
``True``/``False``/``None`` with new meanings:

- ``True`` - Always make a copy.
- ``False`` - Never make a copy. If a copy is required, a ``ValueError`` is raised.
- ``None`` - A copy will only be made if it is necessary (previously ``False``).

The ``copy`` keyword in ``astype`` will stick to its current meaning, because
"never copy" when asking for a cast to a different dtype doesn't quite make
sense.

There is still one hiccup for the change in semantics: if for user code
``np.array(obj, copy=False)``, NumPy may end up calling ``obj.__array__`` and
in that case turning the result into a NumPy array is the responsibility of the
implementer of ``obj.__array__``. Therefore, we need to add a ``copy=None``
keyword to ``__array__`` as well, and pass the copy keyword value along - taking
care to not break backwards compatibility when the implementer of ``__array__``
does not yet have the new keyword (a ``DeprecationWarning`` will be emitted in
that case, to allow for a gradual transition).


New function name aliases
^^^^^^^^^^^^^^^^^^^^^^^^^

In the Python API cleanup for NumPy 2.0 (see :ref:`NEP52`) we spent a lot of
effort removing aliases. So introducing new aliases has to have a good
rationale. In this case, it is needed in order to match other libraries.
The main set of aliases added is for trigonometric functions, where
the array API standard chose to follow C99 and other libraries in using
``acos``, ``asin`` etc. rather than ``arccos``, ``arcsin``, etc. NumPy usually
also follows C99; it is not entirely clear why this naming choice was made many
years ago.

In total 13 aliases are added to the main namespace and 2 aliases to
``numpy.linalg``:

- trigonometry functions: ``acos``, ``acosh``, ``asin``, ``asinh``, ``atan``,
  ``atanh``, ``atan2``
- bit-wise functions: ``bitwise_left_shift``, ``bitwise_invert``,
  ``bitwise_right_shift``
- other functions: ``concat``, ``permute_dims``, ``pow``
- in ``numpy.linalg``: ``tensordot``, ``matmul``

In the future NumPy can choose to hide the original names from its ``__dir__``
to nudge users to the preferred spelling for each function.


New keywords with overlapping semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly to function name aliases, there are a couple of new keywords which
have overlap with existing ones:

- ``correction`` keyword for ``std`` and ``var`` (overlaps with ``ddof``)
- ``stable`` keyword for ``sort`` and ``argsort`` (overlaps with ``kind``)

The ``correction`` name is for clarity ("delta degrees of freedom" is not easy
to understand). ``stable`` is complementary to ``kind``, which already has
``'stable'`` as an option (a separate keyword may be more discoverable though
and hence nice to have anyway), allowing a library to reserve the right to
change/improve the stable and unstable sorting algorithms.


New ``unique_*`` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``unique`` function, with ``return_index``, ``return_inverse``, and
``return_counts`` arguments that influence the cardinality of the returned
tuple, is replaced in the array API by four respective functions:
``unique_all``, ``unique_counts``, ``unique_inverse``, and ``unique_values``.
These new functions avoid polymorphism, which tends to be a problem for JIT
compilers and static typing. Use of these functions therefore helps tools like
Numba as well as users of static type checkers like Mypy.


``np.bool`` addition
^^^^^^^^^^^^^^^^^^^^

One of the aliases that used to live in NumPy but was removed is ``np.bool``.
To comply with the array API it was reintroduced with a different meaning, as
now it points to NumPy's bool instead of a Python builtin. This change is a
good idea and we were planning to make it anyway, because ``bool`` is a nicer
name than ``bool_``. However, we may not have scheduled that reintroduction of
the name for 2.0 if it had not been part of the array API standard.


Parts of the standard that are not adopted
------------------------------------------

There are a couple of things that the standard prescribes which we propose *not*
to follow (at least at this time). These are:

1. The requirement for ``sum`` and ``prod`` to always upcast lower-precision
   floating-point dtypes to ``float64`` when ``dtype=None``.

   *Rationale: this is potentially disruptive (e.g.,* ``float32_arr - float32_arr.mean()``
   *would yield a float64 array, and double memory use). While this upcasting
   is already done for inputs with lower-precision integer dtypes and seems
   useful there to prevent overflows, it seems less reasonable to require this
   for floating-point dtypes.*

   `array-api#731 <https://github.com/data-apis/array-api/issues/731>`__ was
   opened to reconsider this design choice in the standard, and that was accepted
   for the next standard version.

2. Making function signatures positional-only and keyword-only in many places.

   *Rationale: the 2022.12 version of the standard said "must", but this has
   already been softened to "should" in the about-to-be-released 2023.12
   version, to recognize that it's okay to not do this - it's still possible for
   users of the array library to write their code using the recommended style
   after all. For NumPy these changes would be useful, and it seems likely that
   we may introduce many or all of them over time (and in fact ufuncs are
   already compliant), however there is no need to rush this change - doing so
   for 2.0 would be unnecessarily disruptive.*

3. The requirement "An in-place operation must have the same behavior
   (including special cases) as its respective binary (i.e., two operand,
   non-assignment) operation" (excluding the effect on views).

   *Rationale: the requirement is very reasonable and probably expected
   behavior for most NumPy users. However, deprecating unsafe casts for
   in-place operators is a change for which the impact is hard to predict.
   Hence this needs to be investigated first, and then if the impact is low
   enough it may be possible to deprecate the current behavior according to
   NumPy's normal backwards compatibility guidelines.*

   This topic is tracked in
   `gh-25621 <https://github.com/numpy/numpy/issues/25621>`__.

.. note::

   We note that one NumPy-specific behavior that remains is returning array
   scalars rather than 0-D arrays in most cases where the standard, and other
   array libraries, return 0-D arrays (e.g., indexing and reductions). Array
   scalars basically duck type 0-D arrays, which is allowed by the standard (it
   doesn't mandate that there is only one array type, nor contains
   ``isinstance`` checks or other semantics that won't work with array
   scalars). There have been multiple discussions over the past year about the
   feasibility of removing array scalars from NumPy, or at least no longer
   returning them by default. However, this would be a large effort with some
   uncertainty about technical risks and impact of the change, and no one has
   taken it on. Given that array scalars implement a largely array-compatible
   interface, this doesn't seem like the highest-prio item regarding array API
   standard compatibility (or in general).


Related work
------------

The array API standard (`html docs <https://data-apis.org/array-api/2022.12/>`__,
`repository <https://github.com/data-apis/array-api/>`__) is the first related
work; a lot of design discussion in its issue tracker may be relevant in case
reasons for particular decisions need to be found.

Downstream adoption from array-consuming libraries is actively happening at the moment,
see for example:

- scikit-learn `docs on array API support <https://scikit-learn.org/dev/modules/array_api.html>`__ and
  `PRs <https://github.com/scikit-learn/scikit-learn/pulls?q=is%3Aopen+is%3Apr+label%3A%22Array+API%22>`__ and
  `issues <https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aopen+is%3Aissue+label%3A%22Array+API%22>`__
  labeled with *Array API*.
- SciPy `docs on array API support <http://scipy.github.io/devdocs/dev/api-dev/array_api.html>`__
  and `PRs <https://github.com/scipy/scipy/pulls?q=is%3Aopen+is%3Apr+label%3A%22array+types%22>`__
  and `issues <https://github.com/scipy/scipy/issues?q=is%3Aopen+is%3Aissue+label%3A%22array+types%22>`__ labeled with *array types*.
- Einops `docs on supported frameworks <https://einops.rocks/#supported-frameworks>`__
  and `PR to implement array API standard support <https://github.com/arogozhnikov/einops/pull/261>`__.

Other array libraries either already have support or are implementing support
for the array API standard (in sync with the changes for NumPy 2.0, since they
usually try to be as compatible to NumPy as possible). For example:

- CuPy's `docs on array API support <https://docs.cupy.dev/en/stable/reference/array_api.html>`__
  and `PRs labelled with array-api <https://github.com/cupy/cupy/pulls?q=is%3Aopen+is%3Apr+label%3Aarray-api>`__.
- JAX: enhancement proposal `Scope of JAX NumPy & SciPy Wrappers <https://jax.readthedocs.io/en/latest/jep/18137-numpy-scipy-scope.html#axis-2-array-api-alignment>`__
  and `tracking issue <https://github.com/google/jax/issues/18353>`__.


Implementation
--------------

The tracking issue for Array API standard support
(`gh-25076  <https://github.com/numpy/numpy/issues/25076>`__)
records progress of implementing full support and links to related discussions.
It lists all relevant PRs (merged and pending) that verify or provide array API
support.

As NEP 52 blends to some degree with this NEP, we can find some relevant implementations
and discussion also on its tracking issue (`gh-23999 <https://github.com/numpy/numpy/issues/23999>`__).

The PR that was merged as one of the first contained a new CI job that adds the
`array-api-tests <https://github.com/data-apis/array-api-tests>`__ test suite.
This way we had a better control over which batch of functions/aliases were being
added each time, and could be sure that the implementations conformed to the array
API standard (see `gh-25167 <https://github.com/numpy/numpy/pull/25167>`__).

Then, we continued to merge one batch at the time, adding a specific API
section. Below we list some of the more substantial ones, including some that
we discussed in the previous sections of this NEP:

- `gh-25167: MAINT: Add array-api-tests CI stage, add ndarray.__array_namespace__ <https://github.com/numpy/numpy/pull/25167>`__.
- `gh-25088: API: Add Array API setops [Array API] <https://github.com/numpy/numpy/pull/25088>`__
- `gh-25155: API: Add matrix_norm, vector_norm, vecdot and matrix_transpose [Array API] <https://github.com/numpy/numpy/pull/25155>`__
- `gh-25080: API: Add and redefine numpy.bool [Array API] <https://github.com/numpy/numpy/pull/25080>`__
- `gh-25054: API: Introduce np.isdtype function [Array API] <https://github.com/numpy/numpy/pull/25054>`__
- `gh-25168: API: Introduce copy argument for np.asarray [Array API] <https://github.com/numpy/numpy/pull/25168>`__


Alternatives
------------

The alternatives to implementing support for the array API standard in NumPy's
main namespace include:

- one or more of the superseded NEPs, or
- making ``ndarray.__array_namespace__()`` return a hidden namespace (or even
  another new public namespace) with compatible functions,
- not implementing support for the array API standard at all.

The superseded NEPs all have some drawbacks compared to the array API standard,
and by now a lot of work has gone into the standard - as well as adoption by
other key libraries. So those alternatives are not appealing. Given the amount
of interest in this topic, doing nothing also is not appealing. The "hidden
namespace" option would be a smaller change to this proposal. We prefer not to
do that since it leads to duplicate implementations staying around, a more
complex implementation (e.g., potential issues with static typing), and still
having two flavors of essentially the same API.

An alternative to removing ``numpy.array_api`` from NumPy is to keep it in its
current place, since it is still useful - it is the best way to test if
downstream code is actually portable between array libraries. This is a very
reasonable alternative, however there is a slight preference for taking that
module and turning it into a standalone package.


Discussion
----------



References and footnotes
------------------------

.. [1] https://labs.quansight.org/blog/array-api-support-scikit-learn
.. [2] https://labs.quansight.org/blog/scipy-array-api
.. [3] A. Meurer et al., "Python Array API Standard: Toward Array Interoperability in the Scientific Python Ecosystem." (2023), https://conference.scipy.org/proceedings/scipy2023/pdfs/aaron_meurer.pdf
.. [4] https://numpy.org/user-survey-2020/, 2020 NumPy User Survey results


Copyright
---------

This document has been placed in the public domain.
