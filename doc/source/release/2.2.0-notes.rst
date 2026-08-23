.. currentmodule:: numpy

==========================
NumPy 2.2.0 Release Notes
==========================

The NumPy 2.2.0 release is quick release that brings us back into sync with the
usual twice yearly release cycle. There have been an number of small cleanups,
as well as work bringing the new StringDType to completion and improving support
for free threaded Python. Highlights are: 

* New functions ``matvec`` and ``vecmat``, see below.
* Many improved annotations.
* Improved support for the new StringDType.
* Improved support for free threaded Python
* Fixes for f2py

This release supports Python versions 3.10-3.13.


Deprecations
============

* ``_add_newdoc_ufunc`` is now deprecated. ``ufunc.__doc__ = newdoc`` should
  be used instead.

  (`gh-27735 <https://github.com/numpy/numpy/pull/27735>`__)


Expired deprecations
====================

* ``bool(np.array([]))`` and other empty arrays will now raise an error.
  Use ``arr.size > 0`` instead to check whether an array has no elements.

  (`gh-27160 <https://github.com/numpy/numpy/pull/27160>`__)


Compatibility notes
===================

* `numpy.cov` now properly transposes single-row (2d array) design matrices
  when ``rowvar=False``. Previously, single-row design matrices would return a
  scalar in this scenario, which is not correct, so this is a behavior change
  and an array of the appropriate shape will now be returned.

  (`gh-27661 <https://github.com/numpy/numpy/pull/27661>`__)


New Features
============

* New functions for matrix-vector and vector-matrix products

  Two new generalized ufuncs were defined:

  * `numpy.matvec` - matrix-vector product, treating the arguments as
    stacks of matrices and column vectors, respectively.

  * `numpy.vecmat` - vector-matrix product, treating the arguments as
    stacks of column vectors and matrices, respectively. For complex
    vectors, the conjugate is taken.

  These add to the existing `numpy.matmul` as well as to `numpy.vecdot`,
  which was added in numpy 2.0.

  Note that `numpy.matmul` never takes a complex conjugate, also not
  when its left input is a vector, while both `numpy.vecdot` and
  `numpy.vecmat` do take the conjugate for complex vectors on the
  left-hand side (which are taken to be the ones that are transposed,
  following the physics convention).

  (`gh-25675 <https://github.com/numpy/numpy/pull/25675>`__)

* ``np.complexfloating[T, T]`` can now also be written as
  ``np.complexfloating[T]``

  (`gh-27420 <https://github.com/numpy/numpy/pull/27420>`__)

* UFuncs now support ``__dict__`` attribute and allow overriding ``__doc__``
  (either directly or via ``ufunc.__dict__["__doc__"]``). ``__dict__`` can be
  used to also override other properties, such as ``__module__`` or
  ``__qualname__``.

  (`gh-27735 <https://github.com/numpy/numpy/pull/27735>`__)

* The "nbit" type parameter of ``np.number`` and its subtypes now defaults
  to ``typing.Any``. This way, type-checkers will infer annotations such as
  ``x: np.floating``  as ``x: np.floating[Any]``, even in strict mode.

  (`gh-27736 <https://github.com/numpy/numpy/pull/27736>`__)


Improvements
============

* The ``datetime64`` and ``timedelta64`` hashes now correctly match the Pythons
  builtin ``datetime`` and ``timedelta`` ones.  The hashes now evaluated equal
  even for equal values with different time units.

  (`gh-14622 <https://github.com/numpy/numpy/pull/14622>`__)

* Fixed a number of issues around promotion for string ufuncs with StringDType
  arguments. Mixing StringDType and the fixed-width DTypes using the string
  ufuncs should now generate much more uniform results.

  (`gh-27636 <https://github.com/numpy/numpy/pull/27636>`__)

* Improved support for empty `memmap`. Previously an empty `memmap` would fail
  unless a non-zero ``offset`` was set. Now a zero-size `memmap` is supported
  even if ``offset=0``. To achieve this, if a `memmap` is mapped to an empty
  file that file is padded with a single byte.

  (`gh-27723 <https://github.com/numpy/numpy/pull/27723>`__)

``f2py`` handles multiple modules and exposes variables again
-------------------------------------------------------------
A regression has been fixed which allows F2PY users to expose variables to
Python in modules with only assignments, and also fixes situations where
multiple modules are present within a single source file.

(`gh-27695 <https://github.com/numpy/numpy/pull/27695>`__)


Performance improvements and changes
====================================

* Improved multithreaded scaling on the free-threaded build when many threads
  simultaneously call the same ufunc operations.

  (`gh-27896 <https://github.com/numpy/numpy/pull/27896>`__)

* NumPy now uses fast-on-failure attribute lookups for protocols.  This can
  greatly reduce overheads of function calls or array creation especially with
  custom Python objects.  The largest improvements will be seen on Python 3.12
  or newer.

  (`gh-27119 <https://github.com/numpy/numpy/pull/27119>`__)

* OpenBLAS on x86_64 and i686 is built with fewer kernels. Based on
  benchmarking, there are 5 clusters of performance around these kernels:
  ``PRESCOTT NEHALEM SANDYBRIDGE HASWELL SKYLAKEX``.

* OpenBLAS on windows is linked without quadmath, simplifying licensing

* Due to a regression in OpenBLAS on windows, the performance improvements
  when using multiple threads for OpenBLAS 0.3.26 were reverted.

  (`gh-27147 <https://github.com/numpy/numpy/pull/27147>`__)

* NumPy now indicates hugepages also for large ``np.zeros`` allocations
  on linux.  Thus should generally improve performance.

  (`gh-27808 <https://github.com/numpy/numpy/pull/27808>`__)


Changes
=======

* `numpy.fix` now won't perform casting to a floating data-type for integer
  and boolean data-type input arrays.

  (`gh-26766 <https://github.com/numpy/numpy/pull/26766>`__)

* The type annotations of ``numpy.float64`` and ``numpy.complex128`` now
  reflect that they are also subtypes of the built-in ``float`` and ``complex``
  types, respectively. This update prevents static type-checkers from reporting
  errors in cases such as:

  .. code-block:: python

     x: float = numpy.float64(6.28)  # valid
     z: complex = numpy.complex128(-1j)  # valid

  (`gh-27334 <https://github.com/numpy/numpy/pull/27334>`__)

* The ``repr`` of arrays large enough to be summarized (i.e., where elements
  are replaced with ``...``) now includes the ``shape`` of the array, similar
  to what already was the case for arrays with zero size and non-obvious
  shape. With this change, the shape is always given when it cannot be
  inferred from the values.  Note that while written as ``shape=...``, this
  argument cannot actually be passed in to the ``np.array`` constructor. If
  you encounter problems, e.g., due to failing doctests, you can use the print
  option ``legacy=2.1`` to get the old behaviour.

  (`gh-27482 <https://github.com/numpy/numpy/pull/27482>`__)

* Calling ``__array_wrap__`` directly on NumPy arrays or scalars now does the
  right thing when ``return_scalar`` is passed (Added in NumPy 2).  It is
  further safe now to call the scalar ``__array_wrap__`` on a non-scalar
  result.

  (`gh-27807 <https://github.com/numpy/numpy/pull/27807>`__)

Bump the musllinux CI image and wheels to 1_2 from 1_1. This is because 1_1 is
`end of life <https://github.com/pypa/manylinux/issues/1629>`_.

(`gh-27088 <https://github.com/numpy/numpy/pull/27088>`__)

NEP 50 promotion state option removed
-------------------------------------
The NEP 50 promotion state settings are now removed. They were always meant as
temporary means for testing.  A warning will be given if the environment
variable is set to anything but ``NPY_PROMOTION_STATE=weak`` while
``_set_promotion_state`` and ``_get_promotion_state`` are removed.  In case
code used ``_no_nep50_warning``, a ``contextlib.nullcontext`` could be used to
replace it when not available.

(`gh-27156 <https://github.com/numpy/numpy/pull/27156>`__)

