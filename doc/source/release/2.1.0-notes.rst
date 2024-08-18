.. currentmodule:: numpy

=========================
NumPy 2.1.0 Release Notes
=========================

NumPy 2.1.0 provides support for the upcoming Python 3.13 release and drops
support for Python 3.9. In addition to the usual bug fixes and updated Python
support, it helps get us back into our usual release cycle after the extended
development of 2.0. The highlights for this release are:

- Support for the array-api 2023.12 standard.
- Support for Python 3.13.
- Preliminary support for free threaded Python 3.13.

Python versions 3.10-3.13 are supported in this release. 


New functions
=============

New function ``numpy.unstack``
------------------------------

A new function ``np.unstack(array, axis=...)`` was added, which splits
an array into a tuple of arrays along an axis. It serves as the inverse
of `numpy.stack`.

(`gh-26579 <https://github.com/numpy/numpy/pull/26579>`__)


Deprecations
============

* The ``fix_imports`` keyword argument in ``numpy.save`` is deprecated. Since
  NumPy 1.17, ``numpy.save`` uses a pickle protocol that no longer supports
  Python 2, and ignored ``fix_imports`` keyword. This keyword is kept only
  for backward compatibility. It is now deprecated.

  (`gh-26452 <https://github.com/numpy/numpy/pull/26452>`__)

* Passing non-integer inputs as the first argument of `bincount` is now
  deprecated, because such inputs are silently cast to integers with no
  warning about loss of precision.

  (`gh-27076 <https://github.com/numpy/numpy/pull/27076>`__)


Expired deprecations
====================

* Scalars and 0D arrays are disallowed for ``numpy.nonzero`` and ``numpy.ndarray.nonzero``.

  (`gh-26268 <https://github.com/numpy/numpy/pull/26268>`__)

* ``set_string_function`` internal function was removed and ``PyArray_SetStringFunction``
  was stubbed out.

  (`gh-26611 <https://github.com/numpy/numpy/pull/26611>`__)


C API changes
=============

API symbols now hidden but customizable
---------------------------------------
NumPy now defaults to hide the API symbols it adds to allow all NumPy API
usage.  This means that by default you cannot dynamically fetch the NumPy API
from another library (this was never possible on windows).

If you are experiencing linking errors related to ``PyArray_API`` or
``PyArray_RUNTIME_VERSION``, you can define the
``NPY_API_SYMBOL_ATTRIBUTE`` to opt-out of this change.

If you are experiencing problems due to an upstream header including NumPy,
the solution is to make sure you ``#include "numpy/ndarrayobject.h"`` before
their header and import NumPy yourself based on  ``including-the-c-api``.

(`gh-26103 <https://github.com/numpy/numpy/pull/26103>`__)

Many shims removed from npy_3kcompat.h
--------------------------------------
Many of the old shims and helper functions were removed from
``npy_3kcompat.h``. If you find yourself in need of these, vendor the previous
version of the file into your codebase.

(`gh-26842 <https://github.com/numpy/numpy/pull/26842>`__)

New ``PyUFuncObject`` field ``process_core_dims_func``
------------------------------------------------------
The field ``process_core_dims_func`` was added to the structure
``PyUFuncObject``.  For generalized ufuncs, this field can be set to a function
of type ``PyUFunc_ProcessCoreDimsFunc`` that will be called when the ufunc is
called. It allows the ufunc author to check that core dimensions satisfy
additional constraints, and to set output core dimension sizes if they have not
been provided.

(`gh-26908 <https://github.com/numpy/numpy/pull/26908>`__)


New Features
============

Preliminary Support for Free-Threaded CPython 3.13
--------------------------------------------------

CPython 3.13 will be available as an experimental free-threaded build. See
https://py-free-threading.github.io, `PEP 703
<https://peps.python.org/pep-0703/>`_ and the `CPython 3.13 release notes
<https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython>`_ for
more detail about free-threaded Python.

NumPy 2.1 has preliminary support for the free-threaded build of CPython
3.13. This support was enabled by fixing a number of C thread-safety issues in
NumPy. Before NumPy 2.1, NumPy used a large number of C global static variables
to store runtime caches and other state. We have either refactored to avoid the
need for global state, converted the global state to thread-local state, or
added locking.

Support for free-threaded Python does not mean that NumPy is thread
safe. Read-only shared access to ndarray should be safe. NumPy exposes shared
mutable state and we have not added any locking to the array object itself to
serialize access to shared state. Care must be taken in user code to avoid
races if you would like to mutate the same array in multiple threads. It is
certainly possible to crash NumPy by mutating an array simultaneously in
multiple threads, for example by calling a ufunc and the ``resize`` method
simultaneously. For now our guidance is: "don't do that". In the future we would
like to provide stronger guarantees.

Object arrays in particular need special care, since the GIL
previously provided locking for object array access and no longer does. See
`Issue #27199 <https://github.com/numpy/numpy/issues/27199>`_ for more
information about object arrays in the free-threaded build.

If you are interested in free-threaded Python, for example because you have a
multiprocessing-based workflow that you are interested in running with Python
threads, we encourage testing and experimentation.

If you run into problems that you suspect are because of NumPy, please `open an
issue <https://github.com/numpy/numpy/issues/new/choose>`_, checking first if
the bug also occurs in the "regular" non-free-threaded CPython 3.13 build. Many
threading bugs can also occur in code that releases the GIL; disabling the GIL
only makes it easier to hit threading bugs.

(`gh-26157 <https://github.com/numpy/numpy/issues/26157#issuecomment-2233864940>`__)

* ``numpy.reshape`` and ``numpy.ndarray.reshape`` now support ``shape`` and
  ``copy`` arguments.

  (`gh-26292 <https://github.com/numpy/numpy/pull/26292>`__)

* NumPy now supports DLPack v1, support for older versions will
  be deprecated in the future.

  (`gh-26501 <https://github.com/numpy/numpy/pull/26501>`__)

* ``numpy.asanyarray`` now supports ``copy`` and ``device`` arguments, matching
  ``numpy.asarray``.

  (`gh-26580 <https://github.com/numpy/numpy/pull/26580>`__)

* ``numpy.printoptions``, ``numpy.get_printoptions``, and
  ``numpy.set_printoptions`` now support a new option, ``override_repr``, for
  defining custom ``repr(array)`` behavior.

  (`gh-26611 <https://github.com/numpy/numpy/pull/26611>`__)

* ``numpy.cumulative_sum`` and ``numpy.cumulative_prod`` were added as Array
  API compatible alternatives for ``numpy.cumsum`` and ``numpy.cumprod``. The
  new functions can include a fixed initial (zeros for ``sum`` and ones for
  ``prod``) in the result.

  (`gh-26724 <https://github.com/numpy/numpy/pull/26724>`__)

* ``numpy.clip`` now supports ``max`` and ``min`` keyword arguments which are
  meant to replace ``a_min`` and ``a_max``. Also, for ``np.clip(a)`` or
  ``np.clip(a, None, None)`` a copy of the input array will be returned instead
  of raising an error.

  (`gh-26724 <https://github.com/numpy/numpy/pull/26724>`__)

* ``numpy.astype`` now supports ``device`` argument.

  (`gh-26724 <https://github.com/numpy/numpy/pull/26724>`__)

``f2py`` can generate freethreading-compatible C extensions
-----------------------------------------------------------
Pass ``--freethreading-compatible`` to the f2py CLI tool to produce a C
extension marked as compatible with the free threading CPython
interpreter. Doing so prevents the interpreter from re-enabling the GIL at
runtime when it imports the C extension. Note that ``f2py`` does not analyze
fortran code for thread safety, so you must verify that the wrapped fortran
code is thread safe before marking the extension as compatible.

(`gh-26981 <https://github.com/numpy/numpy/pull/26981>`__)


Improvements
============

``histogram`` auto-binning now returns bin sizes >=1 for integer input data
---------------------------------------------------------------------------
For integer input data, bin sizes smaller than 1 result in spurious empty
bins.  This is now avoided when the number of bins is computed using one of the
algorithms provided by ``histogram_bin_edges``.

(`gh-12150 <https://github.com/numpy/numpy/pull/12150>`__)

``ndarray`` shape-type parameter is now covariant and bound to ``tuple[int, ...]``
----------------------------------------------------------------------------------
Static typing for ``ndarray`` is a long-term effort that continues
with this change.  It is a generic type with type parameters for
the shape and the data type.  Previously, the shape type parameter could be
any value.  This change restricts it to a tuple of ints, as one would expect
from using ``ndarray.shape``.  Further, the shape-type parameter has been
changed from invariant to covariant.  This change also applies to the subtypes
of ``ndarray``, e.g. ``numpy.ma.MaskedArray``.  See the
`typing docs <https://typing.readthedocs.io/en/latest/reference/generics.html#variance-of-generic-types>`_
for more information.

(`gh-26081 <https://github.com/numpy/numpy/pull/26081>`__)

``np.quantile`` with method ``closest_observation`` chooses nearest even order statistic
----------------------------------------------------------------------------------------
This changes the definition of nearest for border cases from the nearest odd
order statistic to nearest even order statistic. The numpy implementation now
matches other reference implementations.

(`gh-26656 <https://github.com/numpy/numpy/pull/26656>`__)

``lapack_lite`` is now thread safe
----------------------------------
NumPy provides a minimal low-performance version of LAPACK named ``lapack_lite``
that can be used if no BLAS/LAPACK system is detected at build time.

Until now, ``lapack_lite`` was not thread safe. Single-threaded use cases did
not hit any issues, but running linear algebra operations in multiple threads
could lead to errors, incorrect results, or segfaults due to data races.

We have added a global lock, serializing access to ``lapack_lite`` in multiple
threads.

(`gh-26750 <https://github.com/numpy/numpy/pull/26750>`__)

The ``numpy.printoptions`` context manager is now thread and async-safe
-----------------------------------------------------------------------
In prior versions of NumPy, the printoptions were defined using a combination
of Python and C global variables. We have refactored so the state is stored in
a python ``ContextVar``, making the context manager thread and async-safe.

(`gh-26846 <https://github.com/numpy/numpy/pull/26846>`__)

Type hinting ``numpy.polynomial``
---------------------------------
Starting from the 2.1 release, PEP 484 type annotations have been included for
the functions and convenience classes in ``numpy.polynomial`` and its
sub-packages.

(`gh-26897 <https://github.com/numpy/numpy/pull/26897>`__)

Improved ``numpy.dtypes`` type hints
------------------------------------
The type annotations for ``numpy.dtypes`` are now a better reflection of the
runtime: The ``numpy.dtype`` type-aliases have been replaced with specialized
``dtype`` *subtypes*, and the previously missing annotations for
``numpy.dtypes.StringDType`` have been added.

(`gh-27008 <https://github.com/numpy/numpy/pull/27008>`__)


Performance improvements and changes
====================================

* ``numpy.save`` now uses pickle protocol version 4 for saving arrays with
  object dtype, which allows for pickle objects larger than 4GB and improves
  saving speed by about 5% for large arrays.

  (`gh-26388 <https://github.com/numpy/numpy/pull/26388>`__)

* OpenBLAS on x86_64 and i686 is built with fewer kernels. Based on
  benchmarking, there are 5 clusters of performance around these kernels:
  ``PRESCOTT NEHALEM SANDYBRIDGE HASWELL SKYLAKEX``.

  (`gh-27147 <https://github.com/numpy/numpy/pull/27147>`__)

* OpenBLAS on windows is linked without quadmath, simplifying licensing

  (`gh-27147 <https://github.com/numpy/numpy/pull/27147>`__)

* Due to a regression in OpenBLAS on windows, the performance improvements when
  using multiple threads for OpenBLAS 0.3.26 were reverted.

  (`gh-27147 <https://github.com/numpy/numpy/pull/27147>`__)

``ma.cov`` and ``ma.corrcoef`` are now significantly faster
-----------------------------------------------------------
The private function has been refactored along with ``ma.cov`` and
``ma.corrcoef``. They are now significantly faster, particularly on large,
masked arrays.

(`gh-26285 <https://github.com/numpy/numpy/pull/26285>`__)


Changes
=======

* As ``numpy.vecdot`` is now a ufunc it has a less precise signature.
  This is due to the limitations of ufunc's typing stub.

  (`gh-26313 <https://github.com/numpy/numpy/pull/26313>`__)

* ``numpy.floor``, ``numpy.ceil``, and ``numpy.trunc`` now won't perform
  casting to a floating dtype for integer and boolean dtype input arrays.

  (`gh-26766 <https://github.com/numpy/numpy/pull/26766>`__)

``ma.corrcoef`` may return a slightly different result
------------------------------------------------------
A pairwise observation approach is currently used in ``ma.corrcoef`` to
calculate the standard deviations for each pair of variables. This has been
changed as it is being used to normalise the covariance, estimated using
``ma.cov``, which does not consider the observations for each variable in a
pairwise manner, rendering it unnecessary. The normalisation has been replaced
by the more appropriate standard deviation for each variable, which
significantly reduces the wall time, but will return slightly different
estimates of the correlation coefficients in cases where the observations
between a pair of variables are not aligned. However, it will return the same
estimates in all other cases, including returning the same correlation matrix
as ``corrcoef`` when using a masked array with no masked values.

(`gh-26285 <https://github.com/numpy/numpy/pull/26285>`__)

Cast-safety fixes in ``copyto`` and ``full``
--------------------------------------------
``copyto`` now uses NEP 50 correctly and applies this to its cast safety.
Python integer to NumPy integer casts and Python float to NumPy float casts
are now considered "safe" even if assignment may fail or precision may be lost.
This means the following examples change slightly:

* ``np.copyto(int8_arr, 1000)`` previously performed an unsafe/same-kind cast
   of the Python integer.  It will now always raise, to achieve an unsafe cast
   you must pass an array or NumPy scalar.

* ``np.copyto(uint8_arr, 1000, casting="safe")`` will raise an OverflowError
  rather than a TypeError due to same-kind casting.

* ``np.copyto(float32_arr, 1e300, casting="safe")`` will overflow to ``inf``
  (float32 cannot hold ``1e300``) rather raising a TypeError.

Further, only the dtype is used when assigning NumPy scalars (or 0-d arrays),
meaning that the following behaves differently:

* ``np.copyto(float32_arr, np.float64(3.0), casting="safe")`` raises.

* ``np.coptyo(int8_arr, np.int64(100), casting="safe")`` raises.
  Previously, NumPy checked whether the 100 fits the ``int8_arr``.

This aligns ``copyto``, ``full``, and ``full_like`` with the correct NumPy 2
behavior.

(`gh-27091 <https://github.com/numpy/numpy/pull/27091>`__)

