==========================
NumPy 1.15.2 Release Notes
==========================

This is a bugfix release for bugs and regressions reported following the 1.15.1
release.

* The matrix PendingDeprecationWarning is now suppressed in pytest 3.8.
* The new cached allocations machinery has been fixed to be thread safe.
* The boolean indexing of subclasses now works correctly.
* A small memory leak in PyArray_AdaptFlexibleDType has been fixed.

The Python versions supported by this release are 2.7, 3.4-3.7. The wheels are
linked with OpenBLAS v0.3.0, which should fix some of the linalg problems
reported for NumPy 1.14.

Compatibility Note
==================

The NumPy 1.15.x OS X wheels released on PyPI no longer contain 32-bit
binaries.  That will also be the case in future releases. See
`#11625 <https://github.com/numpy/numpy/issues/11625>`__ for the related
discussion.  Those needing 32-bit support should look elsewhere or build
from source.

Contributors
============

A total of 4 people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.

* Charles Harris
* Julian Taylor
* Marten van Kerkwijk
* Matti Picus

Pull requests merged
====================

A total of 4 pull requests were merged for this release.

* `#11902 <https://github.com/numpy/numpy/pull/11902>`__: BUG: Fix matrix PendingDeprecationWarning suppression for pytest...
* `#11981 <https://github.com/numpy/numpy/pull/11981>`__: BUG: fix cached allocations without the GIL for 1.15.x
* `#11982 <https://github.com/numpy/numpy/pull/11982>`__: BUG: fix refcount leak in PyArray_AdaptFlexibleDType
* `#11992 <https://github.com/numpy/numpy/pull/11992>`__: BUG: Ensure boolean indexing of subclasses sets base correctly.
