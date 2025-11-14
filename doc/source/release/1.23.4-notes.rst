.. currentmodule:: numpy

==========================
NumPy 1.23.4 Release Notes
==========================
NumPy 1.23.4 is a maintenance release that fixes bugs discovered after the
1.23.3 release and keeps the build infrastructure current. The main
improvements are fixes for some annotation corner cases, a fix for a long time
``nested_iters`` memory leak, and a fix of complex vector dot for very large
arrays. The Python versions supported for this release are 3.8-3.11.

Note that the mypy version needs to be 0.981+ if you test using Python 3.10.7,
otherwise the typing tests will fail.

Contributors
============

A total of 8 people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.

* Bas van Beek
* Charles Harris
* Matthew Barber
* Matti Picus
* Ralf Gommers
* Ross Barnowski
* Sebastian Berg
* Sicheng Zeng +

Pull requests merged
====================

A total of 13 pull requests were merged for this release.

* `#22368 <https://github.com/numpy/numpy/pull/22368>`__: BUG: Add ``__array_api_version__`` to ``numpy.array_api`` namespace
* `#22370 <https://github.com/numpy/numpy/pull/22370>`__: MAINT: update sde toolkit to 9.0, fix download link
* `#22382 <https://github.com/numpy/numpy/pull/22382>`__: BLD: use macos-11 image on azure, macos-1015 is deprecated
* `#22383 <https://github.com/numpy/numpy/pull/22383>`__: MAINT: random: remove ``get_info`` from "extending with Cython"...
* `#22384 <https://github.com/numpy/numpy/pull/22384>`__: BUG: Fix complex vector dot with more than NPY_CBLAS_CHUNK elements
* `#22387 <https://github.com/numpy/numpy/pull/22387>`__: REV: Loosen ``lookfor``'s import try/except again
* `#22388 <https://github.com/numpy/numpy/pull/22388>`__: TYP,ENH: Mark ``numpy.typing`` protocols as runtime checkable
* `#22389 <https://github.com/numpy/numpy/pull/22389>`__: TYP,MAINT: Change more overloads to play nice with pyright
* `#22390 <https://github.com/numpy/numpy/pull/22390>`__: TST,TYP: Bump mypy to 0.981
* `#22391 <https://github.com/numpy/numpy/pull/22391>`__: DOC: Update delimiter param description.
* `#22392 <https://github.com/numpy/numpy/pull/22392>`__: BUG: Memory leaks in numpy.nested_iters
* `#22413 <https://github.com/numpy/numpy/pull/22413>`__: REL: Prepare for the NumPy 1.23.4 release.
* `#22424 <https://github.com/numpy/numpy/pull/22424>`__: TST: Fix failing aarch64 wheel builds.

