.. _NEP57:

===============================
NEP 57 — NumPy platform support
===============================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Status: Draft
:Type: Process
:Created: 2026-01-30
:Resolution: -


Abstract
--------

This NEP documents how a platform - i.e., a specific operating system, CPU
architecture and CPython interpreter - becomes supported in NumPy, what
platforms are currently supported, and were supported in the (recent) past.


Motivation and scope
--------------------

This policy is drafted now (early 2026) because there is a lot of interest in
extending the number of platforms NumPy supports through wheels in particular.
It is a policy specific to NumPy - even though other projects may possibly want
to refer to it - for several reasons:

* It involves committing to a nontrivial amount of maintainer effort,
* Personal commitment from a maintainer may make the difference between a
  yes and a no of supporting a platform (e.g., NumPy supported PyPy for a
  long time because of the efforts of one maintainer)
* Support for a platform being possible at all may depend on features of the
  code base (e.g., NumPy supports 32-bit Python on Windows while SciPy does
  not because there's no suitable compiler toolchain for it).
* The number of wheels depends on whether the Stable ABI can be used (NumPy
  is more performance-sensitive for small arrays, so can't use it)


The scope of this NEP includes:

- The definition of tiers of support for platforms by NumPy
- Policies and decision making for moving a platform to a different tier

Out of scope for this NEP are:

- Binary distributions of NumPy outside of PyPI
- Partial testing in CI (e.g., testing only SIMD-specific code under QEMU)
- More detailed breakdowns of wheels and support matrices, like compiler flavor
  and minimum version, or the BLAS library that is used in the build.


Support tiers
-------------

*This section is inspired by PEP 11 (CPython platform support), although
definitions are not matching, because NumPy is not nearly as large a project as
CPython.*

Platform support is broken down into tiers. Each tier comes with different
requirements which lead to different promises being made about support.

To be promoted to a tier,
`Steering council
<https://numpy.org/doc/stable/dev/governance/governance.html#steering-council>`__
support is required and is expected to be driven by team consensus. Demotion to
a lower tier occurs when the requirements of the current tier are no longer met
for a platform for an extended period of time based on the judgment of the
Steering Council. For platforms which no longer meet the requirements of any
tier by the middle of a new feature release cycle, an announcement will be made
to warn the community of the pending removal of support for the platform. If
the platform is not brought into line for at least one of the tiers by the
first release candidate, it will be listed as unsupported in this NEP.


General principles
~~~~~~~~~~~~~~~~~~

1. Maintainer effort is expensive, and we collectively have limited bandwidth -
   hence platform support is strongly influenced by the willingness of one or
   more maintainers to put in that effort.

   - Maintainers are trusted by the whole team. We generally do not question
     *why* a maintainer is motivated to put in the effort. If they are being
     paid for their effort or doing it as part of their job, that is fine -
     however they should disclose this to the Steering Council, and indicate
     whether long-term support is conditional on their employment or contractor
     status for the support tiers that include releasing wheels to PyPI.

     *Rationale: releasing wheels to PyPI is a long-term commitment by the
     project as a whole, see the backwards compatibility section below.*

2. CI support for the platform is required, preferably with native runners.
   Free is best, however decisions on paid CI are up to the Steering Council.
   Emulation for running the test suite (e.g., under QEMU) or self-hosted
   buildbots are slower and less reliable, hence not preferred.

3. There should be broad enough demand for support for the platform for the
   tiers that include releasing wheels to PyPI.

   - A previously used rule of thumb: >=0.5% of the user base should be on this
     platform. There may be reasons to deviate from this rule of thumb.

     *Note: finding clean data sources isn't always easy. If wheels are already
     being shipped, for NumPy or for a comparable project, then download data
     from PyPI may be obtained through BigQuery. For new platforms, sources
     like the*
     `Steam Hardware & Software Survey <https://store.steampowered.com/hwsurvey/?platform=combined>`__
     *may have to be used.*

4. Adding a regular CI job (i.e., not aimed at uploading wheels to PyPI) for a
   platform to the NumPy CI matrix is much cheaper, and easily reverted in case
   of problems. The bar for adding such jobs is low, and assessed on a
   case-by-case basis.

5. For all platforms in any supported tier: the relevant prerequisites in our
   dependencies must be met. E.g., build tools have support, and for wheels
   there is support in CPython, PyPI, cibuildwheel, manylinux, and
   ``scipy-openblas64`` or another easily-integrated BLAS library.

6. Decision making:

   - Moving a platform to a lower support tier must be discussed on the mailing list.
     The circumstances for each platform are unique so the community will
     evaluate each proposal to demote a platform on a case-by-case basis.
   - Moving a platform to a higher support tier, if that higher tier includes
     releasing wheels on PyPI for that platform, must be discussed on the
     mailing list.
   - Adding an entry to a support tier in this NEP for (a) an unsupported
     platform or (b) a tier which does not include uploading wheels to PyPI can
     be done on GitHub through a regular pull request (assuming it's clear from
     the discussion that the relevant maintainers agree it doesn't need to hit
     the mailing list).


Releasing wheels to PyPI
''''''''''''''''''''''''

The wheels that the NumPy team releases on PyPI for the ``numpy`` package get
hundreds of millions of downloads a month. We therefore highly value both
reliability and supply chain security of those release artifacts. Compromising
on those aspects is unlikely to be acceptable for the NumPy team.

The details of how wheels are produced, tested and distributed can be found in
the `numpy/numpy-release <https://github.com/numpy/numpy-release>`__
repository. Some key requirements of the current setup, which aren't likely to
change soon, are:

1. Must be buildable on publicly-visible CI infrastructure (i.e., GitHub).
2. Must be tested well enough (meaning native runners are preferred; QEMU is quite slow).
3. Must be publishable to PyPI automatically, through PyPI's trusted publishing
   mechanism.


Tier 1
~~~~~~

- Must have regular CI support on GitHub or (exceptionally) through another
  well-integrated CI platform that the release team and Steering Council deem
  acceptable.
- The NumPy team releases wheels on PyPI for this platform.
- CI failures (either regular CI or wheel build CI) block releases.
- All maintainers are responsible to keep the ``main`` branch and wheel builds
  working.

Tier 1 platforms:

+---------------------------+--------------------------------------------------------------------------+
| Platform                  | Notes                                                                    |
+===========================+==========================================================================+
| Windows x86-64            |                                                                          |
+---------------------------+--------------------------------------------------------------------------+
| Windows arm64             |                                                                          |
+---------------------------+--------------------------------------------------------------------------+
| Windows x86               | 32-bit Python: note this is shipped without BLAS, it's legacy            |
+---------------------------+--------------------------------------------------------------------------+
| Linux x86-64 (manylinux)  |                                                                          |
+---------------------------+--------------------------------------------------------------------------+
| Linux aarch64 (manylinux) |                                                                          |
+---------------------------+--------------------------------------------------------------------------+
| macOS arm64               |                                                                          |
+---------------------------+--------------------------------------------------------------------------+
| macOS x86-64              | Expected to move to unsupported by 2027/28 once the platform is dropped  |
|                           | by GitHub                                                                |
+---------------------------+--------------------------------------------------------------------------+


Tier 2
~~~~~~

- Must have regular CI support, either as defined for Tier 1 or through a
  reliable self-hosted service.
- The NumPy team releases wheels on PyPI for this platform.
- CI failures block releases.
- Must have at least one maintainer who commits to take primary and long-term
  responsibility for keeping the ``main`` branch and wheel builds working.

Tier 2 platforms:

+---------------------------+-------+------------------------------------------+
| Platform                  | Notes | Contacts                                 |
+===========================+=======+==========================================+
| Linux x86-64 (musllinux)  |       | Ralf Gommers                             |
+---------------------------+-------+------------------------------------------+
| Linux aarch64 (musllinux) |       | Ralf Gommers                             |
+---------------------------+-------+------------------------------------------+
| Free-threaded CPython     |       | Nathan Goldbaum, Kumar Aditya,           |
|                           |       | Ralf Gommers                             |
+---------------------------+-------+------------------------------------------+


Tier 3
~~~~~~

- Is supported as part of NumPy's regular CI setup for the ``main`` branch. CI
  support as defined for Tier 2.
- No wheels are released on PyPI for this platform.
- CI failures block releases (skips may be applied when the failure is clearly
  platform-specific and does not indicate a regression in core functionality).
- Must have at least one maintainer or a regular contributor trusted by the
  NumPy maintainers who commits to take responsibility for CI on the ``main``
  branch working.

Tier 3 platforms:

+--------------------+----------------------------------------+----------------------------------+
| Platform           | Notes                                  | Contacts                         |
+====================+========================================+==================================+
| FreeBSD            | Runs on Cirrus CI                      | Ralf Gommers                     |
+--------------------+----------------------------------------+----------------------------------+
| Linux ppc64le      | Runs on IBM-provided self-hosted       | Sandeep Gupta                    |
|                    | runners, see gh-22318_                 |                                  |
+--------------------+----------------------------------------+----------------------------------+
| Emscripten/Pyodide | We currently provide nightly wheels,   | Agriya Khetarpal, Gyeongjae Choi |
|                    | used for interactive docs              |                                  |
+--------------------+----------------------------------------+----------------------------------+


Unsupported platforms
~~~~~~~~~~~~~~~~~~~~~

All platforms not listed in the above tiers are unsupported by the NumPy team.
We do not develop and test on such platforms, and so cannot provide any
promises that NumPy will work on them.

However, the code base does include unsupported code – that is, code specific
to unsupported platforms. Contributions in this area are welcome as long as
they:

- pose a minimal maintenance burden to the core team, and
- benefit substantially more people than the contributor.

Unsupported platforms (previously in a supported tier, may be an incomplete
list):

+------------------------------------+--------------------------------------------------+
| Platform                           | Notes                                            |
+====================================+==================================================+
| PyPy                               | Was Tier 2 for releases <=2.4.x, see gh-30416_   |
+------------------------------------+--------------------------------------------------+
| macOS ppc64, universal, universal2 |                                                  |
+------------------------------------+--------------------------------------------------+
| Linux i686                         | Dropped in 1.22.0, low demand                    |
+------------------------------------+--------------------------------------------------+
| Linux on IBM Z (s390x)             | CI jobs used to run on TravisCI                  |
+------------------------------------+--------------------------------------------------+

Unsupported platforms (known interest in moving to a higher tier):

+----------+------------------+
| Platform | Notes            |
+==========+==================+
| iOS      | See gh-28759_    |
+----------+------------------+
| Android  | See gh-30412_    |
+----------+------------------+
| RISC-V   | See gh-30216_    |
+----------+------------------+
| WASI     | See gh-25859_    |
+----------+------------------+


Backward compatibility
----------------------

Moving a platform to a lower tier of support is generally backwards compatible.
The exception is stopping to release wheels on PyPI for a platform. That causes
significant disruption for existing users on that platform. Their install commands
(e.g., ``pip install numpy``) may stop working because if a new release no longer
has wheels for the platform, by default ``pip`` will try to build from source rather
than using a wheel from an older version of ``numpy``. Therefore, we should be very
reluctant to drop wheels for any platform.


Discussion
----------

- `ENH: Provide Windows ARM64 wheels (numpy#22530) <https://github.com/numpy/numpy/issues/22530>`__
- `Releasing PowerPC (ppc64le) wheels? (numpy#22318) <https://github.com/numpy/numpy/issues/22318>`__
- `MAINT: drop support for PyPy (numpy#30416) <https://github.com/numpy/numpy/issues/30416>`__
- `ENH: Build and distribute manylinux wheels for riscv64 <https://github.com/numpy/numpy/issues/30216>`__
- `BLD: Add support for building iOS wheels (numpy#28759) <https://github.com/numpy/numpy/pull/28759>`__
- `BLD: Add Android support <https://github.com/numpy/numpy/pull/30412>`__
- `ENH: WASI Build <https://github.com/numpy/numpy/issues/25859>`__
- `PEP 11 - CPython platform support <https://peps.python.org/pep-0011/>`__
- `Debian's supported architectures <https://wiki.debian.org/SupportedArchitectures>`__
- `Discussion about supported platforms for wheels (scientific-python issue/discussion (Nov 2025) <https://github.com/scientific-python/summit-2025-nov/issues/4>`__
- `What platforms should wheels be provided for by default? (Packaging Discourse thread, 2026) <https://discuss.python.org/t/what-platforms-should-wheels-be-provided-for-by-default/105822>`__
- `Expectations that projects provide ever more wheels (pypackaging-native) <https://pypackaging-native.github.io/meta-topics/user_expectations_wheels/>`__


References and footnotes
------------------------

.. _gh-22318: https://github.com/numpy/numpy/issues/22318
.. _gh-22530: https://github.com/numpy/numpy/issues/22530
.. _gh-25859: https://github.com/numpy/numpy/issues/25859
.. _gh-28759: https://github.com/numpy/numpy/pull/28759
.. _gh-30216: https://github.com/numpy/numpy/issues/30216
.. _gh-30412: https://github.com/numpy/numpy/pull/30412
.. _gh-30416: https://github.com/numpy/numpy/issues/30416


Copyright
---------

This document has been placed in the public domain.
