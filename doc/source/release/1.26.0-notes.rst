.. currentmodule:: numpy

==========================
NumPy 1.26.0 Release Notes
==========================

The NumPy 1.26.0 release is a continuation of the 1.25.x release cycle with the
addition of Python 3.12.0 support. Python 3.12 dropped distutils, consequently
supporting it required finding a replacement for the setup.py/distutils based
build system NumPy was using. We have chosen to use the Meson build system
instead, and this is the first NumPy release supporting it. This is also the
first release that supports Cython 3.0 in addition to retaining 0.29.X
compatibility. Supporting those two upgrades was a large project, over 100
files have been touched in this release. The changelog doesn't capture the full
extent of the work, special thanks to Ralf Gommers, Sayed Adel, Stéfan van der
Walt, and Matti Picus who did much of the work in the main development branch.

The highlights of this release are:

- Python 3.12.0 support.
- Cython 3.0.0 compatibility.
- Use of the Meson build system
- Updated SIMD support
- f2py fixes, meson and bind(x) support
- Support for the updated Accelerate BLAS/LAPACK library

The Python versions supported in this release are 3.9-3.12.


New Features
============

Array API v2022.12 support in ``numpy.array_api``
-------------------------------------------------

- ``numpy.array_api`` now full supports the `v2022.12 version
  <https://data-apis.org/array-api/2022.12>`__ of the array API standard. Note
  that this does not yet include the optional ``fft`` extension in the
  standard.

(`gh-23789 <https://github.com/numpy/numpy/pull/23789>`__)

Support for the updated Accelerate BLAS/LAPACK library
------------------------------------------------------
Support for the updated Accelerate BLAS/LAPACK library, including ILP64 (64-bit
integer) support, in macOS 13.3 has been added. This brings arm64 support, and
significant performance improvements of up to 10x for commonly used linear
algebra operations. When Accelerate is selected at build time, the 13.3+
version will automatically be used if available.

(`gh-24053 <https://github.com/numpy/numpy/pull/24053>`__)

``meson`` backend for ``f2py``
------------------------------
``f2py`` in compile mode (i.e. ``f2py -c``) now accepts the ``--backend meson``
option. This is the default option for Python ``3.12`` on-wards. Older versions
will still default to ``--backend distutils``.

To support this in realistic use-cases, in compile mode ``f2py`` takes a
``--dep`` flag one or many times which maps to ``dependency()`` calls in the
``meson`` backend, and does nothing in the ``distutils`` backend.

There are no changes for users of ``f2py`` only as a code generator, i.e.
without ``-c``.

(`gh-24532 <https://github.com/numpy/numpy/pull/24532>`__)

``bind(c)`` support for ``f2py``
--------------------------------
Both functions and subroutines can be annotated with ``bind(c)``. ``f2py`` will
handle both the correct type mapping, and preserve the unique label for other
``C`` interfaces.

**Note:** ``bind(c, name = 'routine_name_other_than_fortran_routine')`` is not
honored by the ``f2py`` bindings by design, since ``bind(c)`` with the ``name``
is meant to guarantee only the same name in ``C`` and ``Fortran``, not in
``Python`` and ``Fortran``.

(`gh-24555 <https://github.com/numpy/numpy/pull/24555>`__)


Improvements
============

``iso_c_binding`` support for ``f2py``
--------------------------------------
Previously, users would have to define their own custom ``f2cmap`` file to use
type mappings defined by the Fortran2003 ``iso_c_binding`` intrinsic module.
These type maps are now natively supported by ``f2py``

(`gh-24555 <https://github.com/numpy/numpy/pull/24555>`__)


Build system changes
====================

In this release, NumPy has switched to Meson as the build system and
meson-python as the build backend. Installing NumPy or building a wheel can be
done with standard tools like ``pip`` and ``pypa/build``. The following are
supported:

- Regular installs: ``pip install numpy`` or (in a cloned repo)
  ``pip install .``
- Building a wheel: ``python -m build`` (preferred), or ``pip wheel .``
- Editable installs: ``pip install -e . --no-build-isolation``
- Development builds through the custom CLI implemented with
  `spin <https://github.com/scientific-python/spin>`__: ``spin build``.

All the regular ``pip`` and ``pypa/build`` flags (e.g.,
``--no-build-isolation``) should work as expected.

NumPy-specific build customization
----------------------------------

Many of the NumPy-specific ways of customizing builds have changed.
The ``NPY_*`` environment variables which control BLAS/LAPACK, SIMD, threading,
and other such options are no longer supported, nor is a ``site.cfg`` file to
select BLAS and LAPACK. Instead, there are command-line flags that can be
passed to the build via ``pip``/``build``'s config-settings interface. These
flags are all listed in the ``meson_options.txt`` file in the root of the repo.
Detailed documented will be available before the final 1.26.0 release; for now
please see `the SciPy "building from source" docs
<http://scipy.github.io/devdocs/building/index.html>`__ since most build
customization works in an almost identical way in SciPy as it does in NumPy.

Build dependencies
------------------

While the runtime dependencies of NumPy have not changed, the build
dependencies have. Because we temporarily vendor Meson and meson-python,
there are several new dependencies - please see the ``[build-system]`` section
of ``pyproject.toml`` for details.

Troubleshooting
---------------

This build system change is quite large. In case of unexpected issues, it is
still possible to use a ``setup.py``-based build as a temporary workaround (on
Python 3.9-3.11, not 3.12), by copying ``pyproject.toml.setuppy`` to
``pyproject.toml``. However, please open an issue with details on the NumPy
issue tracker. We aim to phase out ``setup.py`` builds as soon as possible, and
therefore would like to see all potential blockers surfaced early on in the
1.26.0 release cycle.


Contributors
============

A total of 20 people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.

* @DWesl
* Albert Steppi +
* Bas van Beek
* Charles Harris
* Developer-Ecosystem-Engineering
* Filipe Laíns +
* Jake Vanderplas
* Liang Yan +
* Marten van Kerkwijk
* Matti Picus
* Melissa Weber Mendonça
* Namami Shanker
* Nathan Goldbaum
* Ralf Gommers
* Rohit Goswami
* Sayed Adel
* Sebastian Berg
* Stefan van der Walt
* Tyler Reddy
* Warren Weckesser


Pull requests merged
====================

A total of 59 pull requests were merged for this release.

* `#24305 <https://github.com/numpy/numpy/pull/24305>`__: MAINT: Prepare 1.26.x branch for development
* `#24308 <https://github.com/numpy/numpy/pull/24308>`__: MAINT: Massive update of files from main for numpy 1.26
* `#24322 <https://github.com/numpy/numpy/pull/24322>`__: CI: fix wheel builds on the 1.26.x branch
* `#24326 <https://github.com/numpy/numpy/pull/24326>`__: BLD: update openblas to newer version
* `#24327 <https://github.com/numpy/numpy/pull/24327>`__: TYP: Trim down the ``_NestedSequence.__getitem__`` signature
* `#24328 <https://github.com/numpy/numpy/pull/24328>`__: BUG: fix choose refcount leak
* `#24337 <https://github.com/numpy/numpy/pull/24337>`__: TST: fix running the test suite in builds without BLAS/LAPACK
* `#24338 <https://github.com/numpy/numpy/pull/24338>`__: BUG: random: Fix generation of nan by dirichlet.
* `#24340 <https://github.com/numpy/numpy/pull/24340>`__: MAINT: Dependabot updates from main
* `#24342 <https://github.com/numpy/numpy/pull/24342>`__: MAINT: Add back NPY_RUN_MYPY_IN_TESTSUITE=1
* `#24353 <https://github.com/numpy/numpy/pull/24353>`__: MAINT: Update ``extbuild.py`` from main.
* `#24356 <https://github.com/numpy/numpy/pull/24356>`__: TST: fix distutils tests for deprecations in recent setuptools...
* `#24375 <https://github.com/numpy/numpy/pull/24375>`__: MAINT: Update cibuildwheel to version 2.15.0
* `#24381 <https://github.com/numpy/numpy/pull/24381>`__: MAINT: Fix codespaces setup.sh script
* `#24403 <https://github.com/numpy/numpy/pull/24403>`__: ENH: Vendor meson for multi-target build support
* `#24404 <https://github.com/numpy/numpy/pull/24404>`__: BLD: vendor meson-python to make the Windows builds with SIMD...
* `#24405 <https://github.com/numpy/numpy/pull/24405>`__: BLD, SIMD: The meson CPU dispatcher implementation
* `#24406 <https://github.com/numpy/numpy/pull/24406>`__: MAINT: Remove versioneer
* `#24409 <https://github.com/numpy/numpy/pull/24409>`__: REL: Prepare for the NumPy 1.26.0b1 release.
* `#24453 <https://github.com/numpy/numpy/pull/24453>`__: MAINT: Pin upper version of sphinx.
* `#24455 <https://github.com/numpy/numpy/pull/24455>`__: ENH: Add prefix to _ALIGN Macro
* `#24456 <https://github.com/numpy/numpy/pull/24456>`__: BUG: cleanup warnings [skip azp][skip circle][skip travis][skip...
* `#24460 <https://github.com/numpy/numpy/pull/24460>`__: MAINT: Upgrade to spin 0.5
* `#24495 <https://github.com/numpy/numpy/pull/24495>`__: BUG: ``asv dev`` has been removed, use ``asv run``.
* `#24496 <https://github.com/numpy/numpy/pull/24496>`__: BUG: Fix meson build failure due to unchanged inplace auto-generated...
* `#24521 <https://github.com/numpy/numpy/pull/24521>`__: BUG: fix issue with git-version script, needs a shebang to run
* `#24522 <https://github.com/numpy/numpy/pull/24522>`__: BUG: Use a default assignment for git_hash [skip ci]
* `#24524 <https://github.com/numpy/numpy/pull/24524>`__: BUG: fix NPY_cast_info error handling in choose
* `#24526 <https://github.com/numpy/numpy/pull/24526>`__: BUG: Fix common block handling in f2py
* `#24541 <https://github.com/numpy/numpy/pull/24541>`__: CI,TYP: Bump mypy to 1.4.1
* `#24542 <https://github.com/numpy/numpy/pull/24542>`__: BUG: Fix assumed length f2py regression
* `#24544 <https://github.com/numpy/numpy/pull/24544>`__: MAINT: Harmonize fortranobject
* `#24545 <https://github.com/numpy/numpy/pull/24545>`__: TYP: add kind argument to numpy.isin type specification
* `#24561 <https://github.com/numpy/numpy/pull/24561>`__: BUG: fix comparisons between masked and unmasked structured arrays
* `#24590 <https://github.com/numpy/numpy/pull/24590>`__: CI: Exclude import libraries from list of DLLs on Cygwin.
* `#24591 <https://github.com/numpy/numpy/pull/24591>`__: BLD: fix ``_umath_linalg`` dependencies
* `#24594 <https://github.com/numpy/numpy/pull/24594>`__: MAINT: Stop testing on ppc64le.
* `#24602 <https://github.com/numpy/numpy/pull/24602>`__: BLD: meson-cpu: fix SIMD support on platforms with no features
* `#24606 <https://github.com/numpy/numpy/pull/24606>`__: BUG: Change Cython ``binding`` directive to "False".
* `#24613 <https://github.com/numpy/numpy/pull/24613>`__: ENH: Adopt new macOS Accelerate BLAS/LAPACK Interfaces, including...
* `#24614 <https://github.com/numpy/numpy/pull/24614>`__: DOC: Update building docs to use Meson
* `#24615 <https://github.com/numpy/numpy/pull/24615>`__: TYP: Add the missing ``casting`` keyword to ``np.clip``
* `#24616 <https://github.com/numpy/numpy/pull/24616>`__: TST: convert cython test from setup.py to meson
* `#24617 <https://github.com/numpy/numpy/pull/24617>`__: MAINT: Fixup ``fromnumeric.pyi``
* `#24622 <https://github.com/numpy/numpy/pull/24622>`__: BUG, ENH: Fix ``iso_c_binding`` type maps and fix ``bind(c)``...
* `#24629 <https://github.com/numpy/numpy/pull/24629>`__: TYP: Allow ``binary_repr`` to accept any object implementing...
* `#24630 <https://github.com/numpy/numpy/pull/24630>`__: TYP: Explicitly declare ``dtype`` and ``generic`` hashable
* `#24637 <https://github.com/numpy/numpy/pull/24637>`__: ENH: Refactor the typing "reveal" tests using `typing.assert_type`
* `#24638 <https://github.com/numpy/numpy/pull/24638>`__: MAINT: Bump actions/checkout from 3.6.0 to 4.0.0
* `#24647 <https://github.com/numpy/numpy/pull/24647>`__: ENH: ``meson`` backend for ``f2py``
* `#24648 <https://github.com/numpy/numpy/pull/24648>`__: MAINT: Refactor partial load Workaround for Clang
* `#24653 <https://github.com/numpy/numpy/pull/24653>`__: REL: Prepare for the NumPy 1.26.0rc1 release.
* `#24659 <https://github.com/numpy/numpy/pull/24659>`__: BLD: allow specifying the long double format to avoid the runtime...
* `#24665 <https://github.com/numpy/numpy/pull/24665>`__: BLD: fix bug in random.mtrand extension, don't link libnpyrandom
* `#24675 <https://github.com/numpy/numpy/pull/24675>`__: BLD: build wheels for 32-bit Python on Windows, using MSVC
* `#24700 <https://github.com/numpy/numpy/pull/24700>`__: BLD: fix issue with compiler selection during cross compilation
* `#24701 <https://github.com/numpy/numpy/pull/24701>`__: BUG: Fix data stmt handling for complex values in f2py
* `#24707 <https://github.com/numpy/numpy/pull/24707>`__: TYP: Add annotations for the py3.12 buffer protocol
* `#24718 <https://github.com/numpy/numpy/pull/24718>`__: DOC: fix a few doc build issues on 1.26.x and update `spin docs`...
