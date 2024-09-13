
Contributors
============

A total of 148 people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.

* @DWesl
* @partev +
* @pierreloicq +
* @pkubaj +
* @pmvz +
* @pratiklp00 +
* @tajbinjohn +
* A Chethan Reddy +
* Aaron Meurer
* Aleksei Nikiforov +
* Alex Rogozhnikov
* Alexander Heger
* Alexander Neumann +
* Andrew Nelson
* Arun Kota +
* Bas van Beek
* Ben Greiner +
* Berke Kocaoğlu +
* Bob Eldering
* Brian Soto
* Brian Walshe +
* Brock Mendel
* Charles Harris
* Charles Young +
* Chris Brown
* Chris Sidebottom +
* Christian Lorentzen +
* Christian Veenhuis +
* Christopher Sidebottom +
* Chun-Wei Chen +
* Clément Robert
* Cédric Hannotier +
* Daiki Shintani +
* Daniel McCloy +
* Derek Homeier
* Developer-Ecosystem-Engineering
* DhavalParmar61 +
* Dimitri Papadopoulos Orfanos
* Dmitry Belov +
* Dominic Davis-Foster +
* Eddie Darling +
* Edward E +
* Eero Vaher
* Eric Wieser
* Eugene Kim +
* Evgeni Burovski
* Facundo Batista +
* Francesc Elies +
* Ganesh Kathiresan
* Hongyang Peng +
* Hood Chatham
* Ikko Ashimine
* Ikko Eltociear Ashimine +
* Inessa Pawson
* Irit Katriel
* Ivan Gonzalez
* JP Ungaretti +
* Jarrod Millman
* Jean-François B +
* Johana De La Rosa +
* Johnson Sun +
* Jonathan Kohler +
* Jory Klaverstijn +
* Joyce Brum +
* Jules Kouatchou +
* Kentaro Kawakami +
* Khem Raj +
* Kyle Sunden
* Larry Bradley
* Lars Grüter
* Laurenz Kremeyer +
* Lee Johnston +
* Lefteris Loukas +
* Leona Taric
* Lillian Zha
* Lu Yun Chi +
* Malte Londschien +
* Manuchehr Aminian +
* Mariusz Felisiak
* Mark Harfouche
* Mark J. Olson +
* Marko Pacak +
* Marten van Kerkwijk
* Matteo Raso
* Matthew Muresan +
* Matti Picus
* Meekail Zain
* Melissa Weber Mendonça
* Michael Kiffer +
* Michael Lamparski
* Michael Siebert
* Michail Gaganis +
* Mike Toews
* Mike-gag +
* Miki Watanabe
* Miki Watanabe (渡邉 美希)
* Miles Cranmer
* Muhammad Ishaque Nizamani +
* Mukulika Pahari
* Nathan Goldbaum
* Nico Schlömer
* Norwid Behrnd +
* Noé Rubinstein +
* Oleksandr Pavlyk
* Oscar Gustafsson
* Pamphile Roy
* Panagiotis Zestanakis +
* Paul Romano +
* Paulo Almeida +
* Pedro Lameiras +
* Peter Hawkins
* Pey Lian Lim
* Peyton Murray +
* Philip Holzmann +
* Pierre Blanchard +
* Pieter Eendebak
* Pradipta Ghosh
* Pratyay Banerjee +
* Prithvi Singh +
* Raghuveer Devulapalli
* Ralf Gommers
* Richie Cotton +
* Robert Kern
* Rohit Goswami
* Ross Barnowski
* Roy Smart +
* Rustam Uzdenov +
* Sadi Gulcelik +
* Sarah Kaiser +
* Sayed Adel
* Sebastian Berg
* Simon Altrogge +
* Somasree Majumder
* Stefan Behnel
* Stefan van der Walt
* Stefanie Molin
* StepSecurity Bot +
* Syam Gadde +
* Sylvain Ferriol +
* Talha Mohsin +
* Taras Tsugrii +
* Thomas A Caswell
* Tyler Reddy
* Warren Weckesser
* Will Tirone +
* Yamada Fuyuka +
* Younes Sandi +
* Yuki K +

Pull requests merged
====================

A total of 530 pull requests were merged for this release.

* `#10615 <https://github.com/numpy/numpy/pull/10615>`__: DEP: deprecate scalar conversions for arrays with ndim > 0
* `#16604 <https://github.com/numpy/numpy/pull/16604>`__: BUG: SWIG overloaded long functions on pyfragments.swg will seg...
* `#18053 <https://github.com/numpy/numpy/pull/18053>`__: ENH: Adding Object dtype to einsum
* `#18535 <https://github.com/numpy/numpy/pull/18535>`__: BUG: Fix <complex 0>^{non-zero}
* `#20064 <https://github.com/numpy/numpy/pull/20064>`__: MAINT, DOC: get rid of unicode, ``unicode_``, ``string_``
* `#20970 <https://github.com/numpy/numpy/pull/20970>`__: ENH: Move identity to the ArrayMethod to allow customization
* `#21056 <https://github.com/numpy/numpy/pull/21056>`__: ENH: re-implement SIMD kernels of complex operations
* `#21120 <https://github.com/numpy/numpy/pull/21120>`__: ENH: Add support for inplace matrix multiplication
* `#21785 <https://github.com/numpy/numpy/pull/21785>`__: BUG: fix ``_selected_real_kind_func`` return values for macOS on...
* `#21888 <https://github.com/numpy/numpy/pull/21888>`__: MAINT: changing the method of checking for nan / inf values linalg.eig
* `#22051 <https://github.com/numpy/numpy/pull/22051>`__: BLD: Add compile and runtime checks for AVX512_SPR
* `#22137 <https://github.com/numpy/numpy/pull/22137>`__: ENH allow for specifying CPU features to enable via ``NPY_ENABLE_CPU_FEATURES``...
* `#22165 <https://github.com/numpy/numpy/pull/22165>`__: ENH: Implement SIMD versions of isnan,isinf, isfinite and signbit
* `#22166 <https://github.com/numpy/numpy/pull/22166>`__: ENH: Add SIMD versions of negative
* `#22167 <https://github.com/numpy/numpy/pull/22167>`__: ENH: Add SIMD versions of bool logical_&&,||,! and absolute
* `#22284 <https://github.com/numpy/numpy/pull/22284>`__: DOC: Add random generator exponential example
* `#22315 <https://github.com/numpy/numpy/pull/22315>`__: ENH: Vectorize quicksort for 16-bit and 64-bit dtype using AVX512
* `#22493 <https://github.com/numpy/numpy/pull/22493>`__: MAINT: remove redundant open() modes and io.open() alias
* `#22527 <https://github.com/numpy/numpy/pull/22527>`__: DOC: Update the docstrings for np.around and ``np.round_``
* `#22533 <https://github.com/numpy/numpy/pull/22533>`__: API: Add numpy.testing.overrides to aid testing of custom array...
* `#22539 <https://github.com/numpy/numpy/pull/22539>`__: DEP: Deprecate ``np.find_common_type``
* `#22559 <https://github.com/numpy/numpy/pull/22559>`__: DOC #22266 Add examples for diag_indices_from()
* `#22562 <https://github.com/numpy/numpy/pull/22562>`__: DOC: #22266 Add examples for tril_indices_from(), triu_indices_from()
* `#22575 <https://github.com/numpy/numpy/pull/22575>`__: BUG: fix unexpected return of np.pad with mode=wrap
* `#22619 <https://github.com/numpy/numpy/pull/22619>`__: MAINT: Move set_module from numpy.core to numpy._utils
* `#22637 <https://github.com/numpy/numpy/pull/22637>`__: API: (cython) remove ``long_t`` and ``ulong_t``
* `#22638 <https://github.com/numpy/numpy/pull/22638>`__: DEP: Finalize MachAr and machar deprecations
* `#22644 <https://github.com/numpy/numpy/pull/22644>`__: API: Add new ``np.exceptions`` namespace for errors and warnings
* `#22646 <https://github.com/numpy/numpy/pull/22646>`__: MAINT: Rename symbols in textreading/ that may clash when statically...
* `#22647 <https://github.com/numpy/numpy/pull/22647>`__: MAINT: Prepare main for NumPy 1.25.0 development
* `#22649 <https://github.com/numpy/numpy/pull/22649>`__: MAINT: Remove the aarch64 python 3.8 wheel builds
* `#22650 <https://github.com/numpy/numpy/pull/22650>`__: CI: Add cirrus-ci to test linux_aarch64
* `#22653 <https://github.com/numpy/numpy/pull/22653>`__: MAINT: check user-defined dtype has an ensure_canonical implementation
* `#22655 <https://github.com/numpy/numpy/pull/22655>`__: DOC memmap #22643
* `#22663 <https://github.com/numpy/numpy/pull/22663>`__: BLD: enable building NumPy with Meson
* `#22666 <https://github.com/numpy/numpy/pull/22666>`__: MAINT: remove 'six' dependency from pyinstaller
* `#22668 <https://github.com/numpy/numpy/pull/22668>`__: DOC: lib: Use keepdims in a couple docstrings.
* `#22670 <https://github.com/numpy/numpy/pull/22670>`__: BUG: Polynomials now copy properly (#22669)
* `#22671 <https://github.com/numpy/numpy/pull/22671>`__: MAINT: pin ubuntu and python for emscripten
* `#22674 <https://github.com/numpy/numpy/pull/22674>`__: MAINT: replace ``NPY_INLINE`` with ``inline``
* `#22675 <https://github.com/numpy/numpy/pull/22675>`__: ENH: Slightly improve error when gufunc axes has wrong size
* `#22676 <https://github.com/numpy/numpy/pull/22676>`__: BUG: Ensure string aliases ``"int0"``, etc. remain valid for now
* `#22677 <https://github.com/numpy/numpy/pull/22677>`__: MAINT: fix ``.gitignore`` issues, and remove remaining ``NPY_INLINE``...
* `#22678 <https://github.com/numpy/numpy/pull/22678>`__: MAINT: fix typo in loops_unary_fp.dispatch.c.src
* `#22679 <https://github.com/numpy/numpy/pull/22679>`__: BLD: revert function() -> #define for 3 npymath functions
* `#22681 <https://github.com/numpy/numpy/pull/22681>`__: DOC: add numerical integration of x^2 to trapz
* `#22684 <https://github.com/numpy/numpy/pull/22684>`__: MAINT: npymath cleanups for isnan, isinf, isinfinite, signbit,...
* `#22685 <https://github.com/numpy/numpy/pull/22685>`__: BUILD: move wheel uploads to cirrus for aarch64
* `#22687 <https://github.com/numpy/numpy/pull/22687>`__: BLD: revert adding PEP 621 metadata, it confuses setuptools
* `#22689 <https://github.com/numpy/numpy/pull/22689>`__: MAINT: Add ``np._utils`` to meson
* `#22690 <https://github.com/numpy/numpy/pull/22690>`__: BLD: fix cirrus wheel upload triggers
* `#22693 <https://github.com/numpy/numpy/pull/22693>`__: MAINT: unify NPY_NO_SIGNAL macros
* `#22694 <https://github.com/numpy/numpy/pull/22694>`__: CI: Make benchmark asv run quick to only check that benchmarks...
* `#22703 <https://github.com/numpy/numpy/pull/22703>`__: BUG: Quantile function on complex number now throws an error...
* `#22705 <https://github.com/numpy/numpy/pull/22705>`__: DOC: misleading text lead to false hope
* `#22707 <https://github.com/numpy/numpy/pull/22707>`__: ENH,DEP: Add DTypePromotionError and finalize the == and != FutureWarning/Deprecation
* `#22708 <https://github.com/numpy/numpy/pull/22708>`__: DOC: Remove dangling deprecation warning
* `#22713 <https://github.com/numpy/numpy/pull/22713>`__: TST: Skip when numba/numpy compat issues cause SystemError
* `#22721 <https://github.com/numpy/numpy/pull/22721>`__: BUG: ``keepdims=True`` is ignored if ``out`` is not ``None``...
* `#22722 <https://github.com/numpy/numpy/pull/22722>`__: DEV: Add initial devcontainer config for codepaces
* `#22723 <https://github.com/numpy/numpy/pull/22723>`__: DOC: Structured array doc update to note ``dtype[name]``
* `#22724 <https://github.com/numpy/numpy/pull/22724>`__: BUG: Fix some valgrind errors (and probably harmless warnings)
* `#22725 <https://github.com/numpy/numpy/pull/22725>`__: ENH: Speedup masked array creation when mask=None
* `#22726 <https://github.com/numpy/numpy/pull/22726>`__: BENCH: Add a test for masked array creations
* `#22731 <https://github.com/numpy/numpy/pull/22731>`__: BENCH: Update MaskedArray Benchmarks
* `#22732 <https://github.com/numpy/numpy/pull/22732>`__: DOC: Add instruction to do ``git pull``
* `#22735 <https://github.com/numpy/numpy/pull/22735>`__: API: Hide exceptions from the main namespace
* `#22736 <https://github.com/numpy/numpy/pull/22736>`__: DOC: Improve description of the dtype parameter in np.array docstring
* `#22737 <https://github.com/numpy/numpy/pull/22737>`__: TST: skip floating-point error test on wasm
* `#22738 <https://github.com/numpy/numpy/pull/22738>`__: MAINT: check if PyArrayDTypeMeta_Spec->casts is set
* `#22747 <https://github.com/numpy/numpy/pull/22747>`__: DOC: Some updates to the array_api compat document
* `#22750 <https://github.com/numpy/numpy/pull/22750>`__: BUG, SIMD: Fix rounding large numbers on SSE2
* `#22751 <https://github.com/numpy/numpy/pull/22751>`__: CI, SIMD: Add workflow to test default baseline features only
* `#22752 <https://github.com/numpy/numpy/pull/22752>`__: CI: fix CIRRUS_TAG check when tagging. Closes #22730
* `#22753 <https://github.com/numpy/numpy/pull/22753>`__: BUG: Fix deepcopy cleanup on error
* `#22763 <https://github.com/numpy/numpy/pull/22763>`__: MAINT: allow unsized NEP 42 user-defined dtypes
* `#22769 <https://github.com/numpy/numpy/pull/22769>`__: BLD: Meson ``__config__`` generation
* `#22771 <https://github.com/numpy/numpy/pull/22771>`__: BUG, SIMD: Fix invalid value encountered in several ufuncs
* `#22775 <https://github.com/numpy/numpy/pull/22775>`__: DOC: fix typo in basics.dispatch.rst
* `#22776 <https://github.com/numpy/numpy/pull/22776>`__: BUG: fix ma.diff not preserving mask when using append/prepend
* `#22777 <https://github.com/numpy/numpy/pull/22777>`__: ENH: Properly support FreeBSD/powerpc64
* `#22779 <https://github.com/numpy/numpy/pull/22779>`__: DOC: All integer values must be non-negative
* `#22784 <https://github.com/numpy/numpy/pull/22784>`__: DOC: Fix legend placement in ``numpy.percentile()`` docs
* `#22785 <https://github.com/numpy/numpy/pull/22785>`__: DEV: Fix devcontainer configuration
* `#22786 <https://github.com/numpy/numpy/pull/22786>`__: ENH: Add namedtuple return types to linalg functions that return...
* `#22788 <https://github.com/numpy/numpy/pull/22788>`__: MAINT: Remove two TODO notes that got outdated
* `#22789 <https://github.com/numpy/numpy/pull/22789>`__: BUG: Fix infinite recursion in longdouble/large integer scalar...
* `#22791 <https://github.com/numpy/numpy/pull/22791>`__: BUG: Ensure arguments to ``npy_floatstatus_..._barrier()`` can be...
* `#22792 <https://github.com/numpy/numpy/pull/22792>`__: MAINT: elaborate on error message for unaligned casting spec
* `#22794 <https://github.com/numpy/numpy/pull/22794>`__: DOC: Add minimal windows bat file for building the docs
* `#22798 <https://github.com/numpy/numpy/pull/22798>`__: BUG: Fix refcounting errors found using pytest-leaks
* `#22799 <https://github.com/numpy/numpy/pull/22799>`__: TST: Remove outdated xfail from quantile tests
* `#22800 <https://github.com/numpy/numpy/pull/22800>`__: DOC: mention installing test dependencies in testing instructions
* `#22801 <https://github.com/numpy/numpy/pull/22801>`__: MAINT: remove unnecessary forward declaration of _convert_from_any
* `#22802 <https://github.com/numpy/numpy/pull/22802>`__: BLD: fix issue in npymath on macOS arm64 in the Meson build
* `#22803 <https://github.com/numpy/numpy/pull/22803>`__: MAINT: remove unused API documentation generation
* `#22804 <https://github.com/numpy/numpy/pull/22804>`__: REV: revert change to ``numpyconfig.h`` for sizeof(type) hardcoding...
* `#22806 <https://github.com/numpy/numpy/pull/22806>`__: DOC: update discussion in dtypes docs that references Python...
* `#22807 <https://github.com/numpy/numpy/pull/22807>`__: CI: undo permissions in circleci artifact redirector.
* `#22814 <https://github.com/numpy/numpy/pull/22814>`__: BLD: use newer version of delocate
* `#22816 <https://github.com/numpy/numpy/pull/22816>`__: BLD: redo delocate, update labeler
* `#22818 <https://github.com/numpy/numpy/pull/22818>`__: DOC: fix a couple typos in 1.23 notes.
* `#22822 <https://github.com/numpy/numpy/pull/22822>`__: MAINT: Update main after 1.24.0 release.
* `#22824 <https://github.com/numpy/numpy/pull/22824>`__: BLD: CIRRUS_TAG redux
* `#22828 <https://github.com/numpy/numpy/pull/22828>`__: TST: ignore more np.distutils.log imports
* `#22836 <https://github.com/numpy/numpy/pull/22836>`__: BUG: Ensure correct behavior for rows ending in delimiter in...
* `#22838 <https://github.com/numpy/numpy/pull/22838>`__: BUG: Do not use getdata() in np.ma.masked_invalid
* `#22846 <https://github.com/numpy/numpy/pull/22846>`__: BUG, SIMD: Fix the bitmask of the boolean comparison
* `#22849 <https://github.com/numpy/numpy/pull/22849>`__: API: Ensure a full mask is returned for masked_invalid
* `#22851 <https://github.com/numpy/numpy/pull/22851>`__: BUG, SIMD: Fix memory overlap in ufunc comparison loops
* `#22853 <https://github.com/numpy/numpy/pull/22853>`__: MAINT: enhance show_runtime
* `#22855 <https://github.com/numpy/numpy/pull/22855>`__: BUG: Fortify string casts against floating point warnings
* `#22856 <https://github.com/numpy/numpy/pull/22856>`__: BLD: Help raspian arm + clang 13 about ``__builtin_mul_overflow``
* `#22861 <https://github.com/numpy/numpy/pull/22861>`__: BUG, SIMD: Restore behavior converting non bool input to 0x00/0xff
* `#22863 <https://github.com/numpy/numpy/pull/22863>`__: ENH: allow NEP 42 dtypes to work with np.char
* `#22864 <https://github.com/numpy/numpy/pull/22864>`__: CI: musllinux_x86_64
* `#22869 <https://github.com/numpy/numpy/pull/22869>`__: TST: Ignore nan-warnings in randomized nanfunction ``out=`` tests
* `#22872 <https://github.com/numpy/numpy/pull/22872>`__: BUG: Use whole file for encoding checks with ``charset_normalizer``.
* `#22874 <https://github.com/numpy/numpy/pull/22874>`__: TST: Fixup string cast test to not use ``tiny``
* `#22878 <https://github.com/numpy/numpy/pull/22878>`__: BUG: Fix integer overflow in in1d for mixed integer dtypes #22877
* `#22879 <https://github.com/numpy/numpy/pull/22879>`__: BUG: Fixes for numpy.testing.overrides
* `#22880 <https://github.com/numpy/numpy/pull/22880>`__: DOC: Add a note to the documentation of the rot90
* `#22882 <https://github.com/numpy/numpy/pull/22882>`__: MAINT: restore npymath implementations needed for freebsd
* `#22885 <https://github.com/numpy/numpy/pull/22885>`__: MAINT: ``f2py`` cleanup
* `#22886 <https://github.com/numpy/numpy/pull/22886>`__: DOC: Add details on benchmarking versions with ASV
* `#22889 <https://github.com/numpy/numpy/pull/22889>`__: ENH: Speedup ufunc.at when casting is not needed
* `#22890 <https://github.com/numpy/numpy/pull/22890>`__: MAINT: Update main after 1.24.1 release.
* `#22891 <https://github.com/numpy/numpy/pull/22891>`__: TST: tests/core/test_multiarray:TestArg{Min,Max}: stray ``np.repeat``...
* `#22894 <https://github.com/numpy/numpy/pull/22894>`__: MAINT: change labeler
* `#22901 <https://github.com/numpy/numpy/pull/22901>`__: MAINT: Fix runtime information commands for issue template [skip...
* `#22902 <https://github.com/numpy/numpy/pull/22902>`__: MAINT: Replace Python3.8 by Python3.9.
* `#22905 <https://github.com/numpy/numpy/pull/22905>`__: TST: Add linspace test case for any_step_zero and not _mult_inplace
* `#22906 <https://github.com/numpy/numpy/pull/22906>`__: BUG: np.loadtxt cannot load text file with quoted fields separated...
* `#22908 <https://github.com/numpy/numpy/pull/22908>`__: DOC: Pull existing PR workflow
* `#22911 <https://github.com/numpy/numpy/pull/22911>`__: DOC: Remove Gitpod as a local build option for users
* `#22916 <https://github.com/numpy/numpy/pull/22916>`__: ENH: Faster numpy.load (try/except _filter_header)
* `#22917 <https://github.com/numpy/numpy/pull/22917>`__: DOC: document NPY_DISABLE_CPU_FEATURES
* `#22918 <https://github.com/numpy/numpy/pull/22918>`__: TST: split long ufunc.at test
* `#22921 <https://github.com/numpy/numpy/pull/22921>`__: BLD: Build wheels with cibuildwheel 2.12.0
* `#22923 <https://github.com/numpy/numpy/pull/22923>`__: ENH: Create string dtype instances from the abstract dtype
* `#22924 <https://github.com/numpy/numpy/pull/22924>`__: MAINT: Refactor clearing up of array data
* `#22927 <https://github.com/numpy/numpy/pull/22927>`__: MAINT: Remove distutils usage in travis-test.sh.
* `#22931 <https://github.com/numpy/numpy/pull/22931>`__: TST: Add fixture to avoid issue with randomizing test order.
* `#22934 <https://github.com/numpy/numpy/pull/22934>`__: DOC: Add releases to NEP 29
* `#22936 <https://github.com/numpy/numpy/pull/22936>`__: DOC: (LaTeX) fix 'fontenc' key for usage with xelatex
* `#22937 <https://github.com/numpy/numpy/pull/22937>`__: DOC: Fix typo in NEP-19
* `#22938 <https://github.com/numpy/numpy/pull/22938>`__: DOC: Update docstring of ``multivariate_normal``
* `#22939 <https://github.com/numpy/numpy/pull/22939>`__: MAINT: Move export for scipy arm64 helper into main module
* `#22943 <https://github.com/numpy/numpy/pull/22943>`__: MAINT: change circleCI keys
* `#22947 <https://github.com/numpy/numpy/pull/22947>`__: MAINT: Add additional information to missing scalar AttributeError
* `#22948 <https://github.com/numpy/numpy/pull/22948>`__: MAINT/DOC: refactor CircleCI config file
* `#22952 <https://github.com/numpy/numpy/pull/22952>`__: DOC: remove old LaTeX hack if Sphinx is at 5.0.0 or later (fix...
* `#22954 <https://github.com/numpy/numpy/pull/22954>`__: BUG, SIMD: Fix spurious invalid exception for sin/cos on arm64/clang
* `#22955 <https://github.com/numpy/numpy/pull/22955>`__: Update LICENSE.txt
* `#22959 <https://github.com/numpy/numpy/pull/22959>`__: BUG: Fix fill violating read-only flag.
* `#22960 <https://github.com/numpy/numpy/pull/22960>`__: DOC: Add example for np.ma.diag
* `#22962 <https://github.com/numpy/numpy/pull/22962>`__: MAINT: use pypy3.9 in testing
* `#22964 <https://github.com/numpy/numpy/pull/22964>`__: MAINT: Update python 3.11-dev to 3.11.
* `#22980 <https://github.com/numpy/numpy/pull/22980>`__: MAINT: Fix some noisy clang suggestions.
* `#22981 <https://github.com/numpy/numpy/pull/22981>`__: DOC: remove extraneous backtick from warning
* `#22982 <https://github.com/numpy/numpy/pull/22982>`__: ENH: add support for fujitsu C/C++ compiler and SSL2 to numpy.
* `#22986 <https://github.com/numpy/numpy/pull/22986>`__: BUG: Ensure correct loop order in sin, cos, and arctan2
* `#22988 <https://github.com/numpy/numpy/pull/22988>`__: DOC: add information about disabling SIMD for crashes
* `#22991 <https://github.com/numpy/numpy/pull/22991>`__: DOC: Fix gh-22990 by correcting docstring of result_type
* `#22996 <https://github.com/numpy/numpy/pull/22996>`__: ENH: Improve loadtxt error with dtype and non-matchinig column...
* `#22997 <https://github.com/numpy/numpy/pull/22997>`__: API: Fix cython exception handling for exported extern C functions
* `#22998 <https://github.com/numpy/numpy/pull/22998>`__: DEP: Finalize ``+arr`` returning a copy e.g. for string arrays
* `#23002 <https://github.com/numpy/numpy/pull/23002>`__: CI: Fix circleCI devdoc deploy path
* `#23004 <https://github.com/numpy/numpy/pull/23004>`__: CI: Fix CircleCI ssh key missing
* `#23010 <https://github.com/numpy/numpy/pull/23010>`__: BENCH: Add coverage for remaining array_api operations
* `#23011 <https://github.com/numpy/numpy/pull/23011>`__: DEP: deprecate np.finfo(None)
* `#23013 <https://github.com/numpy/numpy/pull/23013>`__: DOC: Fix a typo in f2py meson docs
* `#23015 <https://github.com/numpy/numpy/pull/23015>`__: DOC: Add version added information for the strict parameter in...
* `#23016 <https://github.com/numpy/numpy/pull/23016>`__: BUG: use ``_Alignof`` rather than ``offsetof()`` on most compilers
* `#23018 <https://github.com/numpy/numpy/pull/23018>`__: ENH: Convert methods to vectorcall conversions
* `#23019 <https://github.com/numpy/numpy/pull/23019>`__: DEP: Finalize the non-sequence stacking deprecation
* `#23020 <https://github.com/numpy/numpy/pull/23020>`__: ENH: Improve array function overhead by using vectorcall
* `#23023 <https://github.com/numpy/numpy/pull/23023>`__: BLD: Build PyPy 3.9 wheels.
* `#23026 <https://github.com/numpy/numpy/pull/23026>`__: CI: Bump debug test to ubuntu-latest/22.04 rather than 20.04
* `#23034 <https://github.com/numpy/numpy/pull/23034>`__: API: Allow SciPy to get away with assuming ``trapz`` is a Python...
* `#23038 <https://github.com/numpy/numpy/pull/23038>`__: DOC: Ignore attribute FutureWarning in doc build
* `#23039 <https://github.com/numpy/numpy/pull/23039>`__: BUG: Implement ``ArrayFunctionDispatcher.__get__``
* `#23041 <https://github.com/numpy/numpy/pull/23041>`__: MAINT: Remove all nose testing support.
* `#23045 <https://github.com/numpy/numpy/pull/23045>`__: DOC: Improved nbytes description
* `#23047 <https://github.com/numpy/numpy/pull/23047>`__: MAINT: dtype.name works for NEP 42 dtypes
* `#23054 <https://github.com/numpy/numpy/pull/23054>`__: MAINT: Move unused import into hook for pyinstaller
* `#23060 <https://github.com/numpy/numpy/pull/23060>`__: DEP: Remove the deprecated ``utils.py`` shim.
* `#23061 <https://github.com/numpy/numpy/pull/23061>`__: ENH: Enabled the use of numpy.vectorize as a decorator
* `#23064 <https://github.com/numpy/numpy/pull/23064>`__: DOC: Fixup docs after the code removal
* `#23065 <https://github.com/numpy/numpy/pull/23065>`__: DOC: fail the CI build and do not deploy if docs are not created
* `#23066 <https://github.com/numpy/numpy/pull/23066>`__: BUG: fix broken numpy.distutils Fortran handling
* `#23069 <https://github.com/numpy/numpy/pull/23069>`__: BLD: musllinux wheel build
* `#23073 <https://github.com/numpy/numpy/pull/23073>`__: CI: Rebase NumPy compiled extension test modules on Cygwin
* `#23076 <https://github.com/numpy/numpy/pull/23076>`__: DEV: Fix shell configuration in devcontainer
* `#23077 <https://github.com/numpy/numpy/pull/23077>`__: BUG: Fix for npyv__trunc_s32_f32 (VXE)
* `#23079 <https://github.com/numpy/numpy/pull/23079>`__: BUG: Fix ``integer / float`` scalar promotion
* `#23087 <https://github.com/numpy/numpy/pull/23087>`__: DOC, ENH: Improve docstring of real_if_close
* `#23088 <https://github.com/numpy/numpy/pull/23088>`__: ENH: Improve performance of finfo and _commonType
* `#23089 <https://github.com/numpy/numpy/pull/23089>`__: API: Add environment variable for behavior planned in a 2.0
* `#23090 <https://github.com/numpy/numpy/pull/23090>`__: BUG: Fix crash when using complex double scalars with NEP 50
* `#23092 <https://github.com/numpy/numpy/pull/23092>`__: DOC: fix typo in C API reference
* `#23093 <https://github.com/numpy/numpy/pull/23093>`__: BUG: Fixup f2py's handling a very little bit
* `#23094 <https://github.com/numpy/numpy/pull/23094>`__: BUG: fix type in resolve_descriptors_function typedef
* `#23097 <https://github.com/numpy/numpy/pull/23097>`__: BUILD: error when building with python<3.9 [skip ci]
* `#23098 <https://github.com/numpy/numpy/pull/23098>`__: BUG: Handle arrays in ``conf_data``
* `#23103 <https://github.com/numpy/numpy/pull/23103>`__: TYP: Fix return type to float on _FloatLike_co arguments
* `#23105 <https://github.com/numpy/numpy/pull/23105>`__: API: Raise EOFError when trying to load past the end of a ``.npy``...
* `#23109 <https://github.com/numpy/numpy/pull/23109>`__: ENH: Add some unit tests for finfo and iinfo
* `#23111 <https://github.com/numpy/numpy/pull/23111>`__: MAINT: Allow export/import of bools in dlpack
* `#23113 <https://github.com/numpy/numpy/pull/23113>`__: ENH: Add slots to NDArrayOperatorsMixin allowing them in subclasses
* `#23117 <https://github.com/numpy/numpy/pull/23117>`__: Re-submission of PR23006
* `#23121 <https://github.com/numpy/numpy/pull/23121>`__: DOC:fix type in bitwise_ior
* `#23124 <https://github.com/numpy/numpy/pull/23124>`__: BUILD: use GITHUB_REF_NAME in musllinux merge CI run [skip ci],...
* `#23127 <https://github.com/numpy/numpy/pull/23127>`__: BLD: use conda to install anaconda-client for upload
* `#23128 <https://github.com/numpy/numpy/pull/23128>`__: BUG: Add missing <type_traits> header.
* `#23129 <https://github.com/numpy/numpy/pull/23129>`__: TST: add tests for numpy.quantile
* `#23130 <https://github.com/numpy/numpy/pull/23130>`__: MAINT, BLD: Update wheel and GitPython versions.
* `#23136 <https://github.com/numpy/numpy/pull/23136>`__: ENH: create and use indexed inner loops
* `#23141 <https://github.com/numpy/numpy/pull/23141>`__: CI: reduce CI load - two fewer Azure Windows jobs, and no GHA...
* `#23142 <https://github.com/numpy/numpy/pull/23142>`__: NEP: add a paragraph about accredidation for sponsored work
* `#23143 <https://github.com/numpy/numpy/pull/23143>`__: ENH: Allow trivial pickling of user DType (classes)
* `#23144 <https://github.com/numpy/numpy/pull/23144>`__: TYP,MAINT: Add a missing explicit ``Any`` parameter to the ``npt.ArrayLike``...
* `#23145 <https://github.com/numpy/numpy/pull/23145>`__: TYP,MAINT: Remove typing-related Python <3.9 leftovers
* `#23153 <https://github.com/numpy/numpy/pull/23153>`__: SIMD: Get rid of attribute-based CPU dispatching
* `#23154 <https://github.com/numpy/numpy/pull/23154>`__: ENH: Allow using dtype classes in ``arr.astype()``
* `#23163 <https://github.com/numpy/numpy/pull/23163>`__: MAINT: Update main after 1.24.2 release.
* `#23164 <https://github.com/numpy/numpy/pull/23164>`__: BUG: Correct types, add missing functions to c_distributions.pxd
* `#23165 <https://github.com/numpy/numpy/pull/23165>`__: BLD: add a meson windows build
* `#23167 <https://github.com/numpy/numpy/pull/23167>`__: DOC: Add N-dimensional argmax/argmin example
* `#23168 <https://github.com/numpy/numpy/pull/23168>`__: CI: Update pyodide version in emscripten tests
* `#23172 <https://github.com/numpy/numpy/pull/23172>`__: DOC: LaTeX/PDF: add support for a few needed Chinese characters
* `#23173 <https://github.com/numpy/numpy/pull/23173>`__: ENH: Add PyArray_ArrFunc compare support for NEP42 dtypes
* `#23174 <https://github.com/numpy/numpy/pull/23174>`__: BUG: Fix Apple silicon builds by working around clang partial...
* `#23177 <https://github.com/numpy/numpy/pull/23177>`__: ENH: add indexed loops for maximum, minimum, fmax, fmin
* `#23179 <https://github.com/numpy/numpy/pull/23179>`__: BUG/ENH: Fix fast index loops for 1-el array / allow scalar value
* `#23181 <https://github.com/numpy/numpy/pull/23181>`__: ENH: enable fast indexed loops for complex add, subtract, multiply
* `#23183 <https://github.com/numpy/numpy/pull/23183>`__: CI: check env in wheel uploader
* `#23184 <https://github.com/numpy/numpy/pull/23184>`__: TST: Comment out spurious print in f2py test
* `#23186 <https://github.com/numpy/numpy/pull/23186>`__: MAINT: Fixup random bounds checking code
* `#23188 <https://github.com/numpy/numpy/pull/23188>`__: DOC: update release blurb
* `#23190 <https://github.com/numpy/numpy/pull/23190>`__: ENH: add ``_is_numeric`` attribute for DType classes
* `#23193 <https://github.com/numpy/numpy/pull/23193>`__: DOC: fix typo in overrides.py
* `#23194 <https://github.com/numpy/numpy/pull/23194>`__: BUG: fix for f2py string scalars
* `#23195 <https://github.com/numpy/numpy/pull/23195>`__: API: Add ``rng.spawn()``, ``bit_gen.spawn()``, and ``bit_gen.seed_seq``
* `#23200 <https://github.com/numpy/numpy/pull/23200>`__: Fix typos found by copdespell
* `#23201 <https://github.com/numpy/numpy/pull/23201>`__: BUG: datetime64/timedelta64 comparisons return NotImplemented
* `#23202 <https://github.com/numpy/numpy/pull/23202>`__: DOC: Fix typos found by codespell in NEPs
* `#23211 <https://github.com/numpy/numpy/pull/23211>`__: MAINT: Merge public and private dtype API as much as possible
* `#23212 <https://github.com/numpy/numpy/pull/23212>`__: DOC: Fix matplotlib error in documentation
* `#23216 <https://github.com/numpy/numpy/pull/23216>`__: CI: Ensure submodules are initialized on MacOS CI
* `#23223 <https://github.com/numpy/numpy/pull/23223>`__: STY: testing: Fix some whitespace and minor code issues in utils.py
* `#23224 <https://github.com/numpy/numpy/pull/23224>`__: CI: Ensure submodules are initialized in gitpod.
* `#23225 <https://github.com/numpy/numpy/pull/23225>`__: DOC: Add notation for skipping Cirrus to dev docs.
* `#23227 <https://github.com/numpy/numpy/pull/23227>`__: CI: Include x86-simd-sort in submodules in gitpod.Dockerfile
* `#23228 <https://github.com/numpy/numpy/pull/23228>`__: MAINT: add new simd_qsort files to .gitignore
* `#23229 <https://github.com/numpy/numpy/pull/23229>`__: BUG: Fix busday_count for reversed dates
* `#23232 <https://github.com/numpy/numpy/pull/23232>`__: TYP,MAINT: Add a missing explicit Any parameter to scalars
* `#23233 <https://github.com/numpy/numpy/pull/23233>`__: BUG: Use raw strings for paths ``__config__.py.in``
* `#23237 <https://github.com/numpy/numpy/pull/23237>`__: MAINT: Add debug information to ufunc wrapping error
* `#23239 <https://github.com/numpy/numpy/pull/23239>`__: DOC: update link for logo in the readme
* `#23240 <https://github.com/numpy/numpy/pull/23240>`__: ENH: Allow ``where`` argument to override ``__array_ufunc__``
* `#23241 <https://github.com/numpy/numpy/pull/23241>`__: DOC: Add release note for AVX-512 quicksort
* `#23248 <https://github.com/numpy/numpy/pull/23248>`__: ENH: Avoid use of item XINCREF and DECREF in fasttake
* `#23251 <https://github.com/numpy/numpy/pull/23251>`__: MAINT: Further removal of PyArray_Item_INCREF use
* `#23255 <https://github.com/numpy/numpy/pull/23255>`__: MAINT: Remove malloc(0) calls in linalg and pocketfft
* `#23256 <https://github.com/numpy/numpy/pull/23256>`__: TYP,MAINT: Add a missing explicit Any parameter
* `#23257 <https://github.com/numpy/numpy/pull/23257>`__: CI: clean up GHA boilerplate for ``[skip github]``
* `#23261 <https://github.com/numpy/numpy/pull/23261>`__: MAINT: use full flag name in nditer constructor error
* `#23262 <https://github.com/numpy/numpy/pull/23262>`__: API: expose traversal typedefs and ``get_clear_loop`` slot
* `#23264 <https://github.com/numpy/numpy/pull/23264>`__: BLD: set NDEBUG for release builds with Meson
* `#23269 <https://github.com/numpy/numpy/pull/23269>`__: BUG: masked array proper deepcopies
* `#23272 <https://github.com/numpy/numpy/pull/23272>`__: DOC: Fix wrong section title
* `#23275 <https://github.com/numpy/numpy/pull/23275>`__: ENH: Modify ``np.logspace`` so that the ``base`` argument broadcasts...
* `#23278 <https://github.com/numpy/numpy/pull/23278>`__: DOC: Fix some links in the usage document
* `#23279 <https://github.com/numpy/numpy/pull/23279>`__: BUG: Allow no-op clearing of void dtypes
* `#23280 <https://github.com/numpy/numpy/pull/23280>`__: MAINT: Use strong references/copies for sorting buffer
* `#23284 <https://github.com/numpy/numpy/pull/23284>`__: ENH: add support for xp.take
* `#23285 <https://github.com/numpy/numpy/pull/23285>`__: DOC: Update dtype hierarchy and box-text alignment in png pdf...
* `#23292 <https://github.com/numpy/numpy/pull/23292>`__: BUG: sorting checks ``NPY_NEEDS_PYAPI`` instead of ``NPY_ITEM_REFCOUNT``
* `#23294 <https://github.com/numpy/numpy/pull/23294>`__: Set github workflow permissions to circleci and labeler
* `#23295 <https://github.com/numpy/numpy/pull/23295>`__: ENH: show dtype in array repr when endianness is non-native
* `#23298 <https://github.com/numpy/numpy/pull/23298>`__: ENH: Extend the functionlty of C++ type ``np::Half``
* `#23302 <https://github.com/numpy/numpy/pull/23302>`__: DEP: deprecate ``np.round_``; add ``round``/``min``/``max`` to the docs
* `#23306 <https://github.com/numpy/numpy/pull/23306>`__: MAINT: allow NPY_DT_NUMERIC flag on user dtypes
* `#23310 <https://github.com/numpy/numpy/pull/23310>`__: DOC: fixed meshgrid descr and its return type
* `#23311 <https://github.com/numpy/numpy/pull/23311>`__: DOC: Fix a typo
* `#23314 <https://github.com/numpy/numpy/pull/23314>`__: DEP: deprecate ``product``, ``cumproduct``, ``sometrue``, ``alltrue``
* `#23317 <https://github.com/numpy/numpy/pull/23317>`__: DOC: fix typos
* `#23318 <https://github.com/numpy/numpy/pull/23318>`__: BUG: Fix reference counting error in arraydescr_new
* `#23321 <https://github.com/numpy/numpy/pull/23321>`__: MAINT: Fix failing gitpod build.
* `#23322 <https://github.com/numpy/numpy/pull/23322>`__: ENH: Add support to ``ma.dot()`` for non-2d arrays with ``strict=True``
* `#23325 <https://github.com/numpy/numpy/pull/23325>`__: TYP: Replace duplicate reduce in ufunc type signature with reduceat.
* `#23326 <https://github.com/numpy/numpy/pull/23326>`__: TYP: Remove duplicate CLIP/WRAP/RAISE in __init__.pyi.
* `#23327 <https://github.com/numpy/numpy/pull/23327>`__: TYP: Mark ``d`` argument to fftfreq and rfftfreq as optional...
* `#23328 <https://github.com/numpy/numpy/pull/23328>`__: TYP: Add type annotations for comparison operators to MaskedArray.
* `#23329 <https://github.com/numpy/numpy/pull/23329>`__: MAINT: Update actions in gitpod/docker workflows
* `#23332 <https://github.com/numpy/numpy/pull/23332>`__: MAINT: Make hadolint version more explicit.
* `#23333 <https://github.com/numpy/numpy/pull/23333>`__: MAINT: Ignore hadolint info error DL3059.
* `#23335 <https://github.com/numpy/numpy/pull/23335>`__: BUG: ma with structured dtype
* `#23337 <https://github.com/numpy/numpy/pull/23337>`__: ENH: Add support for argmin and argmax for NEP42 dtypes
* `#23339 <https://github.com/numpy/numpy/pull/23339>`__: MAIN: Apply security best practices
* `#23340 <https://github.com/numpy/numpy/pull/23340>`__: TYP: Remove two stray type-check-only re-exports of ``msort``
* `#23348 <https://github.com/numpy/numpy/pull/23348>`__: DOC: Clearer doc title for repeat
* `#23355 <https://github.com/numpy/numpy/pull/23355>`__: DOC: Document that info() handles array instances.
* `#23357 <https://github.com/numpy/numpy/pull/23357>`__: ENH: ``__repr__`` for NpzFile object
* `#23358 <https://github.com/numpy/numpy/pull/23358>`__: API: Add DType classes into new ``numpy.dtypes`` module
* `#23359 <https://github.com/numpy/numpy/pull/23359>`__: DOC: Add 'may vary' markup in info() docstring.
* `#23361 <https://github.com/numpy/numpy/pull/23361>`__: BLD: add SSL2 default path for Fujitsu C/C++ compiler
* `#23364 <https://github.com/numpy/numpy/pull/23364>`__: DEP: update deprecations for ``np.product`` and co to emit from...
* `#23368 <https://github.com/numpy/numpy/pull/23368>`__: DOC: docs added to routines.ma.rst from issue #23352
* `#23371 <https://github.com/numpy/numpy/pull/23371>`__: ENH: drop dtype metadata
* `#23372 <https://github.com/numpy/numpy/pull/23372>`__: BLD: Check for submodules before build
* `#23376 <https://github.com/numpy/numpy/pull/23376>`__: MAINT: remove ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION`` environment...
* `#23378 <https://github.com/numpy/numpy/pull/23378>`__: DOC: Add ``n_children`` param to rng.spawn() and bitgen.spawn()...
* `#23380 <https://github.com/numpy/numpy/pull/23380>`__: MAINT: Update x86-simd-sort to latest commit
* `#23382 <https://github.com/numpy/numpy/pull/23382>`__: DOC: fixing a typo in polynomial roots Doc & cross-referencing...
* `#23384 <https://github.com/numpy/numpy/pull/23384>`__: MAINT: cleanup unused Python3.8-only code and references
* `#23386 <https://github.com/numpy/numpy/pull/23386>`__: MAINT: Pin devpy
* `#23387 <https://github.com/numpy/numpy/pull/23387>`__: Bump pypa/cibuildwheel from 2.12.0 to 2.12.1
* `#23388 <https://github.com/numpy/numpy/pull/23388>`__: Bump ossf/scorecard-action from 2.0.6 to 2.1.2
* `#23389 <https://github.com/numpy/numpy/pull/23389>`__: Bump egor-tensin/cleanup-path from 1 to 3
* `#23390 <https://github.com/numpy/numpy/pull/23390>`__: Bump cygwin/cygwin-install-action from 2 to 3
* `#23391 <https://github.com/numpy/numpy/pull/23391>`__: Bump docker/setup-buildx-action from 2.4.1 to 2.5.0
* `#23392 <https://github.com/numpy/numpy/pull/23392>`__: DOC: Add Examples for np.ma.sort
* `#23393 <https://github.com/numpy/numpy/pull/23393>`__: DOC: Add Examples for np.ma.right_shift
* `#23395 <https://github.com/numpy/numpy/pull/23395>`__: Bump actions/cache from 3.2.6 to 3.3.1
* `#23396 <https://github.com/numpy/numpy/pull/23396>`__: Bump github/codeql-action from 2.2.5 to 2.2.7
* `#23397 <https://github.com/numpy/numpy/pull/23397>`__: Bump actions/dependency-review-action from 2.5.1 to 3.0.3
* `#23399 <https://github.com/numpy/numpy/pull/23399>`__: ENH: float64 sin/cos using Numpy intrinsics
* `#23403 <https://github.com/numpy/numpy/pull/23403>`__: DEP: remove deprecated casting in np.clip
* `#23404 <https://github.com/numpy/numpy/pull/23404>`__: ENH: allow using dtype classes in array creation functions
* `#23406 <https://github.com/numpy/numpy/pull/23406>`__: MAINT: Bump actions/checkout from 3.3.0 to 3.4.0
* `#23407 <https://github.com/numpy/numpy/pull/23407>`__: MAINT: Rename devpy to spin
* `#23408 <https://github.com/numpy/numpy/pull/23408>`__: DOC: fix extra space in error message
* `#23411 <https://github.com/numpy/numpy/pull/23411>`__: DEV: use micromamba to set up Codespaces
* `#23424 <https://github.com/numpy/numpy/pull/23424>`__: DOC: Clarifies usage of the np alias for numpy
* `#23426 <https://github.com/numpy/numpy/pull/23426>`__: DOC: Add example for np.ma.compressed.
* `#23427 <https://github.com/numpy/numpy/pull/23427>`__: Bump actions/dependency-review-action from 3.0.3 to 3.0.4
* `#23428 <https://github.com/numpy/numpy/pull/23428>`__: MAINT: Bump larsoner/circleci-artifacts-redirector-action from...
* `#23430 <https://github.com/numpy/numpy/pull/23430>`__: MAINT: Fix 'tranform' and 'paramter' typos
* `#23435 <https://github.com/numpy/numpy/pull/23435>`__: ENH: Use AVX512 FP16 ISA for sorting float16 arrays
* `#23436 <https://github.com/numpy/numpy/pull/23436>`__: DOC: Remove descriptions in method declarations
* `#23437 <https://github.com/numpy/numpy/pull/23437>`__: DOC: fix broken link to SciPy Tutorial
* `#23438 <https://github.com/numpy/numpy/pull/23438>`__: DOC: fix a link redirect to WSL
* `#23439 <https://github.com/numpy/numpy/pull/23439>`__: MAINT: Bump github/codeql-action from 2.2.7 to 2.2.8
* `#23440 <https://github.com/numpy/numpy/pull/23440>`__: BUG: accept zeros on numpy.random dirichlet function
* `#23441 <https://github.com/numpy/numpy/pull/23441>`__: MAINT: Update spin
* `#23444 <https://github.com/numpy/numpy/pull/23444>`__: MAINT: remove support for Gitpod, add note on using Codespaces
* `#23445 <https://github.com/numpy/numpy/pull/23445>`__: MAINT: Fix a broken section link
* `#23447 <https://github.com/numpy/numpy/pull/23447>`__: Bump actions/checkout from 3.4.0 to 3.5.0
* `#23448 <https://github.com/numpy/numpy/pull/23448>`__: MAINT Fix broken links in absoft.py
* `#23449 <https://github.com/numpy/numpy/pull/23449>`__: MAINT: Stop dependabot docker checks.
* `#23450 <https://github.com/numpy/numpy/pull/23450>`__: MAINT: Fix reference roles of ``ast``
* `#23451 <https://github.com/numpy/numpy/pull/23451>`__: DOC: add entry for ``numpy.character``
* `#23453 <https://github.com/numpy/numpy/pull/23453>`__: DOC: typo, remove unfollowed conjunction "However" [skip ci]
* `#23455 <https://github.com/numpy/numpy/pull/23455>`__: CI: Ensure coverage is run for merges
* `#23456 <https://github.com/numpy/numpy/pull/23456>`__: BUILD: add ci run after merge for coverage
* `#23458 <https://github.com/numpy/numpy/pull/23458>`__: TST: Fix failing test (introduced deprecation after written)
* `#23459 <https://github.com/numpy/numpy/pull/23459>`__: BUG: Use output when given on numpy.dot C-API branch
* `#23460 <https://github.com/numpy/numpy/pull/23460>`__: MAINT: Fix ReDOS vulnerability in crackfortran.py
* `#23465 <https://github.com/numpy/numpy/pull/23465>`__: DOC: typo, fixed "Howeve" to "However" [skip ci]
* `#23466 <https://github.com/numpy/numpy/pull/23466>`__: Revert "ENH: Enabled the use of numpy.vectorize as a decorator"
* `#23467 <https://github.com/numpy/numpy/pull/23467>`__: BUG: in the fastest path form ufunc.at, properly increment args[2]
* `#23470 <https://github.com/numpy/numpy/pull/23470>`__: BUG: Fix bug in parsing F77 style string arrays.
* `#23471 <https://github.com/numpy/numpy/pull/23471>`__: DOC: typo, fixed PyArary to PyArray
* `#23472 <https://github.com/numpy/numpy/pull/23472>`__: MAINT: Bump github/codeql-action from 2.2.8 to 2.2.9
* `#23473 <https://github.com/numpy/numpy/pull/23473>`__: DOC: Fix a wrong format of reference
* `#23474 <https://github.com/numpy/numpy/pull/23474>`__: MAINT: Fix missing asterisk
* `#23480 <https://github.com/numpy/numpy/pull/23480>`__: DEP: remove deprecated ``numpy.dual`` module
* `#23484 <https://github.com/numpy/numpy/pull/23484>`__: MAINT: use PyArray_ClearBuffer in PyArray_FillWithScalar
* `#23489 <https://github.com/numpy/numpy/pull/23489>`__: DOC: Fixing incorrect sentence - Boolean array indexing #23377
* `#23491 <https://github.com/numpy/numpy/pull/23491>`__: DOC: Removed ``.shape`` setting note from reshape
* `#23495 <https://github.com/numpy/numpy/pull/23495>`__: BUG: fix loading and storing big arrays on s390x
* `#23498 <https://github.com/numpy/numpy/pull/23498>`__: MAINT: improve error when a dtype doesn't support the buffer...
* `#23501 <https://github.com/numpy/numpy/pull/23501>`__: MAINT: Bump ossf/scorecard-action from 2.1.2 to 2.1.3
* `#23502 <https://github.com/numpy/numpy/pull/23502>`__: CI: Add CI to test using gcc-12 on Intel Sapphire Rapids
* `#23503 <https://github.com/numpy/numpy/pull/23503>`__: DOC: fixed typos & rectified grammar
* `#23504 <https://github.com/numpy/numpy/pull/23504>`__: DOC: Remove doc for non-existing ``PyArray_XDECREF_ERR``
* `#23505 <https://github.com/numpy/numpy/pull/23505>`__: BUG: Use 2GiB chunking code for fwrite() on mingw32/64
* `#23507 <https://github.com/numpy/numpy/pull/23507>`__: DOC: default integer dtype reflect C long size
* `#23509 <https://github.com/numpy/numpy/pull/23509>`__: DOC: Fix a structure in ``NPY_TYPES``
* `#23514 <https://github.com/numpy/numpy/pull/23514>`__: ENH: Enabled the use of numpy.vectorize as a decorator
* `#23515 <https://github.com/numpy/numpy/pull/23515>`__: CI: Update to Pyodide 0.23.1
* `#23516 <https://github.com/numpy/numpy/pull/23516>`__: DOC: add missing punctuation in a C API .rst file
* `#23519 <https://github.com/numpy/numpy/pull/23519>`__: DOC: Fix a reference to built-in ``len`` in ``char.str_len`` docstring
* `#23521 <https://github.com/numpy/numpy/pull/23521>`__: ENH: Raise C++ standard to C++17
* `#23528 <https://github.com/numpy/numpy/pull/23528>`__: ENH: Allow, and default to, downstream building with old API
* `#23529 <https://github.com/numpy/numpy/pull/23529>`__: DOC: quantile parameter q is a probability
* `#23530 <https://github.com/numpy/numpy/pull/23530>`__: DOC: Add example for Polynomial.degree().
* `#23535 <https://github.com/numpy/numpy/pull/23535>`__: MAINT: Bump github/codeql-action from 2.2.9 to 2.2.10
* `#23542 <https://github.com/numpy/numpy/pull/23542>`__: MAINT: Bump github/codeql-action from 2.2.10 to 2.2.11
* `#23549 <https://github.com/numpy/numpy/pull/23549>`__: DOC: Fix incorrect operators for member access in ``array.rst``
* `#23551 <https://github.com/numpy/numpy/pull/23551>`__: BLD: fix instances of MSVC detection in ``meson.build``
* `#23553 <https://github.com/numpy/numpy/pull/23553>`__: DOC: Fix description of ``PyArray_DiscardWritebackIfCopy``
* `#23554 <https://github.com/numpy/numpy/pull/23554>`__: MAINT: Bump cygwin/cygwin-install-action from 3 to 4
* `#23555 <https://github.com/numpy/numpy/pull/23555>`__: DEP: deprecate np.math and np.lib.math
* `#23557 <https://github.com/numpy/numpy/pull/23557>`__: ENH: Use threshold also inside SubArrayFormat.
* `#23558 <https://github.com/numpy/numpy/pull/23558>`__: DOC: Fix missing punctuation in ``array.rst``
* `#23559 <https://github.com/numpy/numpy/pull/23559>`__: BLD: add static to std_c_flags program. Move macosx_arm64 wheel...
* `#23560 <https://github.com/numpy/numpy/pull/23560>`__: DOC: Fix parameter type of ``PyArray_DiscardWritebackIfCopy``
* `#23562 <https://github.com/numpy/numpy/pull/23562>`__: DOC: Fix document structure of ``PyUfuncObject``
* `#23565 <https://github.com/numpy/numpy/pull/23565>`__: DOC: Add directive for C-type ``PyArrayMapIterObject``
* `#23566 <https://github.com/numpy/numpy/pull/23566>`__: CI: disable intel_spr_sde_test for now
* `#23568 <https://github.com/numpy/numpy/pull/23568>`__: MAINT: remove usages of sys.exc_info
* `#23569 <https://github.com/numpy/numpy/pull/23569>`__: TST: try readding test_new_policy on musl
* `#23571 <https://github.com/numpy/numpy/pull/23571>`__: MAINT: Bump larsoner/circleci-artifacts-redirector-action
* `#23572 <https://github.com/numpy/numpy/pull/23572>`__: MAINT: Bump larsoner/circleci-artifacts-redirector-action
* `#23573 <https://github.com/numpy/numpy/pull/23573>`__: MAINT: Bump actions/checkout from 3.5.0 to 3.5.1
* `#23578 <https://github.com/numpy/numpy/pull/23578>`__: CI: fix Circle CI doc build html preview link
* `#23579 <https://github.com/numpy/numpy/pull/23579>`__: NEP: Create draft NEP for C-API evolution
* `#23581 <https://github.com/numpy/numpy/pull/23581>`__: CI: .cirrus.star typo
* `#23584 <https://github.com/numpy/numpy/pull/23584>`__: DOC: Remove descriptions of non-existent C-types
* `#23585 <https://github.com/numpy/numpy/pull/23585>`__: DOC: np.random index reorganization
* `#23587 <https://github.com/numpy/numpy/pull/23587>`__: DOC: Fix module name of ``autoattribute`` in maskedarray document
* `#23589 <https://github.com/numpy/numpy/pull/23589>`__: MAINT: Bump actions/checkout from 3.5.1 to 3.5.2
* `#23590 <https://github.com/numpy/numpy/pull/23590>`__: MAINT: Bump github/codeql-action from 2.2.11 to 2.2.12
* `#23591 <https://github.com/numpy/numpy/pull/23591>`__: ENH: refactor zero-filling and expose dtype API slot for it
* `#23593 <https://github.com/numpy/numpy/pull/23593>`__: BUG: lib: Tiny fix for the loadtxt tokenizer when PyMem_Malloc()...
* `#23594 <https://github.com/numpy/numpy/pull/23594>`__: DOC: Refactor description of ``PyArray_Descr``
* `#23596 <https://github.com/numpy/numpy/pull/23596>`__: MAINT: Update conftest for newer hypothesis versions
* `#23597 <https://github.com/numpy/numpy/pull/23597>`__: MAINT: Fix broken links in site.cfg.example
* `#23599 <https://github.com/numpy/numpy/pull/23599>`__: BUG: Fix crackfortran groups for endifs with comments
* `#23600 <https://github.com/numpy/numpy/pull/23600>`__: BUG: Infer return types for Fortran functions in ``f2py``
* `#23601 <https://github.com/numpy/numpy/pull/23601>`__: BLD: use the C++ linker to link ``_multiarray_umath.so``
* `#23602 <https://github.com/numpy/numpy/pull/23602>`__: BUG: Add packaging to benchmark dependencies
* `#23604 <https://github.com/numpy/numpy/pull/23604>`__: BUG, BLD: Fix indentation bug in distutils
* `#23605 <https://github.com/numpy/numpy/pull/23605>`__: DOC: pull tags and initialize submodules in development install...
* `#23614 <https://github.com/numpy/numpy/pull/23614>`__: DOC: state an other requirement to build a .f90 based module
* `#23615 <https://github.com/numpy/numpy/pull/23615>`__: MAINT: Bump pypa/cibuildwheel from 2.12.1 to 2.12.3
* `#23617 <https://github.com/numpy/numpy/pull/23617>`__: DOC: Use correct fill_value instead of value for np.full in 1.24.0...
* `#23620 <https://github.com/numpy/numpy/pull/23620>`__: MAINT: Add a proper implementation for structured zerofill
* `#23621 <https://github.com/numpy/numpy/pull/23621>`__: MAINT: Bump actions/setup-python from 4.5.0 to 4.6.0
* `#23623 <https://github.com/numpy/numpy/pull/23623>`__: DOC: Fix incorrect structure in ``sqrt`` docstring
* `#23626 <https://github.com/numpy/numpy/pull/23626>`__: BUG: Fix masked array raveling when ``order="A"`` or ``order="K"``
* `#23627 <https://github.com/numpy/numpy/pull/23627>`__: BUG: Ignore invalid and overflow warnings in masked setitem
* `#23631 <https://github.com/numpy/numpy/pull/23631>`__: DOC: Clarify that defining NPY_NO_DEPRECATED_API does not determine...
* `#23639 <https://github.com/numpy/numpy/pull/23639>`__: DOC: Fix incorrectly formatted roles in c-api document
* `#23640 <https://github.com/numpy/numpy/pull/23640>`__: DOC: Downgrade sphinx version to 5
* `#23643 <https://github.com/numpy/numpy/pull/23643>`__: DOC: Convert titles to sentence case
* `#23644 <https://github.com/numpy/numpy/pull/23644>`__: MAINT: Update main after 1.24.3 release.
* `#23648 <https://github.com/numpy/numpy/pull/23648>`__: DOC: Example on how to use np.lib.tracemalloc_domain.
* `#23650 <https://github.com/numpy/numpy/pull/23650>`__: DOC: Corrects scalar string documentation in regards to trailing...
* `#23652 <https://github.com/numpy/numpy/pull/23652>`__: ENH: structured_to_unstructured: view more often
* `#23654 <https://github.com/numpy/numpy/pull/23654>`__: BUG: Avoid uses -Werror during tests default C/C++ standards
* `#23655 <https://github.com/numpy/numpy/pull/23655>`__: CI: Enable CI on gcc 12.x on Intel SPR
* `#23656 <https://github.com/numpy/numpy/pull/23656>`__: MAINT: Bump github/codeql-action from 2.2.12 to 2.3.0
* `#23657 <https://github.com/numpy/numpy/pull/23657>`__: DOC: Improve description of skip commands for CI
* `#23658 <https://github.com/numpy/numpy/pull/23658>`__: MAINT: allow sphinx6, sync with conda environment.yml
* `#23659 <https://github.com/numpy/numpy/pull/23659>`__: ENH: Restore TypeError cleanup in array function dispatching
* `#23660 <https://github.com/numpy/numpy/pull/23660>`__: DEP: Finalize checking for sequence-like if something is array-like
* `#23661 <https://github.com/numpy/numpy/pull/23661>`__: ENH: add __contains__() to np.lib.npyio.NpzFile
* `#23662 <https://github.com/numpy/numpy/pull/23662>`__: TST: Remove crackfortran.nameargspattern time test that failed...
* `#23665 <https://github.com/numpy/numpy/pull/23665>`__: ENH: Speed up 64-bit qsort by 1.6x
* `#23666 <https://github.com/numpy/numpy/pull/23666>`__: DEP,BUG: Finalize subarray dtype FutureWarning and fix its assignment
* `#23668 <https://github.com/numpy/numpy/pull/23668>`__: DOC: add release note for npzfile membership test
* `#23676 <https://github.com/numpy/numpy/pull/23676>`__: MAINT: Bump github/codeql-action from 2.3.0 to 2.3.1
* `#23677 <https://github.com/numpy/numpy/pull/23677>`__: MAINT: bump website theme version
* `#23678 <https://github.com/numpy/numpy/pull/23678>`__: MAINT: Pin rtools version on Windows.
* `#23680 <https://github.com/numpy/numpy/pull/23680>`__: BUG: Fix masked array ravel order for A (and somewhat K)
* `#23682 <https://github.com/numpy/numpy/pull/23682>`__: CI: Disable intel_spr_sde_test, again
* `#23683 <https://github.com/numpy/numpy/pull/23683>`__: MAINT: Bump github/codeql-action from 2.3.1 to 2.3.2
* `#23684 <https://github.com/numpy/numpy/pull/23684>`__: DOC: Fix return type of ``PyArray_EinsteinSum``
* `#23691 <https://github.com/numpy/numpy/pull/23691>`__: BUG: random: Don't return negative values from Generator.geometric.
* `#23695 <https://github.com/numpy/numpy/pull/23695>`__: BUG: Correct sin/cos float64 range check functions
* `#23696 <https://github.com/numpy/numpy/pull/23696>`__: DOC: Fix link to site.cfg.example
* `#23699 <https://github.com/numpy/numpy/pull/23699>`__: MAINT: refactor PyArray_Repeat to avoid PyArray_INCREF
* `#23700 <https://github.com/numpy/numpy/pull/23700>`__: DOC: Resolve length/index ambiguity in numpy.outer docstring
* `#23705 <https://github.com/numpy/numpy/pull/23705>`__: MAINT: Reorganize the way windowing functions ensure float64...
* `#23706 <https://github.com/numpy/numpy/pull/23706>`__: MAINT: Remove gisnan, gisinf, and gisfinite from testing code
* `#23707 <https://github.com/numpy/numpy/pull/23707>`__: ENH: speed up 32-bit and 64-bit np.argsort by 5x with AVX-512
* `#23709 <https://github.com/numpy/numpy/pull/23709>`__: MAINT: Add "noexcept" markers to Cython functions that do not...
* `#23711 <https://github.com/numpy/numpy/pull/23711>`__: BUG: Fix compilation of halffloat with gcc 13.1
* `#23713 <https://github.com/numpy/numpy/pull/23713>`__: ENH: Make signed/unsigned integer comparisons exact
* `#23714 <https://github.com/numpy/numpy/pull/23714>`__: CI: Enable CI to build with gcc-12
* `#23715 <https://github.com/numpy/numpy/pull/23715>`__: MAINT, BLD: Install rtools 4.0 for Windows wheels.
* `#23716 <https://github.com/numpy/numpy/pull/23716>`__: BUG: Cannot mix ``uses`` and ``run`` in workflow script
* `#23721 <https://github.com/numpy/numpy/pull/23721>`__: BLD: update to OpenBLAS 0.3.23
* `#23723 <https://github.com/numpy/numpy/pull/23723>`__: MAINT: Do not use copyswap as part of indexing
* `#23724 <https://github.com/numpy/numpy/pull/23724>`__: MAINT: Bump actions/upload-artifact from 3.1.0 to 3.1.2
* `#23725 <https://github.com/numpy/numpy/pull/23725>`__: MAINT: Bump github/codeql-action from 2.3.2 to 2.3.3
* `#23726 <https://github.com/numpy/numpy/pull/23726>`__: DOC: fix incorrect description of raise condition in numpy.testing.assert_array_less's docstring
* `#23727 <https://github.com/numpy/numpy/pull/23727>`__: DOC: Rewrite docstrings of ``ogrid`` and ``mgrid``
* `#23728 <https://github.com/numpy/numpy/pull/23728>`__: BUG: fix the method for checking local files
* `#23734 <https://github.com/numpy/numpy/pull/23734>`__: Update index.rst #23732
* `#23735 <https://github.com/numpy/numpy/pull/23735>`__: TYP: Update type annotations for the numpy 1.25 release
* `#23736 <https://github.com/numpy/numpy/pull/23736>`__: MAINT: Copy rtools installation from install-rtools.
* `#23740 <https://github.com/numpy/numpy/pull/23740>`__: BLD: updates to the Meson build
* `#23742 <https://github.com/numpy/numpy/pull/23742>`__: BUG: Fix NEP 50 promotion in some concat/choose paths
* `#23743 <https://github.com/numpy/numpy/pull/23743>`__: DOC: Add symbol in docstring for classes derived from ABCPolyBase
* `#23746 <https://github.com/numpy/numpy/pull/23746>`__: ENH: add fast path for str(scalar_int)
* `#23747 <https://github.com/numpy/numpy/pull/23747>`__: DOC: clarify differences between ravel and reshape(-1)
* `#23750 <https://github.com/numpy/numpy/pull/23750>`__: MAINT: do not use copyswapn in array sorting internals
* `#23753 <https://github.com/numpy/numpy/pull/23753>`__: MAINT, BLD: Disable spr for clang
* `#23756 <https://github.com/numpy/numpy/pull/23756>`__: DOC: Rm bool8 from docs
* `#23757 <https://github.com/numpy/numpy/pull/23757>`__: BUG: properly handle tuple keys in NpZFile.__getitem__
* `#23758 <https://github.com/numpy/numpy/pull/23758>`__: MAINT: compatibility with cython3
* `#23763 <https://github.com/numpy/numpy/pull/23763>`__: BUG: Fix weak scalar logic for large ints in ufuncs
* `#23765 <https://github.com/numpy/numpy/pull/23765>`__: MAINT: fix signed/unsigned int comparison warnings
* `#23768 <https://github.com/numpy/numpy/pull/23768>`__: TYP: Relax the ``genfromtxt`` return dtype when the dtype is unspecified
* `#23769 <https://github.com/numpy/numpy/pull/23769>`__: DOC: clarify how inputs of PyArray_ResultType are used
* `#23770 <https://github.com/numpy/numpy/pull/23770>`__: MAINT: do not use copyswap in where internals
* `#23772 <https://github.com/numpy/numpy/pull/23772>`__: DOC: added note for git tagging
* `#23774 <https://github.com/numpy/numpy/pull/23774>`__: BUG: Fix median and quantile NaT handling
* `#23776 <https://github.com/numpy/numpy/pull/23776>`__: ENH: replace depcrecated distutils use in convenience function
* `#23777 <https://github.com/numpy/numpy/pull/23777>`__: Speed up _string_helpers init.
* `#23779 <https://github.com/numpy/numpy/pull/23779>`__: DOC: Fix link from mtrand.randint -> Generator.integers.
* `#23790 <https://github.com/numpy/numpy/pull/23790>`__: DOC: used a more readable if/else form than old-style ternary...
* `#23792 <https://github.com/numpy/numpy/pull/23792>`__: CI: Upload nighlighties to new location
* `#23794 <https://github.com/numpy/numpy/pull/23794>`__: API: add aliases for object and void dtype classes to dtype API
* `#23797 <https://github.com/numpy/numpy/pull/23797>`__: TST: adjust tests for cython3, pypy
* `#23802 <https://github.com/numpy/numpy/pull/23802>`__: MAINT: Bump actions/setup-python from 4.6.0 to 4.6.1
* `#23840 <https://github.com/numpy/numpy/pull/23840>`__: REL: Prepare for the NumPy 1.25.0rc1 release
* `#23872 <https://github.com/numpy/numpy/pull/23872>`__: DOC: Try to clarify the `NPY_TARGET_VERSION` release note
* `#23875 <https://github.com/numpy/numpy/pull/23875>`__: MAINT: Update ``download-wheels``.
* `#23876 <https://github.com/numpy/numpy/pull/23876>`__: NEP: Fix NEP 53 file format and minor formatting issue
* `#23877 <https://github.com/numpy/numpy/pull/23877>`__: DOC: update distutils migration guide
* `#23882 <https://github.com/numpy/numpy/pull/23882>`__: TST: Add tests for np.argsort (#23846)
* `#23888 <https://github.com/numpy/numpy/pull/23888>`__: TYP,DOC: Annotate and document the ``metadata`` parameter of...
* `#23905 <https://github.com/numpy/numpy/pull/23905>`__: MAINT: Use ``--allow-downgrade`` option for rtools.
* `#23932 <https://github.com/numpy/numpy/pull/23932>`__: BUG: Allow np.info on non-hashable objects with a dtype
* `#23933 <https://github.com/numpy/numpy/pull/23933>`__: MAINT: Disable SIMD version of float64 sin and cos
* `#23945 <https://github.com/numpy/numpy/pull/23945>`__: BUG: Fixup for win64 fwrite issue
* `#23946 <https://github.com/numpy/numpy/pull/23946>`__: BUG: Fix NpyIter cleanup in einsum error path
* `#23948 <https://github.com/numpy/numpy/pull/23948>`__: DOC: Update required C++ version in building.rst (and copy-edit).
* `#23949 <https://github.com/numpy/numpy/pull/23949>`__: BUG:Fix for call to 'vec_st' is ambiguous
* `#23951 <https://github.com/numpy/numpy/pull/23951>`__: MAINT: Upgrade install-rtools version
* `#23955 <https://github.com/numpy/numpy/pull/23955>`__: ENH: Add array API standard v2022.12 support to numpy.array_api
* `#23956 <https://github.com/numpy/numpy/pull/23956>`__: BUG: Fix AVX2 intrinsic npyv_store2_till_s64 on MSVC > 19.29
* `#23957 <https://github.com/numpy/numpy/pull/23957>`__: ENH: add copy parameter for api.reshape function
* `#23963 <https://github.com/numpy/numpy/pull/23963>`__: TEST: change subprocess call to capture stderr too
