
Contributors
============

A total of 106 people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.

* !Dreamge +
* !bersbersbers +
* !fengluoqiuwu +
* !h-vetinari
* !hutauf +
* !musvaage +
* !nullSoup +
* Aarni Koskela +
* Abhishek Kumar +
* Abraham Medina +
* Aditi Juneja +
* Adrien Corenflos +
* Agriya Khetarpal
* Ajay Kumar Janapareddi
* Akula Guru Datta +
* Amit Subhash Chejara +
* Andrew Nelson
* Anne Gunn
* Austin Ran +
* Ben Walsh
* Benjamin A. Beasley +
* Benoit Prabel +
* Charles Harris
* Chris Fu (傅立业)
* Chris Sidebottom
* Christian Lorentzen
* Christopher Sidebottom
* Clément Robert
* Dane Reimers +
* Dimitri Papadopoulos Orfanos
* Evgeni Burovski
* GUAN MING
* Habiba Hye +
* Harry Zhang +
* Hugo van Kemenade
* Ian Harris +
* Isaac Warren +
* Ishan Koradia +
* Ishan Purekar +
* Jake VanderPlas
* Jianyu Wen +
* Johannes Kaisinger
* John Kirkham
* Joren Hammudoglu
* João Eiras +
* KM Khalid Saifullah +
* Karel Planken +
* Katie Rust +
* Khem Raj
* Kira Prokopenko +
* Lars Grüter
* Linus Sommer
* Lucas Colley
* Luiz Eduardo Amaral
* Luke Aarohi +
* Marcel Telka +
* Mark Harfouche
* Marten van Kerkwijk
* Maryanne Wachter +
* Mateusz Sokół
* Matt Haberland
* Matthias Diener +
* Matthieu Darbois
* Matti Picus
* Maximilian Weigand +
* Melissa Weber Mendonça
* Michael Davidsaver +
* Nathan Goldbaum
* Nicolas Tessore +
* Nitish Satyavolu +
* Oscar Armas-Luy +
* Peter Hawkins
* Peter Kämpf +
* Pieter Eendebak
* Raghu Rajan +
* Raghuveer Devulapalli
* Ralf Gommers
* Robert Kern
* Rohit Goswami
* Ross Barnowski
* Ryan Teoh +
* Santhana Mikhail Antony S +
* Sayed Adel
* Sebastian Berg
* Sebastian Vittersø +
* Sebin Thomas +
* Serge Panev +
* Shaurya Barkund +
* Shiv Katira +
* Simon Altrogge
* Slava Gorloff +
* Slobodan Miletic +
* Soutrik Bandyopadhyay +
* Stan Ulbrych +
* Stefan van der Walt
* Tim Hoffmann
* Timo Röhling
* Tyler Reddy
* Vahid Tavanashad +
* Victor Herdeiro +
* Vijayakumar Z +
* Warren Weckesser
* Xiao Yuan +
* Yashasvi Misra
* bilderbuchi +
* dependabot[bot]

Pull requests merged
====================

A total of 317 pull requests were merged for this release.

* `#14622 <https://github.com/numpy/numpy/pull/14622>`__: BUG: fix datetime64/timedelta64 hash and match Python
* `#15181 <https://github.com/numpy/numpy/pull/15181>`__: ENH: Add nd-support to trim_zeros
* `#17780 <https://github.com/numpy/numpy/pull/17780>`__: ENH, BLD: Define RISCV-32 support
* `#23547 <https://github.com/numpy/numpy/pull/23547>`__: DOC: Fix a typo in description and add an example of ``numpy.tensordot``
* `#25984 <https://github.com/numpy/numpy/pull/25984>`__: BUG: Allow fitting of degree zero polynomials with Polynomial.fit
* `#26398 <https://github.com/numpy/numpy/pull/26398>`__: DOC: order of indices returned in tril_indices and triu_indices
* `#26406 <https://github.com/numpy/numpy/pull/26406>`__: DOC: Changed vdot docs as suggested
* `#26570 <https://github.com/numpy/numpy/pull/26570>`__: CI, BLD: Use ``cibuildwheel`` to build WASM NumPy wheels
* `#26642 <https://github.com/numpy/numpy/pull/26642>`__: DOC: Add examples to ``np.char``
* `#26855 <https://github.com/numpy/numpy/pull/26855>`__: TYP: improved ``numpy.frompyfunc`` type hints
* `#26857 <https://github.com/numpy/numpy/pull/26857>`__: MAINT: Start applying ruff/Pycodestyle rules
* `#26865 <https://github.com/numpy/numpy/pull/26865>`__: TYP: add missing annotations for ``numpy.object_.__new__``
* `#26941 <https://github.com/numpy/numpy/pull/26941>`__: TYP: Non-distributive ``numpy.generic`` type args.
* `#26944 <https://github.com/numpy/numpy/pull/26944>`__: TYP: Annotate ``numpy._core._type_aliases`` .
* `#26979 <https://github.com/numpy/numpy/pull/26979>`__: TYP: Explicit ``numpy.__all__`` in the stubs
* `#26994 <https://github.com/numpy/numpy/pull/26994>`__: TYP: Typing fixes for ``numpy.iinfo`` & ``numpy.finfo``
* `#27049 <https://github.com/numpy/numpy/pull/27049>`__: BUG: f2py: better handle filtering of public/private subroutines
* `#27088 <https://github.com/numpy/numpy/pull/27088>`__: WHL: bump (musl) linux image [wheel build]
* `#27100 <https://github.com/numpy/numpy/pull/27100>`__: TYP: Fixed & improved type hints for ``numpy.histogram2d``
* `#27101 <https://github.com/numpy/numpy/pull/27101>`__: TST, DOC: add doc and test for transpose axes with negative indices
* `#27116 <https://github.com/numpy/numpy/pull/27116>`__: DOC: update NEP 50 draft status to "Final"
* `#27119 <https://github.com/numpy/numpy/pull/27119>`__: ENH: Use ``PyObject_GetOptionalAttr``
* `#27132 <https://github.com/numpy/numpy/pull/27132>`__: TYP: Assume that ``typing_extensions`` is always available in...
* `#27134 <https://github.com/numpy/numpy/pull/27134>`__: REL: Prepare main for 2.2.0 development
* `#27139 <https://github.com/numpy/numpy/pull/27139>`__: TYP: Fixed & improved ``numpy.dtype.__new__``
* `#27140 <https://github.com/numpy/numpy/pull/27140>`__: MAINT: Scipy openblas 0.3.27.44.4
* `#27143 <https://github.com/numpy/numpy/pull/27143>`__: BUG: Do not accidentally store dtype metadata in ``np.save``
* `#27145 <https://github.com/numpy/numpy/pull/27145>`__: ENH: fix thread-unsafe C API usages
* `#27147 <https://github.com/numpy/numpy/pull/27147>`__: BLD: use smaller scipy-openblas builds
* `#27148 <https://github.com/numpy/numpy/pull/27148>`__: BUG: Raise if histogram cannot create finite bin sizes
* `#27150 <https://github.com/numpy/numpy/pull/27150>`__: TYP: Sane defaults for the platform-specific ``NBitBase`` types.
* `#27152 <https://github.com/numpy/numpy/pull/27152>`__: TYP: Simplified ufunc imports in ``numpy._typing``
* `#27153 <https://github.com/numpy/numpy/pull/27153>`__: TYP: Fix incompatible overrides in the ``numpy._typing._ufunc``...
* `#27154 <https://github.com/numpy/numpy/pull/27154>`__: TYP: Use ``typing_extensions.Self`` in the ``numpy`` stubs
* `#27156 <https://github.com/numpy/numpy/pull/27156>`__: MAINT: Remove any promotion-state switching logic
* `#27157 <https://github.com/numpy/numpy/pull/27157>`__: TYP: add td64 overload for ``np.mean``
* `#27158 <https://github.com/numpy/numpy/pull/27158>`__: CI: Re-enable nightly OpenBLAS test runs
* `#27160 <https://github.com/numpy/numpy/pull/27160>`__: DEP: Finalize ``bool(empty_array)`` deprecation
* `#27164 <https://github.com/numpy/numpy/pull/27164>`__: MAINT: use npy_argparse for einsum
* `#27168 <https://github.com/numpy/numpy/pull/27168>`__: DOC: add td64 example in ``np.mean``
* `#27171 <https://github.com/numpy/numpy/pull/27171>`__: TYP: Shape-typed array constructors: ``numpy.{empty,zeros,ones,full}``
* `#27177 <https://github.com/numpy/numpy/pull/27177>`__: TYP: 1-d ``numpy.arange`` return shape-type
* `#27178 <https://github.com/numpy/numpy/pull/27178>`__: TYP,TST: Bump mypy to 1.11.1
* `#27179 <https://github.com/numpy/numpy/pull/27179>`__: TYP: Improved ``numpy.piecewise`` type-hints
* `#27182 <https://github.com/numpy/numpy/pull/27182>`__: REV: Revert undef I and document it
* `#27184 <https://github.com/numpy/numpy/pull/27184>`__: BUILD: update to OpenBLAS 0.3.28
* `#27187 <https://github.com/numpy/numpy/pull/27187>`__: MAINT: update default NPY_FEATURE_VERSION after dropping py39
* `#27189 <https://github.com/numpy/numpy/pull/27189>`__: MAINT: improve download script
* `#27202 <https://github.com/numpy/numpy/pull/27202>`__: BUG: Fix NPY_RAVEL_AXIS on backwards compatible NumPy 2 builds
* `#27203 <https://github.com/numpy/numpy/pull/27203>`__: DOC: update PyArray_CheckAxis doc
* `#27207 <https://github.com/numpy/numpy/pull/27207>`__: TYP: Deprecate calling ``numpy.save`` with ``fix_imports`` (PEP...
* `#27208 <https://github.com/numpy/numpy/pull/27208>`__: TYP: Disallow scalars and 0d-arrays in ``numpy.nonzero``
* `#27210 <https://github.com/numpy/numpy/pull/27210>`__: TYP: Semi-transparent ``numpy.shape`` shape-type annotations.
* `#27211 <https://github.com/numpy/numpy/pull/27211>`__: TYP: Stop using ``Any`` as shape-type default
* `#27215 <https://github.com/numpy/numpy/pull/27215>`__: MAINT: Bump github/codeql-action from 3.26.0 to 3.26.2
* `#27218 <https://github.com/numpy/numpy/pull/27218>`__: DEV: Add ``.editorconfig`` rules for Python
* `#27219 <https://github.com/numpy/numpy/pull/27219>`__: TYP: Replace ``ellipsis`` with ``types.EllipsisType``
* `#27220 <https://github.com/numpy/numpy/pull/27220>`__: TYP: Fixed & improved ``TypeVar`` use for ``numpy.char.chararray``
* `#27221 <https://github.com/numpy/numpy/pull/27221>`__: MAINT: Bump actions/upload-artifact from 4.3.3 to 4.3.6
* `#27223 <https://github.com/numpy/numpy/pull/27223>`__: DOC: add docs on thread safety in NumPy
* `#27226 <https://github.com/numpy/numpy/pull/27226>`__: BUG: Fix ``PyArray_ZeroContiguousBuffer`` (resize) with struct...
* `#27228 <https://github.com/numpy/numpy/pull/27228>`__: DOC: Remove obsolete note from the top of the 2.0.0 release notes.
* `#27235 <https://github.com/numpy/numpy/pull/27235>`__: MAINT: MSVC does not support #warning directive
* `#27237 <https://github.com/numpy/numpy/pull/27237>`__: TYP: Fix several typing issues in ``numpy.polynomial``
* `#27238 <https://github.com/numpy/numpy/pull/27238>`__: DOC: update ``np.unique`` docstring
* `#27242 <https://github.com/numpy/numpy/pull/27242>`__: MAINT: Update main after 2.1.0 release.
* `#27246 <https://github.com/numpy/numpy/pull/27246>`__: MAINT: Bump github/codeql-action from 3.26.2 to 3.26.3
* `#27247 <https://github.com/numpy/numpy/pull/27247>`__: DOC: update documentation release process
* `#27249 <https://github.com/numpy/numpy/pull/27249>`__: BUG: fix reference counting bug in __array_interface__ implementation
* `#27255 <https://github.com/numpy/numpy/pull/27255>`__: BUG: revert unintended change in the return value of set_printoptions
* `#27261 <https://github.com/numpy/numpy/pull/27261>`__: TST: Add regression test for missing descr in array-interface
* `#27262 <https://github.com/numpy/numpy/pull/27262>`__: BUG: Fix #27256 and #27257
* `#27268 <https://github.com/numpy/numpy/pull/27268>`__: MAINT: Bump github/codeql-action from 3.26.3 to 3.26.4
* `#27272 <https://github.com/numpy/numpy/pull/27272>`__: ENH: make check-{docs,tutorials} fail on dtype mismatch
* `#27275 <https://github.com/numpy/numpy/pull/27275>`__: BUG: Fix array_equal for numeric and non-numeric scalar types
* `#27277 <https://github.com/numpy/numpy/pull/27277>`__: DOC/DEV/CI: mambaforge -> miniforge
* `#27281 <https://github.com/numpy/numpy/pull/27281>`__: MAINT: Bump github/codeql-action from 3.26.4 to 3.26.5
* `#27284 <https://github.com/numpy/numpy/pull/27284>`__: BLD: cp311- macosx_arm64 wheels [wheel build]
* `#27286 <https://github.com/numpy/numpy/pull/27286>`__: MAINT: Update main after the 2.0.2 release
* `#27289 <https://github.com/numpy/numpy/pull/27289>`__: MAINT: Start applying ruff rules (RUF)
* `#27290 <https://github.com/numpy/numpy/pull/27290>`__: MAINT: Keep applying ruff/pyupgrade rules (UP)
* `#27291 <https://github.com/numpy/numpy/pull/27291>`__: DOC, MAINT: Fix new typos found by codespell
* `#27292 <https://github.com/numpy/numpy/pull/27292>`__: MAINT: Start applying ruff/flake8-type-checking rules (TCH)
* `#27293 <https://github.com/numpy/numpy/pull/27293>`__: MAINT: Keep applying ruff/flake8-bugbear rules (B)
* `#27294 <https://github.com/numpy/numpy/pull/27294>`__: BUILD: refactor circleci to use spin [skip actions][skip azp][skip...
* `#27295 <https://github.com/numpy/numpy/pull/27295>`__: MAINT: Start applying rruff/flake8-pie rules (PIE)
* `#27296 <https://github.com/numpy/numpy/pull/27296>`__: MAINT: Start applying ruff/flake8-comprehensions rules (C4)
* `#27297 <https://github.com/numpy/numpy/pull/27297>`__: MAINT: Apply ruff/flake8-raise rules (RSE)
* `#27298 <https://github.com/numpy/numpy/pull/27298>`__: MAINT: Apply ruff/flynt rules (FLY)
* `#27302 <https://github.com/numpy/numpy/pull/27302>`__: BUG: Fix bug in ``doc/neps/tools/build_index.py``
* `#27307 <https://github.com/numpy/numpy/pull/27307>`__: MAINT: Apply ruff/pycodestyle warning rules (W)
* `#27311 <https://github.com/numpy/numpy/pull/27311>`__: MAINT: Bump actions/setup-python from 5.1.1 to 5.2.0
* `#27312 <https://github.com/numpy/numpy/pull/27312>`__: MAINT: Bump github/codeql-action from 3.26.5 to 3.26.6
* `#27316 <https://github.com/numpy/numpy/pull/27316>`__: BUILD: update pypy test version
* `#27320 <https://github.com/numpy/numpy/pull/27320>`__: MAINT: increase max line length from 79 to 88, upgrade pycodestyle
* `#27322 <https://github.com/numpy/numpy/pull/27322>`__: DOC: Removed reference to deprecated "newshape" parameter in...
* `#27323 <https://github.com/numpy/numpy/pull/27323>`__: TYP: add ``ma.zeros_like`` and ``ma.ones_like`` typing
* `#27326 <https://github.com/numpy/numpy/pull/27326>`__: MAINT: Bump actions/upload-artifact from 4.3.6 to 4.4.0
* `#27330 <https://github.com/numpy/numpy/pull/27330>`__: BLD: Win-arm64 cross compile workflow
* `#27331 <https://github.com/numpy/numpy/pull/27331>`__: MAINT: GitHub Actions: Replace deprecated macos-12 with macos-latest
* `#27332 <https://github.com/numpy/numpy/pull/27332>`__: MAINT: Update main after 2.1.1 release.
* `#27334 <https://github.com/numpy/numpy/pull/27334>`__: TYP: Concrete ``float64`` and ``complex128`` scalar types with...
* `#27335 <https://github.com/numpy/numpy/pull/27335>`__: ENH: Add ``allow_pickle`` flag to ``savez``
* `#27344 <https://github.com/numpy/numpy/pull/27344>`__: MAINT: fix typos
* `#27346 <https://github.com/numpy/numpy/pull/27346>`__: BUG,TYP: Allow subscripting ``iinfo`` and ``finfo`` generic types...
* `#27347 <https://github.com/numpy/numpy/pull/27347>`__: DOC: Mention that c is reassigned but still points to a (quickstart)
* `#27353 <https://github.com/numpy/numpy/pull/27353>`__: MNT, CI: Use separate jobs for WASM wheel builds/uploads
* `#27355 <https://github.com/numpy/numpy/pull/27355>`__: MAINT: Bump actions/setup-python from 5.1.1 to 5.2.0
* `#27356 <https://github.com/numpy/numpy/pull/27356>`__: MAINT: Bump actions/upload-artifact from 4.3.6 to 4.4.0
* `#27359 <https://github.com/numpy/numpy/pull/27359>`__: MAINT: fix typo in random.binomial
* `#27360 <https://github.com/numpy/numpy/pull/27360>`__: BUG: fix _shrink edge case in np.ma.mask_or
* `#27361 <https://github.com/numpy/numpy/pull/27361>`__: BUILD: fix missing include for std::ptrdiff_t for C++23 language...
* `#27363 <https://github.com/numpy/numpy/pull/27363>`__: DOC: Remove reshape from appearing twice in toctree
* `#27364 <https://github.com/numpy/numpy/pull/27364>`__: DOC: Update np.\*stack doc to reflect behavior
* `#27365 <https://github.com/numpy/numpy/pull/27365>`__: MAINT: Bump deadsnakes/action from 3.1.0 to 3.2.0
* `#27369 <https://github.com/numpy/numpy/pull/27369>`__: DOC: fix incorrect definitions
* `#27372 <https://github.com/numpy/numpy/pull/27372>`__: CI: Update cirrus nightly token
* `#27376 <https://github.com/numpy/numpy/pull/27376>`__: MAINT: Fix a few typos - and sometimes improve wording
* `#27381 <https://github.com/numpy/numpy/pull/27381>`__: DOC: add vecdot to 'See also' of np.dot and np.inner
* `#27384 <https://github.com/numpy/numpy/pull/27384>`__: MAINT: Fix a few more typos
* `#27385 <https://github.com/numpy/numpy/pull/27385>`__: DOC: Update np.unique_all example to demonstrate namedtuple output
* `#27387 <https://github.com/numpy/numpy/pull/27387>`__: DOC: Clarify np.searchsorted documentation and add example for...
* `#27390 <https://github.com/numpy/numpy/pull/27390>`__: MAINT: Bump github/codeql-action from 3.26.6 to 3.26.7
* `#27391 <https://github.com/numpy/numpy/pull/27391>`__: MAINT: Bump pypa/cibuildwheel from 2.20.0 to 2.21.0
* `#27392 <https://github.com/numpy/numpy/pull/27392>`__: BUG: apply critical sections around populating the dispatch cache
* `#27403 <https://github.com/numpy/numpy/pull/27403>`__: DOC: Fix minor issues in arrays.promotion.rst
* `#27406 <https://github.com/numpy/numpy/pull/27406>`__: BUG: Stub out ``get_build_msvc_version`` if ``distutils.msvccompiler``...
* `#27408 <https://github.com/numpy/numpy/pull/27408>`__: DOC: more informative _excluded_ argument explanation in np.vectorize
* `#27412 <https://github.com/numpy/numpy/pull/27412>`__: MAINT: Bump pypa/cibuildwheel from 2.21.0 to 2.21.1
* `#27414 <https://github.com/numpy/numpy/pull/27414>`__: MAINT: add Python 3.13 to classifiers
* `#27417 <https://github.com/numpy/numpy/pull/27417>`__: TYP: Allow callable ``converters`` arg in ``numpy.loadtxt``
* `#27418 <https://github.com/numpy/numpy/pull/27418>`__: TYP: Fix default return dtype of ``numpy.random.Generator.integers``...
* `#27419 <https://github.com/numpy/numpy/pull/27419>`__: TYP: Modernized ``numpy.dtypes`` annotations
* `#27420 <https://github.com/numpy/numpy/pull/27420>`__: TYP: Optional 2nd ``numpy.complexfloating`` type parameter
* `#27421 <https://github.com/numpy/numpy/pull/27421>`__: BUG: Add regression test for gh-27273
* `#27423 <https://github.com/numpy/numpy/pull/27423>`__: TYP: Add missing type arguments
* `#27424 <https://github.com/numpy/numpy/pull/27424>`__: DOC: Add release notes for #27334
* `#27425 <https://github.com/numpy/numpy/pull/27425>`__: MAINT: Use correct Python interpreter in tests
* `#27426 <https://github.com/numpy/numpy/pull/27426>`__: MAINT: Bump github/codeql-action from 3.26.7 to 3.26.8
* `#27427 <https://github.com/numpy/numpy/pull/27427>`__: TYP: Fixed & improved type-hinting for ``any`` and ``all``
* `#27429 <https://github.com/numpy/numpy/pull/27429>`__: BLD: pin setuptools to avoid breaking numpy.distutils
* `#27430 <https://github.com/numpy/numpy/pull/27430>`__: TYP: Fix type of ``copy`` argument in ``ndarray.reshape``
* `#27431 <https://github.com/numpy/numpy/pull/27431>`__: BUG: Allow unsigned shift argument for np.roll
* `#27434 <https://github.com/numpy/numpy/pull/27434>`__: ENH: make np.dtype(scalar_type) return the default dtype instance
* `#27438 <https://github.com/numpy/numpy/pull/27438>`__: BUG: Disable SVE VQSort
* `#27440 <https://github.com/numpy/numpy/pull/27440>`__: DOC: Add a link to the migration guide for the deprecation warning...
* `#27441 <https://github.com/numpy/numpy/pull/27441>`__: DOC: remove old versionadded comments from arrays.classes.rst
* `#27442 <https://github.com/numpy/numpy/pull/27442>`__: DOC: Remove old versionchanged directives from config.rst
* `#27443 <https://github.com/numpy/numpy/pull/27443>`__: updated the version of mean param from the release notes (2.0.0)
* `#27444 <https://github.com/numpy/numpy/pull/27444>`__: TST: Added the test case for masked array tofile failing
* `#27445 <https://github.com/numpy/numpy/pull/27445>`__: DOC: removed older versionadded directives to ufuncs.rst
* `#27448 <https://github.com/numpy/numpy/pull/27448>`__: DOC: Example for char.array
* `#27453 <https://github.com/numpy/numpy/pull/27453>`__: DOC: Added docstring for numpy.ma.take() function.
* `#27454 <https://github.com/numpy/numpy/pull/27454>`__: DOC: Remove outdated versionadded/changed directives
* `#27458 <https://github.com/numpy/numpy/pull/27458>`__: MAINT: Bump github/codeql-action from 3.26.8 to 3.26.9
* `#27464 <https://github.com/numpy/numpy/pull/27464>`__: DOC: Fix a copy-paste mistake in the cumulative_sum docstring.
* `#27465 <https://github.com/numpy/numpy/pull/27465>`__: DOC: update ndindex reference in np.choose docstring
* `#27466 <https://github.com/numpy/numpy/pull/27466>`__: BUG: rfftn axis bug
* `#27469 <https://github.com/numpy/numpy/pull/27469>`__: DOC: Added ``CONTRIBUTING.rst``
* `#27470 <https://github.com/numpy/numpy/pull/27470>`__: TYP: Add type stubs for stringdtype in np.char and np.strings
* `#27472 <https://github.com/numpy/numpy/pull/27472>`__: MAINT: Check for SVE support on demand
* `#27475 <https://github.com/numpy/numpy/pull/27475>`__: CI: use PyPI not scientific-python-nightly-wheels for CI doc...
* `#27478 <https://github.com/numpy/numpy/pull/27478>`__: BUG: Fix extra decref of PyArray_UInt8DType.
* `#27482 <https://github.com/numpy/numpy/pull/27482>`__: Show shape any time it cannot be inferred in repr
* `#27485 <https://github.com/numpy/numpy/pull/27485>`__: MAINT: Bump github/codeql-action from 3.26.9 to 3.26.10
* `#27486 <https://github.com/numpy/numpy/pull/27486>`__: MAINT: Bump scientific-python/upload-nightly-action from 0.5.0...
* `#27490 <https://github.com/numpy/numpy/pull/27490>`__: API: register NEP 35 functions as array_functions
* `#27491 <https://github.com/numpy/numpy/pull/27491>`__: MAINT: Bump mamba-org/setup-micromamba from 1.9.0 to 1.10.0
* `#27495 <https://github.com/numpy/numpy/pull/27495>`__: MAINT: Bump pypa/cibuildwheel from 2.21.1 to 2.21.2
* `#27496 <https://github.com/numpy/numpy/pull/27496>`__: MAINT: Bump mamba-org/setup-micromamba from 1.10.0 to 2.0.0
* `#27497 <https://github.com/numpy/numpy/pull/27497>`__: DOC: Correct selected C docstrings to eliminate warnings
* `#27499 <https://github.com/numpy/numpy/pull/27499>`__: DOC: fix missing arguments (copy and device) from asanyarray's...
* `#27502 <https://github.com/numpy/numpy/pull/27502>`__: MAINT: Bump github/codeql-action from 3.26.10 to 3.26.11
* `#27503 <https://github.com/numpy/numpy/pull/27503>`__: BUG: avoid segfault on bad arguments in ndarray.__array_function__
* `#27504 <https://github.com/numpy/numpy/pull/27504>`__: ENH: Allow ``ndarray.__array_function__`` to dispatch functions...
* `#27508 <https://github.com/numpy/numpy/pull/27508>`__: MAINT: Pin setuptools for testing [wheel build]
* `#27510 <https://github.com/numpy/numpy/pull/27510>`__: TYP: Mark stub-only classes as ``@type_check_only``
* `#27511 <https://github.com/numpy/numpy/pull/27511>`__: TYP: Annotate type aliases without annotation
* `#27513 <https://github.com/numpy/numpy/pull/27513>`__: MAINT: Update main after NumPy 2.1.2 release
* `#27517 <https://github.com/numpy/numpy/pull/27517>`__: BENCH: Add benchmarks for np.non_zero
* `#27518 <https://github.com/numpy/numpy/pull/27518>`__: TST: Add tests for np.nonzero with different input types
* `#27520 <https://github.com/numpy/numpy/pull/27520>`__: TYP: Remove unused imports in the stubs
* `#27521 <https://github.com/numpy/numpy/pull/27521>`__: TYP: Fill in the missing ``__all__`` exports
* `#27524 <https://github.com/numpy/numpy/pull/27524>`__: MAINT: Bump actions/cache from 4.0.2 to 4.1.0
* `#27525 <https://github.com/numpy/numpy/pull/27525>`__: MAINT: Bump actions/upload-artifact from 4.4.0 to 4.4.1
* `#27526 <https://github.com/numpy/numpy/pull/27526>`__: MAINT: Bump github/codeql-action from 3.26.11 to 3.26.12
* `#27532 <https://github.com/numpy/numpy/pull/27532>`__: MAINT: Bump actions/cache from 4.1.0 to 4.1.1
* `#27534 <https://github.com/numpy/numpy/pull/27534>`__: BUG: Fix user dtype can-cast with python scalar during promotion
* `#27535 <https://github.com/numpy/numpy/pull/27535>`__: MAINT: Bump pypa/cibuildwheel from 2.21.2 to 2.21.3
* `#27536 <https://github.com/numpy/numpy/pull/27536>`__: MAINT: Bump actions/upload-artifact from 4.4.1 to 4.4.3
* `#27549 <https://github.com/numpy/numpy/pull/27549>`__: BUG: weighted quantile for some zero weights
* `#27550 <https://github.com/numpy/numpy/pull/27550>`__: BLD: update vendored Meson to 1.5.2
* `#27551 <https://github.com/numpy/numpy/pull/27551>`__: MAINT: Bump github/codeql-action from 3.26.12 to 3.26.13
* `#27553 <https://github.com/numpy/numpy/pull/27553>`__: BLD: rename ``meson_options.txt`` to ``meson.options``
* `#27555 <https://github.com/numpy/numpy/pull/27555>`__: DEV: bump ``python`` to 3.12 in environment.yml
* `#27556 <https://github.com/numpy/numpy/pull/27556>`__: DOC: Clarify use of standard deviation in mtrand.pyx
* `#27557 <https://github.com/numpy/numpy/pull/27557>`__: BUG: Fix warning "differs in levels of indirection" in npy_atomic.h...
* `#27558 <https://github.com/numpy/numpy/pull/27558>`__: MAINT: distutils: remove obsolete search for ``ecc`` executable
* `#27560 <https://github.com/numpy/numpy/pull/27560>`__: CI: start building Windows free-threaded wheels
* `#27564 <https://github.com/numpy/numpy/pull/27564>`__: BUILD: satisfy gcc-13 pendantic errors
* `#27567 <https://github.com/numpy/numpy/pull/27567>`__: BUG: handle possible error for PyTraceMallocTrack
* `#27568 <https://github.com/numpy/numpy/pull/27568>`__: BUILD: vendor tempita from Cython
* `#27579 <https://github.com/numpy/numpy/pull/27579>`__: BUG: Adjust numpy.i for SWIG 4.3 compatibility
* `#27586 <https://github.com/numpy/numpy/pull/27586>`__: MAINT: Update Highway to latest
* `#27587 <https://github.com/numpy/numpy/pull/27587>`__: BLD: treat SVML object files better to avoid compiler warnings
* `#27595 <https://github.com/numpy/numpy/pull/27595>`__: DOC: Clarify obj parameter types in numpy.delete documentation
* `#27598 <https://github.com/numpy/numpy/pull/27598>`__: DOC: add examples to ctypeslib
* `#27602 <https://github.com/numpy/numpy/pull/27602>`__: Update documentation for floating-point precision and determinant...
* `#27604 <https://github.com/numpy/numpy/pull/27604>`__: DOC: Fix rendering in docstring of nan_to_num
* `#27612 <https://github.com/numpy/numpy/pull/27612>`__: ENH: Add comments to ``string_fastsearch.h`` , rename some C-methods
* `#27613 <https://github.com/numpy/numpy/pull/27613>`__: BUG: Fix Linux QEMU CI workflow
* `#27615 <https://github.com/numpy/numpy/pull/27615>`__: ENH: Fix np.insert to handle boolean arrays as masks
* `#27617 <https://github.com/numpy/numpy/pull/27617>`__: DOC: Update the RELEASE_WALKTHROUGH.rst file.
* `#27619 <https://github.com/numpy/numpy/pull/27619>`__: MAINT: Bump actions/cache from 4.1.1 to 4.1.2
* `#27620 <https://github.com/numpy/numpy/pull/27620>`__: MAINT: Bump actions/dependency-review-action from 4.3.4 to 4.3.5
* `#27621 <https://github.com/numpy/numpy/pull/27621>`__: MAINT: Bump github/codeql-action from 3.26.13 to 3.27.0
* `#27627 <https://github.com/numpy/numpy/pull/27627>`__: ENH: Re-enable VSX from build targets for sin/cos
* `#27630 <https://github.com/numpy/numpy/pull/27630>`__: ENH: Extern memory management to Cython
* `#27634 <https://github.com/numpy/numpy/pull/27634>`__: MAINT: Bump actions/setup-python from 5.2.0 to 5.3.0
* `#27636 <https://github.com/numpy/numpy/pull/27636>`__: BUG: fixes for StringDType/unicode promoters
* `#27643 <https://github.com/numpy/numpy/pull/27643>`__: BUG : avoid maximum fill value of datetime and timedelta return...
* `#27644 <https://github.com/numpy/numpy/pull/27644>`__: DOC: Remove ambiguity in docs for ndarray.byteswap()
* `#27650 <https://github.com/numpy/numpy/pull/27650>`__: BLD: Do not set __STDC_VERSION__ to zero during build
* `#27652 <https://github.com/numpy/numpy/pull/27652>`__: TYP,TST: Bump ``mypy`` from ``1.11.1`` to ``1.13.0``
* `#27653 <https://github.com/numpy/numpy/pull/27653>`__: TYP: Fix Array API method signatures
* `#27659 <https://github.com/numpy/numpy/pull/27659>`__: TYP: Transparent ``ndarray`` unary operator method signatures
* `#27661 <https://github.com/numpy/numpy/pull/27661>`__: BUG: np.cov transpose control
* `#27663 <https://github.com/numpy/numpy/pull/27663>`__: MAINT: fix wasm32 runtime type error in numpy._core
* `#27664 <https://github.com/numpy/numpy/pull/27664>`__: MAINT: Bump actions/dependency-review-action from 4.3.5 to 4.4.0
* `#27665 <https://github.com/numpy/numpy/pull/27665>`__: ENH: Re-enable VXE from build targets for sin/cos
* `#27666 <https://github.com/numpy/numpy/pull/27666>`__: BUG: Fix a reference count leak in npy_find_descr_for_scalar.
* `#27667 <https://github.com/numpy/numpy/pull/27667>`__: TYP: Allow returning non-array-likes from the ``apply_along_axis``...
* `#27676 <https://github.com/numpy/numpy/pull/27676>`__: CI: Attempt to fix CI on 32 bit linux
* `#27678 <https://github.com/numpy/numpy/pull/27678>`__: DOC: fix incorrect versionadded for np.std
* `#27680 <https://github.com/numpy/numpy/pull/27680>`__: MAINT: fix typo / copy paste error
* `#27681 <https://github.com/numpy/numpy/pull/27681>`__: TYP: Fix some inconsistencies in the scalar methods and properties
* `#27683 <https://github.com/numpy/numpy/pull/27683>`__: TYP: Improve ``np.sum`` and ``np.mean`` return types with given...
* `#27684 <https://github.com/numpy/numpy/pull/27684>`__: DOC: fix spelling of "reality" in ``_nanfunctions_impl.pyi``
* `#27685 <https://github.com/numpy/numpy/pull/27685>`__: MAINT: Drop useless shebang
* `#27691 <https://github.com/numpy/numpy/pull/27691>`__: TYP: Use ``_typeshed`` to clean up the stubs
* `#27693 <https://github.com/numpy/numpy/pull/27693>`__: MAINT: Update main after 2.1.3 release.
* `#27695 <https://github.com/numpy/numpy/pull/27695>`__: BUG: Fix multiple modules in F2PY and COMMON handling
* `#27702 <https://github.com/numpy/numpy/pull/27702>`__: MAINT: Bump conda-incubator/setup-miniconda from 3.0.4 to 3.1.0
* `#27705 <https://github.com/numpy/numpy/pull/27705>`__: MAINT: Bump mamba-org/setup-micromamba from 2.0.0 to 2.0.1
* `#27706 <https://github.com/numpy/numpy/pull/27706>`__: DOC: Remove empty notes
* `#27707 <https://github.com/numpy/numpy/pull/27707>`__: CI: Set up free-threaded CI using quansight-labs/setup-python
* `#27708 <https://github.com/numpy/numpy/pull/27708>`__: DOC: Remove version notes
* `#27714 <https://github.com/numpy/numpy/pull/27714>`__: DOC: fix a mistake in the docstring of vector_norm
* `#27715 <https://github.com/numpy/numpy/pull/27715>`__: BUG: fix incorrect output descriptor in fancy indexing
* `#27716 <https://github.com/numpy/numpy/pull/27716>`__: ENH: Make ``__module__`` attribute coherent across API
* `#27721 <https://github.com/numpy/numpy/pull/27721>`__: DOC: fix name of shape parameter kappa of von Mises distribution
* `#27723 <https://github.com/numpy/numpy/pull/27723>`__: BUG: Allow empty memmaps in most situations
* `#27724 <https://github.com/numpy/numpy/pull/27724>`__: MAINT: Bump github/codeql-action from 3.27.0 to 3.27.1
* `#27728 <https://github.com/numpy/numpy/pull/27728>`__: BUG: Handle ``--lower`` for F2PY directives and callbacks
* `#27729 <https://github.com/numpy/numpy/pull/27729>`__: BUG: f2py: fix issues with thread-local storage define
* `#27730 <https://github.com/numpy/numpy/pull/27730>`__: TST: Add an F2PY check for exposing variables without functions
* `#27731 <https://github.com/numpy/numpy/pull/27731>`__: BUG: Fix ``fortranname`` for functions
* `#27734 <https://github.com/numpy/numpy/pull/27734>`__: Fix documentation for the chi-square distribution
* `#27735 <https://github.com/numpy/numpy/pull/27735>`__: ENH: Add a ``__dict__`` to ufunc objects and allow overriding...
* `#27736 <https://github.com/numpy/numpy/pull/27736>`__: TYP: Optional ``numpy.number`` type parameters
* `#27742 <https://github.com/numpy/numpy/pull/27742>`__: MAINT: Bump github/codeql-action from 3.27.1 to 3.27.2
* `#27743 <https://github.com/numpy/numpy/pull/27743>`__: DOC: Fix typos in subclassing documentation
* `#27746 <https://github.com/numpy/numpy/pull/27746>`__: DOC: Added additional guidance for compiling in Windows
* `#27750 <https://github.com/numpy/numpy/pull/27750>`__: TYP: Fix ``ndarray.item()`` and improve ``ndarray.tolist()``
* `#27753 <https://github.com/numpy/numpy/pull/27753>`__: TYP: Fix the annotations of ``ndarray.real`` and ``ndarray.imag``
* `#27754 <https://github.com/numpy/numpy/pull/27754>`__: MAINT: Bump github/codeql-action from 3.27.2 to 3.27.3
* `#27755 <https://github.com/numpy/numpy/pull/27755>`__: TYP: Annotate ``__setitem__`` , ``__contains__`` and ``__iter__``...
* `#27756 <https://github.com/numpy/numpy/pull/27756>`__: TYP: 1-d shape-typing for ``ndarray.flatten`` and ``ravel``
* `#27757 <https://github.com/numpy/numpy/pull/27757>`__: TYP: Remove the non-existent ``bitwise_count`` methods of ``ndarray``...
* `#27758 <https://github.com/numpy/numpy/pull/27758>`__: TYP: Remove ``ndarray`` binop overloads for ``NDArray[Never]``
* `#27763 <https://github.com/numpy/numpy/pull/27763>`__: DOC: Note that allow-pickle is not safe also in error
* `#27765 <https://github.com/numpy/numpy/pull/27765>`__: TYP: Shape-typed ``ndarray`` inplace binary operator methods.
* `#27766 <https://github.com/numpy/numpy/pull/27766>`__: MAINT: Bump github/codeql-action from 3.27.3 to 3.27.4
* `#27767 <https://github.com/numpy/numpy/pull/27767>`__: TYP: Support shape-typing in ``reshape`` and ``resize``
* `#27769 <https://github.com/numpy/numpy/pull/27769>`__: TYP: Towards a less messy ``__init__.pyi``
* `#27770 <https://github.com/numpy/numpy/pull/27770>`__: TYP: Fix incorrect baseclass of ``linalg.LinAlgError``
* `#27771 <https://github.com/numpy/numpy/pull/27771>`__: ENH: ``default_rng`` coerces ``RandomState`` to ``Generator``
* `#27773 <https://github.com/numpy/numpy/pull/27773>`__: BUG: Fix repeat, accumulate for strings and accumulate API logic
* `#27775 <https://github.com/numpy/numpy/pull/27775>`__: TYP: Fix undefined type-parameter name
* `#27776 <https://github.com/numpy/numpy/pull/27776>`__: TYP: Fix method overload issues in ``ndarray`` and ``generic``
* `#27778 <https://github.com/numpy/numpy/pull/27778>`__: TYP: Generic ``numpy.generic`` type parameter for the ``item()``...
* `#27779 <https://github.com/numpy/numpy/pull/27779>`__: TYP: Type hints for ``numpy.__config__``
* `#27788 <https://github.com/numpy/numpy/pull/27788>`__: DOC: Make wording in absolute beginners guide more beginner friendly
* `#27790 <https://github.com/numpy/numpy/pull/27790>`__: TYP: Generic ``timedelta64`` and ``datetime64`` scalar types
* `#27792 <https://github.com/numpy/numpy/pull/27792>`__: TYP: Generic ``numpy.bool`` and statically typed boolean logic
* `#27794 <https://github.com/numpy/numpy/pull/27794>`__: MAINT: Upgrade to spin 0.13
* `#27795 <https://github.com/numpy/numpy/pull/27795>`__: update pythoncapi-compat to latest HEAD
* `#27800 <https://github.com/numpy/numpy/pull/27800>`__: BUG: Ensure context path is taken in masked array array-wrap
* `#27802 <https://github.com/numpy/numpy/pull/27802>`__: BUG: Ensure that same-kind casting works for uints (mostly)
* `#27803 <https://github.com/numpy/numpy/pull/27803>`__: MAINT: Bump github/codeql-action from 3.27.4 to 3.27.5
* `#27806 <https://github.com/numpy/numpy/pull/27806>`__: DOC: Improve choice() documentation about return types
* `#27807 <https://github.com/numpy/numpy/pull/27807>`__: BUG,ENH: Fix internal ``__array_wrap__`` for direct calls
* `#27808 <https://github.com/numpy/numpy/pull/27808>`__: ENH: Ensure hugepages are also indicated for calloc allocations
* `#27809 <https://github.com/numpy/numpy/pull/27809>`__: BUG: Fix array flags propagation in boolean indexing
* `#27810 <https://github.com/numpy/numpy/pull/27810>`__: MAINT: Bump actions/dependency-review-action from 4.4.0 to 4.5.0
* `#27812 <https://github.com/numpy/numpy/pull/27812>`__: BUG: ``timedelta64.__[r]divmod__`` segfaults for incompatible...
* `#27813 <https://github.com/numpy/numpy/pull/27813>`__: DOC: fix broken reference in arrays.classes.rst
* `#27815 <https://github.com/numpy/numpy/pull/27815>`__: DOC: Add a release fragment for gh-14622
* `#27816 <https://github.com/numpy/numpy/pull/27816>`__: MAINT: Fixup that spin can be installed via conda too now
* `#27817 <https://github.com/numpy/numpy/pull/27817>`__: DEV: changelog: make title processing more robust
* `#27828 <https://github.com/numpy/numpy/pull/27828>`__: CI: skip ninja installation in linux_qemu workflows
* `#27829 <https://github.com/numpy/numpy/pull/27829>`__: CI: update circleci to python3.11.10, limit parallel builds....
* `#27831 <https://github.com/numpy/numpy/pull/27831>`__: BUG: Fix mismatch in definition and declaration for a couple...
* `#27843 <https://github.com/numpy/numpy/pull/27843>`__: DOC: Correct version-added for mean arg for nanvar and nanstd
* `#27845 <https://github.com/numpy/numpy/pull/27845>`__: BUG: Never negate strides in reductions (for now)
* `#27846 <https://github.com/numpy/numpy/pull/27846>`__: ENH: add matvec and vecmat gufuncs
* `#27852 <https://github.com/numpy/numpy/pull/27852>`__: DOC: Correct versionadded for vecmat and matvec.
* `#27853 <https://github.com/numpy/numpy/pull/27853>`__: REL: Prepare for the NumPy 2.2.0rc1 release [wheel build]
* `#27874 <https://github.com/numpy/numpy/pull/27874>`__: BUG: fix importing numpy in Python's optimized mode (#27868)
* `#27895 <https://github.com/numpy/numpy/pull/27895>`__: DOC: Fix double import in docs (#27878)
* `#27904 <https://github.com/numpy/numpy/pull/27904>`__: MAINT: Ensure correct handling for very large unicode strings
* `#27906 <https://github.com/numpy/numpy/pull/27906>`__: MAINT: Use mask_store instead of store for compiler workaround
* `#27908 <https://github.com/numpy/numpy/pull/27908>`__: MAINT: Update highway from main.
* `#27911 <https://github.com/numpy/numpy/pull/27911>`__: ENH: update __module__ in numpy.random module
* `#27912 <https://github.com/numpy/numpy/pull/27912>`__: ENH: Refactor ``__qualname__`` across API
* `#27913 <https://github.com/numpy/numpy/pull/27913>`__: PERF: improve multithreaded ufunc scaling
* `#27916 <https://github.com/numpy/numpy/pull/27916>`__: MAINT: Bump actions/cache from 4.1.2 to 4.2.0

