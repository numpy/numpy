=============================================================
NEP 38 — Using SIMD optimization instructions for performance
=============================================================

:Author: Sayed Adel, Matti Picus, Ralf Gommers
:Status: Draft
:Type: Standards
:Created: 2019-11-25
:Resolution: none


Abstract
--------

While compilers are getting better at using hardware-specific routines to
optimize code, they sometimes do not produce optimal results. Also, we would
like to be able to copy binary optimized C-extension modules from one machine
to another with the same base architecture (x86, ARM, PowerPC) but with
different capabilities without recompiling.

We have a mechanism in the ufunc machinery to `build alternative loops`_
indexed by CPU feature name. At import (in ``InitOperators``), the loop
function that matches the run-time CPU info `is chosen`_ from the candidates.This
NEP proposes a mechanism to build on that for many more features and
architectures.  The steps proposed are to:

- Establish a set of well-defined, architecture-agnostic, universal intrisics
  which capture features available across architectures.
- Capture these universal intrisics in a set of C macros and use the macros
  to build code paths for sets of features from the baseline up to the maximum
  set of features available on that architecture. Offer these as a limited
  number of compiled alternative code paths.
- At runtime, discover which CPU features are available, and choose from among
  the possible code paths accordingly.


Motivation and Scope
--------------------

Traditionally NumPy has counted on the compilers to generate optimal code
specifically for the target architecture.
However few users today compile NumPy locally for their machines. Most use the
binary packages which must provide run-time support for the lowest-common
denominator CPU architecture. Thus NumPy cannot take advantage of 
more advanced features of their CPU processors, since they may not be available
on all users' systems. The ufunc machinery already has a loop-selection
protocol based on dtypes, so it is easy to extend this to also select an
optimal loop for specifically available CPU features at runtime.

Traditionally, these features have been exposed through `intrinsics`_ which are
compiler-specific instructions that map directly to assembly instructions.
Recently there were discussions about the effectiveness of adding more
intrinsics (e.g., `gh-11113`_ for AVX optimizations for floats).  In the past,
architecture-specific code was added to NumPy for `fast avx512 routines`_ in
various ufuncs, using the mechanism described above to choose the best loop
for the architecture. However the code is not generic and does not generalize
to other architectures.

Recently, OpenCV moved to using `universal intrinsics`_ in the Hardware
Abstraction Layer (HAL) which provided a nice abstraction for common shared
Single Instruction Multiple Data (SIMD) constructs. This NEP proposes a similar
mechanism for NumPy. There are three stages to using the mechanism:
- Infrastructure is provided in the code for abstract intrinsics. The ufunc
  machinery will be extended using sets of these abstract intrinsics, so that
  a single ufunc will be expressed as a set of loops, going from a minimal to
  a maximal set of possibly availabe intrinsics.
- At compile time, compiler macros and CPU detection are used to turn the
  abstract intrinsics into concrete intrinsic calls. Any intrinsics not
  available on the platform, either because the CPU does not support them
  (and so cannot be tested) or because the abstract intrinsic does not have a
  parallel concrete intrinsic on the platform will not error, rather the
  corresponding loop will not be produced and added to the set of
  possibilities.
- At runtime, the CPU detection code will further limit the set of loops
  available, and the optimal one will be chosen for the ufunc.

The current NEP proposes only to use the runtime feature detection and optimal
loop selection mechanism for ufuncs. Future NEPS may propose other uses for the
proposed solution.

Usage and Impact
----------------

The end user will be able to get a list of intrinsics available for their
platform and compiler. Optionally,
the user may be able to specify which of the loops available at runtime will be
used, perhaps via an environment variable to enable benchmarking the impact of
the different loops. There should be no direct impact to naive end users, the
results of all the loops should be identical to within a small number (1-3?)
ULPs. On the other hand, users with more powerful machines should notice a
significant performance boost.

Binary releases - wheels on PyPI and conda packages
```````````````````````````````````````````````````

The binaries released by this process will be larger since they include all
possible loops for the architecture. Some packagers may prefer to limit the
number of loops in order to limit the size of the binaries, we would hope they
would still support a wide range of families of architectures. Note this
problem already exists in the Intel MKL offering, where the binary package
includes an extensive set of alternative shared objects (DLLs) for various CPU
alternatives.

Source builds
`````````````

See "Detailed Description" below. A source build where the packager knows
details of the target machine could theoretically produce a smaller binary by
choosing to compile only the loops needed by the target via command line
arguments.

How to run benchmarks to assess performance benefits
````````````````````````````````````````````````````

Adding more code which use intrinsics will make the code harder to maintain.
Therefore, such code should only be added if it yields a significant
performance benefit. Assessing this performance benefit can be nontrivial.
To aid with this, the implementation for this NEP will add a way to select
which instruction sets can be used at *runtime* via environment variables.
(name TBD). This ablility is critical for CI code verification.


Diagnostics
```````````

A new dictionary `__cpu_features__` will be available to python. The keys are
the available features, the value is a boolean whether the feature is available
or not. Various new private
C functions will be used internally to query available features. These
might be exposed via specific c-extension modules for testing.


Workflow for adding a new CPU architecture-specific optimization
````````````````````````````````````````````````````````````````

NumPy will always have a baseline C implementation for any code that may be
a candidate for SIMD vectorization.  If a contributor wants to add SIMD
support for some architecture (typically the one of most interest to them),
this is the proposed workflow:

TODO (see https://github.com/numpy/numpy/pull/13516#issuecomment-558859638,
needs to be worked out more)

Reuse by other projects
```````````````````````

It would be nice if the universal intrinsics would be available to other
libraries like SciPy or Astropy that also build ufuncs, but that is not an
explicit goal of the first implementation of this NEP.

Backward compatibility
----------------------

There should be no impact on backwards compatibility.


Detailed description
--------------------

Two new build options are available to ``runtests.py`` and ``setup.py build``.
The absolute minimum required features to compile are defined by
``--cpu-baseline``.  For instance, on ``x86_64`` this defaults to ``SSE3``. The
set of additional intrinsics that can be detected and used as sets of
requirements to dispatch on are set by ``--cpu-dispatch``. For instance, on
``x86_64`` this defaults to ``[SSSE3, SSE41, POPCNT, SSE42, AVX, F16C, XOP,
FMA4, FMA3, AVX2, AVX512F, AVX512CD, AVX512_KNL, AVX512_KNM, AVX512_SKX,
AVX512_CLX, AVX512_CNL, AVX512_ICL]``. These features are all mapped to a
c-level boolean array ``npy__cpu_have``, and a c-level convenience function
``npy_cpu_have(int feature_id)`` queries this array.

The CPU-specific features are then mapped to unversal intrinsics which are
for all x86 SIMD variants, ARM SIMD variants etc. For example, the
NumPy universal intrinsic ``npyv_load_u32`` maps to:

*  ``vld1q_u32`` for ARM based NEON
* ``_mm256_loadu_si256`` for x86 based AVX2 
* ``_mm512_loadu_si512`` for x86 based AVX-512

Anyone writing a SIMD loop will use the ``npyv_load_u32`` macro instead of the
architecture specific intrinsic. The code also supplies guard macros for
compilation and runtime, so that the proper loops can be chosen.

Related Work
------------

- PIXMAX TBD: what is it?
- `Eigen`_ is a C++ template library for linear algebra: matrices, vectors,
  numerical solvers, and related algorithms. It is a higher level-abstraction
  than the intrinsics discussed here.
- `xsimd`_ is a header-only C++ library for x86 and ARM that implements the
  mathematical functions used in the algorithms of ``boost.SIMD``.
- OpenCV used to have the one-implementation-per-architecture design, but more
  recently moved to a design that is quite similar to what is proposed in this
  NEP. The top-level `dispatch code`_ includes a `generic header`_ that is
  `specialized at compile time`_ by the CMakefile system.


Implementation
--------------

Current PRs:

- `gh-13421 improve runtime detection of CPU features <https://github.com/numpy/numpy/pull/13421>`_
- `gh-13516: enable multi-platform SIMD compiler optimizations <https://github.com/numpy/numpy/pull/13516>`_

The compile-time and runtime code infrastructure are supplied by the first PR.
The second adds a demonstration of use of the infrastructure for a loop. Once
the NEP is approved, more work is needed to write loops using the machnisms
provided by the NEP.

Alternatives
------------

A proposed alternative in gh-13516_ is to implement loops for each CPU
architecture separately by hand, without trying to abstract common patterns in
the SIMD intrinsics (e.g., have `loops.avx512.c.src`, `loops.avx2.c.src`,
`loops.sse.c.src`, `loops.vsx.c.src`, `loops.neon.c.src`, etc.). This is more
similar to what PIXMAX does. There's a lot of duplication here though, and the
manual code duplication requires a champion who will be dedicated to
implementing and maintaining that platform's loop code.


Discussion
----------

.. note::
    Will include a summary of the discussion once the NEP is published

References and Footnotes
------------------------

.. _`build alternative loops`: https://github.com/numpy/numpy/blob/v1.17.4/numpy/core/code_generators/generate_umath.py#L50
.. _`is chosen`: https://github.com/numpy/numpy/blob/v1.17.4/numpy/core/code_generators/generate_umath.py#L1038
.. _`gh-11113"`: https://github.com/numpy/numpy/pull/11113
.. _`fast avx512 routines`: https://github.com/numpy/numpy/pulls?q=is%3Apr+avx512+is%3Aclosed

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. _`xsimd`: https://xsimd.readthedocs.io/en/latest/
.. _`Eigen`: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _`dispatch code`: https://github.com/opencv/opencv/blob/4.1.2/modules/core/src/arithm.dispatch.cpp
.. _`generic header`: https://github.com/opencv/opencv/blob/4.1.2/modules/core/src/arithm.simd.hpp
.. _`specialized at compile time`: https://github.com/opencv/opencv/blob/4.1.2/modules/core/CMakeLists.txt#L3-#L13
.. _`intrinsics`: https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-intrinsics
.. _`universal intrinsics`: https://docs.opencv.org/master/df/d91/group__core__hal__intrin.html

Copyright
---------

This document has been placed in the public domain. [1]_
