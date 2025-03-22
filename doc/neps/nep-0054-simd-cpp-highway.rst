.. _NEP54:

===================================================================================
NEP 54 — SIMD infrastructure evolution: adopting Google Highway when moving to C++
===================================================================================

:Author: Sayed Adel, Jan Wassenberg, Matti Picus, Ralf Gommers, Chris Sidebottom
:Status: Accepted
:Type: Standards Track
:Created: 2023-07-06
:Resolution: TODO


Abstract
--------

We are moving the SIMD intrinsic framework, Universal Intrinsics, from C to
C++. We have also moved to Meson as the build system. The Google Highway
intrinsics project is proposing we use Highway instead of our Universal
Intrinsics as described in :ref:`NEP 38 <NEP38>`. This is a complex and multi-faceted
decision - this NEP is an attempt to describe the trade-offs involved and
what would need to be done.

Motivation and Scope
--------------------

We want to refactor the C-based Universal Intrinsics (see :ref:`NEP 38
<NEP38>`) to C++. This work was ongoing for some time, and Google's Highway
was suggested as an alternative, which was already written in C++ and had 
support for scalable SVE and other reusable components (such as VQSort).
    
The move from C to C++ is motivated by (a) code readability and ease of
development, (b) the need to add support for sizeless SIMD instructions (e.g.,
ARM's SVE, RISC-V's RVV).

As an example of the readability improvement, here is a typical line of C code
from our current C universal intrinsics framework:

.. code::

   // The @name@ is the numpy-specific templating in .c.src files
   npyv_@sfx@  a5 = npyv_load_@sfx@(src1 + npyv_nlanes_@sfx@ * 4);

This will change (as implemented in PR `gh-21057`_) to:

.. code:: C++

   auto a5 = Load(src1 + nlanes * 4);

If the above C++ code were to use Highway under the hood it would look quite
similar, it uses similarly understandable names as ``Load`` for individual
portable intrinsics.

The ``@sfx`` in the C version above is the template variable for type
identifiers, e.g.: ``#sfx = u8, s8, u16, s16, u32, s32, u64, s64, f32, f64#``.
Explicit use of bitsize-encoded types like this won't work for sizeless SIMD
instruction sets. With C++ this is easier to handle; PR `gh-21057`_ shows how
and contains more complete examples of what the C++ code will look like.

The scope of this NEP includes discussing most relevant aspects of adopting
Google Highway to replace our current Universal Intrinsics framework, including
but not limited to:

- Maintainability, domain expertise availability, ease of onboarding new
  contributor, and other social aspects,
- Key technical differences and constraints that may impact NumPy's internal
  design or performance,
- Build system related aspects,
- Release timing related aspects.

Out of scope (at least for now) is revisiting other aspects of our current SIMD
support strategy:

- accuracy vs. performance trade-offs when adding SIMD support to a function
- use of SVML and x86-simd-sort (and possibly its equivalents for aarch64)
- pulling in individual bits or algorithms of Highway (as in `gh-24018`_) or
  SLEEF (as discussed in that same PR)


Usage and Impact
----------------

N/A - there will be no significant user-visible changes.


Backward compatibility
----------------------

There will be no changes in user-facing Python or C APIs: all the methods to
control compilation and runtime CPU feature selection should remain, although
there may be some changes due to moving to C++ without regards to the
Highway/Universal Intrinsics choice.

The naming of the CPU features in Highway is different from that of the
Universal Intrinsics (see "Supported features/targets" below)

On Windows, MSVC may have to be avoided, as a result of Highway's use of
pragmas which are less well supported by MSVC. This means that we likely have
to build our wheels with clang-cl or Mingw-w64. Both of those should work - we
merged clang-cl support a while back (see `gh-20866`_), and SciPy builds with
Mingw-w64. It may however impact other redistributors or end users who build
from source on Windows.

In response to the earlier discussions around this NEP, Highway is now
dual-licensed as Apache 2 / BSD-3.


High-level considerations
-------------------------

.. note::

   Currently this section attempts to cover each topic separately, and
   comparing the future use of a NumPy-specific C++ implementation vs. use of
   Google Highway with our own numerical routines on top of that. It does not
   (yet) assume a decision or proposed decision is made. Hence this NEP is not
   "this is proposed" with another option in the Alternatives section, but
   rather a side-by-side comparison.

    
Development effort and long-term maintainability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving to Highway is likely to be a significant development effort.
Longer-term, this will hopefully be offset by Highway itself having more
maintainer bandwidth to deal with ongoing issues in compiler support and adding
new platforms. 

Highway being used by other projects, like Chromium and `JPEG XL`_ (see
`this more complete list <https://google.github.io/highway/en/master/README.html#examples>`__
in the Highway documentation), does imply that there is likely to be a benefit
of a wider range of testing and bug reporting/fixing.

One concern is that new instructions may have to be added, and that that is
often best done as part of the process of developing the numerical kernel that
needs the instruction. This will be a little more clumsy if the instruction
lives in Highway which is a git submodule inside the NumPy repo - there will be
a need to implement a temporary/generic version first, and then update the
submodule after upstreaming the new intrinsic.

Documentation-wise, Highway would be a clear win. NumPy's
`CPU/SIMD Optimizations`_ docs are fairly sparse compared to
`the Highway docs`_.

Migration strategy - can it be gradual?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
This is a story of two halves. Moving to Highway's statically dispatched
intrinsics could be done gradually, as already seen in PR `gh-24018`_. However,
adopting Highway's way of performing runtime dispatching has to be done in one
go - we can't (or shouldn't) have two ways of doing that.


Highway policies for compiler and platform support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
When adding new instructions, Highway has a policy that they must be
implemented in a way that fairly balances across CPU architectures.

Regarding the support status and whether all currently-supported architectures
will remain supported, Jan stated that Highway can commit to the following:

1. If it cross-compiles with Clang and can be tested via standard QEMU, it can
   go into Highway's CI.
2. If it cross-compiles via clang/gcc and can be tested with a new QEMU
   (possibly with extra flags), then it can be support via manual testing
   before each Highway release.
3. Existing targets will remain supported as long as they compile/run in QEMU.

Highway is not subject to Google's "no longer supported" strategy (or, as
written in its README, *This is not an officially supported Google product*).
That is not a bad thing; it means that it is less likely to go unsupported due
to a Google business decision about the project. Quite a few well-known open
source projects under the ``google`` GitHub org state this, e.g. `JAX`_ and
`tcmalloc`_.


Supported features/targets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both frameworks support a large set of platforms and SIMD instruction sets,
as well as generic scalar/fallback versions. The main differences right now are:

- NumPy supports IBM Z-system (s390x, VX/VXE/VXE2) while Highway supports Z14, Z15.
- Highway supports ARM SVE/SVE2 and RISC-V RVV (sizeless instructions), while
  NumPy does not.

  - The groundwork for sizeless SIMD support in NumPy has been done in
    `gh-21057`_, however SVE/SVE2 and RISC-V are not yet implemented there.

There is also a difference in the granularity of instruction set groups: NumPy
supports a more granular set of architectures than Highway. See the list of
targets for Highway `here <https://github.com/google/highway/#targets>`__
(it's roughly per CPU family) and for NumPy
`here <https://numpy.org/doc/1.25/reference/simd/build-options.html#supported-features>`__
(roughly per SIMD instruction set). Hence with Highway we'd lose some
granularity - but that is probably fine, we don't really need this level of
granularity, and there isn't much evidence that users explicitly play with this
to squeeze out the last bit of performance for their own CPU.


Compilation strategy for multiple targets and runtime dispatching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Highway compiles once while using preprocessing tricks to generate multiple
stanzas for each CPU feature within the same compilation unit (see the
``foreach_target.h`` usage and dynamic dispatch docs for how that is done).
Universal Intrinsics generate multiple compilation units, one for each CPU
feature group, and compiles multiple times, linking them all together (with
different names) for runtime dispatch. The Highway technique may not work
reliably on MSVC, the Universal Intrinsic technique does work on MSVC.

Which one is more robust? The experts disagree. Jan thinks that the Highway
approach is more robust and in particular avoids the linker pulling in
functions with too-new instructions into the final binary. Sayed thinks that
the current NumPy approach (also used by OpenCV) is more robust, and in
particular is less likely to run into compiler-specific bugs or catch them
earlier. Both agree the meson build system allows specifying object link order,
which produces more consistent builds. However that does tie NumPy to meson.

Matti and Ralf think the current build strategy is working well for NumPy and
the advantages of changing the build and runtime dispatch, with possible
unknown instabilities outweighs the advantages that adopting Highway's dynamic
dispatch may bring.

Our experience of the past four years says that bugs with "invalid instruction"
type crashes are invariably due to issues with feature detection - most often
because users are running under emulation, and sometimes because there are
actual issues with our CPU feature detection code. There is little evidence
we're aware of of the linker pulling in a function which is compiled multiple
times for different architectures and picking the one with unsupported
instructions. To ensure to avoid the issue, it's advisable to keep numerical
kernels inside the source code and refrain from defining non-inlined functions
within cache-able objects.


C++ refactoring considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We want to move from C to C++, which will naturally involve a significant
amount of refactoring, for two main reasons:

- get rid of the NumPy-specific templating language for more expressive C++
- this would make using sizeless intrinsics (like for SVE) easier.

In addition, we see the following considerations:

- If we use Highway, we would need to switch the C++ wrappers from universal
  intrinsics to Highway. On the other hand, the work to move to C++ is not
  complete.
- If we use Highway, we'd need to rewrite existing kernels using Highway
  intrinsics. But again, moving to C++ requires touching all those kernels
  anyway.
- One concern regarding Highway was whether it is possible to obtain a function
  pointer for an architecture-specific function instead of calling that
  function directly. This so that we can be sure that calling 1-D inner loop
  many times for a single Python API invocation does not incur the dispatching
  overhead many times. This was investigated: this can be done with Highway
  too.
- A second concern was whether it's possible with Highway to allow the user at
  runtime to select or disable dispatching to certain instruction sets. This is
  possible.
- Use of tags in Highway's C++ implementation reduces code duplication but the
  added templating makes C-level testing and tracing more complicated.


The ``_simd`` unit testing module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewriting the ``_simd testing`` module to use C++ was done very recently in PR
`gh-24069`_. It depends on the main PR for the move to C++, `gh-21057`_.
It allows one to access the C++ intrinsics with almost the same signature, but
from Python. This is a great way not only for testing, but also for designing
new SIMD kernels.

It may be possible to add a similar testing and prototyping feature to Highway
(which uses plain ``googletest``), however currently the NumPy way is quite a
bit nicer.


Math routines
~~~~~~~~~~~~~

Math or numerical routines are written at a higher level of abstraction than
the universal intrinsics that are the main focus of this NEP. Highway has only
a limited number of math routines, and they are not precise enough for NumPy's
needs. So either way, NumPy's existing routines (which use universal
intrinsics) will stay, and if we go the Highway route they'll simply have to
use Highway primitives internally. We could still use Highway sorting routines.
If we do accept lower-precision routines (via a user-supplied choice, i.e.
extending ``errstate`` to allow a precision option), we could use
Highway-native routines.

There may be other libraries that have numerical routines that can be reused in
NumPy (e.g., from SLEEF, or perhaps from JPEG XL or some other Highway-using
libraries). There may be a small benefit here, but likely it doesn't matter too
much.


Supported and missing intrinsics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some specific intrinsics that NumPy needs may be missing from Highway.
Similarly, some intrinsics that NumPy needs to implement routines are already
implemented in Highway and are missing from NumPy.

Highway has more instructions that NumPy's universal intrinsics, so it's 
possible that some future needs for NumPy kernels may already be met there.

Either way, we will always have to implement intrinsics in either solution.


Related Work
------------

- `Google Highway`_
- `Xsimd`_
- OpenCV's SIMD framework (`API reference <https://docs.opencv.org/4.x/df/d91/group__core__hal__intrin.html>`__, `docs <https://github.com/opencv/opencv/wiki/CPU-optimizations-build-options>`__)
- `std::experimental::simd <https://en.cppreference.com/w/cpp/experimental/simd/simd>`__
- See the Related Work section in :ref:`NEP38` for more related work (as of 2019)


Implementation
--------------

TODO



Alternatives
------------

Use Google Highway for dynamic dispatch. Other alternatives include: do nothing and
stay with C universal intrinsics, use `Xsimd`_ as the SIMD framework (less
comprehensive than Highway - no SVE or PowerPC support for example), or
use/vendor `SLEEF`_ (a good library, but inconsistently maintained). Neither of
these alternatives seems appealing.


Discussion
----------




References and Footnotes
------------------------

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/
.. _`gh-20866`: https://github.com/numpy/numpy/pull/20866
.. _`gh-21057`: https://github.com/numpy/numpy/pull/21057
.. _`gh-23096`: https://github.com/numpy/numpy/pull/23096
.. _`gh-24018`: https://github.com/numpy/numpy/pull/24018
.. _`gh-24069`: https://github.com/numpy/numpy/pull/24069
.. _JPEG XL: https://github.com/libjxl/libjxl
.. _CPU/SIMD Optimizations: https://numpy.org/doc/1.25/reference/simd/
.. _the Highway docs: https://google.github.io/highway/
.. _Google Highway: https://github.com/google/highway/
.. _Xsimd: https://github.com/xtensor-stack/xsimd
.. _SLEEF: https://sleef.org/
.. _tcmalloc: https://github.com/google/tcmalloc
.. _JAX: https://github.com/google/jax

Copyright
---------

This document has been placed in the public domain. [1]_
