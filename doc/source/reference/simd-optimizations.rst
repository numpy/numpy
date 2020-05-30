******************
SIMD Optimizations
******************

NumPy provides a set of macros that define `Universal Intrinsics`_, abstracting
typical platform-specific intrinsics so SIMD code needs to be written only
once. There are three layers:

- Code is written using the universal intrinsic macros, with guards that
  will enable use of the macros only when the compiler recognizes them.
  In NumPy, these are used to construct multiple ufunc loops. Current policy is
  to create three loops: One loop is the default and uses no intrinsics. One
  uses the minimum intrinsics required on the architecture. And the thirs is
  written using the maximum set of intrinsics possible.
- At compile time, these macros are overlayed with the appropriate platform /
  architecture intrinsics, and the three loops compiled.
- At runtime import, the CPU is probed for the set of supported intrinsic
  features. A mechanism is used to grab the pointer to the most appropriate
  function, and this will be the one called.


Build options
=============

- ``--cpu-baseline`` minimal set of required optimizations, default
  value is ``min`` which provides the minimum CPU features that can
  safely run on a wide range of users platforms.

- ``--cpu-dispatch`` dispatched set of additional optimizations,
  the default value is ``max -xop -fma4`` which enables all CPU
  features, except for AMD legacy features.

The command arguments can be reached through ``build``, ``build_clib``, ``build_ext``.
if ``build_clib`` or ``build_ext`` are not specified by the user, the arguments of
``build`` will be used instead, which also holds the default values.

Optimization names can be CPU features or group of features that gather several features or
special options perform a series of procedures.


The following tables show the current supported optimizations sorted from the lowest to the highest interest.

``X86`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

    ============  ===================================================================
     Name          Implies
    ============  ===================================================================
    ============  ===================================================================

``X86`` - Group names
~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

    ==============  ================================== ============================================
      Name          Gather                                            Implies
    ==============  ================================== ============================================
    ==============  ================================== ============================================

``IBM/POWER``  - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

    ============  =================
     Name          Implies
    ============  =================
    ============  =================

``ARM`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

    ===============  ================================================================
     Name            Implies
    ===============  ================================================================
    ===============  ================================================================

Special options
~~~~~~~~~~~~~~~

- ``NONE``: enable no features

- ``NATIVE``: Enables all CPU features that supported by the current
   machine, this operation is based on the compiler flags (``-march=native, -xHost, /QxHost``)

- ``MIN``: Enables the minimum CPU features that can safely run on a wide range of users platforms:

.. table::
    :align: left

    ======================================  =======================================
     For Arch                               Returns
    ======================================  =======================================
     ``x86``                                ``SSE`` ``SSE2``
     ``x86`` ``64-bit mode``                ``SSE`` ``SSE2`` ``SSE3``
     ``IBM/POWER`` ``big-endian mode``      ``NONE``
     ``IBM/POWER`` ``little-endian mode``   ``VSX`` ``VSX2``
     ``ARMHF``                              ``NONE``
     ``ARM64`` ``AARCH64``                  ``NEON`` ``NEON_FP16`` ``NEON_VFPV4``
                                            ``ASIMD``
    ======================================  =======================================

- ``MAX``: Enables all supported CPU features by the Compiler and platform.

- ``Operators-/+``: remove or add features, useful with options ``MAX``, ``MIN`` and ``NATIVE``.

NOTES
~~~~~~~~~~~~~
- Case-insensitive among all CPU features and other options.

- Comma or space can be used as a separator, e.g. ``--cpu-dispatch``\ = "avx2 avx512f" or
  ``--cpu-dispatch``\ = "avx2, avx512f" both applicable.

- operand ``+`` is only added for nominal reason, For example:
    - ``--cpu-basline``="min avx2" equivalent to ``--cpu-basline``="min + avx2"
    - ``--cpu-basline``="min,avx2" equivalent to ``--cpu-basline``="min,+avx2"

- If the CPU feature is not supported by the user platform or
  compiler, it will be skipped rather than raising a fatal error.

- Any specified CPU features to ``--cpu-dispatch`` will be skipped if
  it's part of CPU baseline features

- Argument ``--cpu-baseline`` force enables implied features,
  e.g. ``--cpu-baseline``\ ="sse42" equivalent to
  ``--cpu-baseline``\ ="sse sse2 sse3 ssse3 sse41 popcnt sse42"

- The value of ``--cpu-baseline`` will be treated as "native" if
  compiler native flag ``-march=native`` or ``-xHost`` or ``QxHost`` is
  enabled through environment variable ``CFLAGS``

- The user should always check the final report through the build log
  to verify the enabled features.


Special cases
~~~~~~~~~~~~~

Behaviors and Errors
~~~~~~~~~~~~~~~~~~~~

Usage and Examples
~~~~~~~~~~~~~~~~~~

Report and Trace
~~~~~~~~~~~~~~~~

Understanding CPU Dispatching, How the NumPy dispatcher works?
==============================================================

NumPy dispatcher is based on multi-source compiling, which means taking
a certain source and compile it multiple times with variant compiler
flags depend on the required optimizations, then combining the returned
objects together.

| This mechanism is very friendly with all compilers and it doesn't
  require any compiler-specific extension,
| but at the same time it takes a long process that has a sequence of
  steps, which explained as follows:

1. Configuring the required optimization by the user before starting to
   build the source files,

   The required optimizations can be configured through two command
   arguments:

   -  ``--cpu-baseline`` minimal set of required optimizations.

   -  ``--cpu-dispatch`` dispatched set of additional optimizations.

2. Discovering the environment

   In this step, we check what kind compiler and architecture we deal
   with, also handling the caching process which is really important
   to speed up the rebuilding.

3. Parsing the command arguments, we have a very unique syntax that
   gives the user ability to easily manage the optimizations. see
   **TODO**

4. Validating the required optimizations

   By testing it against the compilers, and see what compiler can
   support, according to the required optimizations. the validating
   process isn't strict, for example, if the user requested ``AVX2``
   but the compiler doesn't support it then we just skip it and
   returns the maximum optimization that can handle it by the compiler
   depending on the implied features of ``AVX2``, let us assume
   ``AVX``.

5. Generating the main configuration header(\ ``_cpu_dispatch.h``)

   This header contains all the definitions and headers of
   instruction-sets for the required optimizations that have been
   validated during the previous step.

   It also contains extra definitions that used in defining NumPy
   module's attributes ``__cpu_baseline__`` and ``__cpu_dispatch__``.

   **But how this header looks like?**

   Well let's see how it looks on X86 because the header is dynamically
   generated according to what kinda compiler and architecture we have,
   also assume the compiler supports these features and it had been
   successfully configured through ``--cpu-baseline`` and
   ``--cpu-dispatch``

   .. code:: c

      // it should be located at numpy/numpy/core/src/common/_cpu_dispatch.h
      /**NOTE
       ** C defentions that prefixed with "NPY_HAVE_" are representiong
       ** the required defentions.
       **
       ** C defentions that prefixed with 'NPY__CPU_TARGET_' are protected and
       ** shouldn't be used by any NumPy C sources.
       */
      /******* baseline features *******/
      /** SSE **/
      #define NPY_HAVE_SSE 1
      #include <xmmintrin.h>
      /** SSE2 **/
      #define NPY_HAVE_SSE2 1
      #include <emmintrin.h>
      /** SSE3 **/
      #define NPY_HAVE_SSE3 1
      #include <pmmintrin.h>
      /******* dispatch features *******/
      #ifdef NPY__CPU_TARGET_SSSE3
        /** SSSE3 **/
          #define NPY_HAVE_SSSE3 1
          #include <tmmintrin.h>
      #endif
      #ifdef NPY__CPU_TARGET_SSE41
        #define NPY_HAVE_SSE41 1
        #include <smmintrin.h>
      #endif

   **baseline features** is our minimal set of required optimizations that been configured by
   ``--cpu-baseline``, it has no preprocessor guards and always on.
   That's mean it can be used in any source.

      *Wait here!! Does NumPy's infrastructure pass the compiler's flags
      of baseline features to all sources?*

   Definitely, yes! but wait **dispatch-able sources** treated
   differently.

      *What is **dispatch-able sources**?*

   Please just continue reading, you will find your answer in the next
   steps.

      *Hey wait, What if the user specifies certain **baseline
      features** during the build but the running machine doesn't
      support these kinds of CPU features and at the same time
      there's instruction-sets lay down in a C source activated by one
      of these definitions or maybe the compiler itself auto-generated/vectorized certain
      piece of code depending on the provided flags?*

   Well during the loading of the NumPy module, there's a validating process detecting
   this behavior that raising a Python runtime error to inform the user. otherwise,
   the CPU/Kernel going to interrupt the execution process by raising an illegal instruction error.

   **dispatch features** **TOOD**

6. Dispatch-able sources and configuration statements **TODO**

The baseline
~~~~~~~~~~~~


Dispatcher
~~~~~~~~~~


Groups and Policies
~~~~~~~~~~~~~~~~~~~



Examples
~~~~~~~~


Report and Trace
~~~~~~~~~~~~~~~~


.. _`Universal Intrinsics`: https://numpy.org/neps/nep-0038-SIMD-optimizations.html

