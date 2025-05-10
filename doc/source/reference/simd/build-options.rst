*****************
CPU build options
*****************

Description
-----------

The following options are mainly used to change the default behavior of optimizations
that target certain CPU features:

- ``cpu-baseline``: minimal set of required CPU features.
   Default value is ``min`` which provides the minimum CPU features that can
   safely run on a wide range of platforms within the processor family.

   .. note::

     During the runtime, NumPy modules will fail to load if any of specified features
     are not supported by the target CPU (raises Python runtime error).

- ``cpu-dispatch``: dispatched set of additional CPU features.
   Default value is ``max -xop -fma4`` which enables all CPU
   features, except for AMD legacy features (in case of X86).

   .. note::

      During the runtime, NumPy modules will skip any specified features
      that are not available in the target CPU.

These options are accessible at build time by passing setup arguments to meson-python
via the build frontend (e.g., ``pip`` or ``build``).
They accept a set of :ref:`CPU features <opt-supported-features>`
or groups of features that gather several features or
:ref:`special options <opt-special-options>` that
perform a series of procedures.

To customize CPU/build options::

    pip install . -Csetup-args=-Dcpu-baseline="avx2 fma3" -Csetup-args=-Dcpu-dispatch="max"

Quick start
-----------

In general, the default settings tend to not impose certain CPU features that
may not be available on some older processors. Raising the ceiling of the
baseline features will often improve performance and may also reduce
binary size.


The following are the most common scenarios that may require changing
the default settings:


I am building NumPy for my local use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

And I do not intend to export the build to other users or target a
different CPU than what the host has.

Set ``native`` for baseline, or manually specify the CPU features in case of option
``native`` isn't supported by your platform::

    python -m build --wheel -Csetup-args=-Dcpu-baseline="native"

Building NumPy with extra CPU features isn't necessary for this case,
since all supported features are already defined within the baseline features::

    python -m build --wheel -Csetup-args=-Dcpu-baseline="native" \
    -Csetup-args=-Dcpu-dispatch="none"

.. note::

    A fatal error will be raised if ``native`` isn't supported by the host platform.

I do not want to support the old processors of the x86 architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since most of the CPUs nowadays support at least ``AVX``, ``F16C`` features, you can use::

    python -m build --wheel -Csetup-args=-Dcpu-baseline="avx f16c"

.. note::

    ``cpu-baseline`` force combine all implied features, so there's no need
    to add SSE features.


I'm facing the same case above but with ppc64 architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then raise the ceiling of the baseline features to Power8::

    python -m build --wheel -Csetup-args=-Dcpu-baseline="vsx2"

Having issues with AVX512 features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may have some reservations about including of ``AVX512`` or
any other CPU feature and you want to exclude from the dispatched features::

    python -m build --wheel -Csetup-args=-Dcpu-dispatch="max -avx512f -avx512cd \
    -avx512_knl -avx512_knm -avx512_skx -avx512_clx -avx512_cnl -avx512_icl -avx512_spr"

.. _opt-supported-features:

Supported features
------------------

The names of the features can express one feature or a group of features,
as shown in the following tables supported depend on the lowest interest:

.. note::

    The following features may not be supported by all compilers,
    also some compilers may produce different set of implied features
    when it comes to features like ``AVX512``, ``AVX2``, and ``FMA3``.
    See :ref:`opt-platform-differences` for more details.

.. include:: generated_tables/cpu_features.inc

.. _opt-special-options:

Special options
---------------

- ``NONE``: enable no features.

- ``NATIVE``: Enables all CPU features that supported by the host CPU,
  this operation is based on the compiler flags (``-march=native``, ``-xHost``, ``/QxHost``)

- ``MIN``: Enables the minimum CPU features that can safely run on a wide range of platforms:

  .. table::
      :align: left

      ======================================  =======================================
       For Arch                               Implies
      ======================================  =======================================
       x86 (32-bit mode)                      ``SSE`` ``SSE2``
       x86_64                                 ``SSE`` ``SSE2`` ``SSE3``
       IBM/POWER (big-endian mode)            ``NONE``
       IBM/POWER (little-endian mode)         ``VSX`` ``VSX2``
       ARMHF                                  ``NONE``
       ARM64 A.K. AARCH64                     ``NEON`` ``NEON_FP16`` ``NEON_VFPV4``
                                              ``ASIMD``
       IBM/ZSYSTEM(S390X)                     ``NONE``
      ======================================  =======================================

- ``MAX``: Enables all supported CPU features by the compiler and platform.

- ``Operators-/+``: remove or add features, useful with options ``MAX``, ``MIN`` and ``NATIVE``.

Behaviors
---------

- CPU features and other options are case-insensitive, for example::

    python -m build --wheel -Csetup-args=-Dcpu-dispatch="SSE41 avx2 FMA3"

- The order of the requested optimizations doesn't matter::

    python -m build --wheel -Csetup-args=-Dcpu-dispatch="SSE41 AVX2 FMA3"
    # equivalent to
    python -m build --wheel -Csetup-args=-Dcpu-dispatch="FMA3 AVX2 SSE41"

- Either commas or spaces or '+' can be used as a separator,
  for example::

    python -m build --wheel -Csetup-args=-Dcpu-dispatch="avx2 avx512f"
    # or
    python -m build --wheel -Csetup-args=-Dcpu-dispatch=avx2,avx512f
    # or
    python -m build --wheel -Csetup-args=-Dcpu-dispatch="avx2+avx512f"

  all works but arguments should be enclosed in quotes or escaped
  by backslash if any spaces are used.

- ``cpu-baseline`` combines all implied CPU features, for example::

    python -m build --wheel -Csetup-args=-Dcpu-baseline=sse42
    # equivalent to
    python -m build --wheel -Csetup-args=-Dcpu-baseline="sse sse2 sse3 ssse3 sse41 popcnt sse42"

- ``cpu-baseline`` will be treated as "native" if compiler native flag
  ``-march=native`` or ``-xHost`` or ``/QxHost`` is enabled through environment variable
  ``CFLAGS``::

    export CFLAGS="-march=native"
    pip install .
    # is equivalent to
    pip install . -Csetup-args=-Dcpu-baseline=native

- ``cpu-baseline`` escapes any specified features that aren't supported
  by the target platform or compiler rather than raising fatal errors.

  .. note::

       Since ``cpu-baseline`` combines all implied features, the maximum
       supported of implied features will be enabled rather than escape all of them.
       For example::

          # Requesting `AVX2,FMA3` but the compiler only support **SSE** features
          python -m build --wheel -Csetup-args=-Dcpu-baseline="avx2 fma3"
          # is equivalent to
          python -m build --wheel -Csetup-args=-Dcpu-baseline="sse sse2 sse3 ssse3 sse41 popcnt sse42"

- ``cpu-dispatch`` does not combine any of implied CPU features,
  so you must add them unless you want to disable one or all of them::

    # Only dispatches AVX2 and FMA3
    python -m build --wheel -Csetup-args=-Dcpu-dispatch=avx2,fma3
    # Dispatches AVX and SSE features
    python -m build --wheel -Csetup-args=-Dcpu-dispatch=ssse3,sse41,sse42,avx,avx2,fma3

- ``cpu-dispatch`` escapes any specified baseline features and also escapes
  any features not supported by the target platform or compiler without raising
  fatal errors.

Eventually, you should always check the final report through the build log
to verify the enabled features. See :ref:`opt-build-report` for more details.

.. _opt-platform-differences:

Platform differences
--------------------

Some exceptional conditions force us to link some features together when it come to
certain compilers or architectures, resulting in the impossibility of building them separately.

These conditions can be divided into two parts, as follows:

**Architectural compatibility**

The need to align certain CPU features that are assured to be supported by
successive generations of the same architecture, some cases:

- On ppc64le ``VSX(ISA 2.06)`` and ``VSX2(ISA 2.07)`` both imply one another since the
  first generation that supports little-endian mode is Power-8`(ISA 2.07)`
- On AArch64 ``NEON NEON_FP16 NEON_VFPV4 ASIMD`` implies each other since they are part of the
  hardware baseline.

For example::

    # On ARMv8/A64, specify NEON is going to enable Advanced SIMD
    # and all predecessor extensions
    python -m build --wheel -Csetup-args=-Dcpu-baseline=neon
    # which is equivalent to
    python -m build --wheel -Csetup-args=-Dcpu-baseline="neon neon_fp16 neon_vfpv4 asimd"

.. note::

    Please take a deep look at :ref:`opt-supported-features`,
    in order to determine the features that imply one another.

**Compilation compatibility**

Some compilers don't provide independent support for all CPU features. For instance
**Intel**'s compiler doesn't provide separated flags for ``AVX2`` and ``FMA3``,
it makes sense since all Intel CPUs that comes with ``AVX2`` also support ``FMA3``,
but this approach is incompatible with other **x86** CPUs from **AMD** or **VIA**.

For example::

    # Specify AVX2 will force enables FMA3 on Intel compilers
    python -m build --wheel -Csetup-args=-Dcpu-baseline=avx2
    # which is equivalent to
    python -m build --wheel -Csetup-args=-Dcpu-baseline="avx2 fma3"


The following tables only show the differences imposed by some compilers from the
general context that been shown in the :ref:`opt-supported-features` tables:

.. note::

    Features names with strikeout represent the unsupported CPU features.

.. raw:: html

    <style>
        .enabled-feature {color:green; font-weight:bold;}
        .disabled-feature {color:red; text-decoration: line-through;}
    </style>

.. role:: enabled
    :class: enabled-feature

.. role:: disabled
    :class: disabled-feature

.. include:: generated_tables/compilers-diff.inc

.. _opt-build-report:

Build report
------------

In most cases, the CPU build options do not produce any fatal errors that lead to hanging the build.
Most of the errors that may appear in the build log serve as heavy warnings due to the lack of some
expected CPU features by the compiler.

So we strongly recommend checking the final report log, to be aware of what kind of CPU features
are enabled and what are not.

You can find the final report of CPU optimizations at the end of the build log,
and here is how it looks on x86_64/gcc:

.. raw:: html

    <style>#build-report .highlight-bash pre{max-height:450px; overflow-y: scroll;}</style>

.. literalinclude:: log_example.txt
   :language: bash

There is a separate report for each of ``build_ext`` and ``build_clib``
that includes several sections, and each section has several values, representing the following:

**Platform**:

- :enabled:`Architecture`: The architecture name of target CPU. It should be one of
  ``x86``, ``x64``, ``ppc64``, ``ppc64le``, ``armhf``, ``aarch64``, ``s390x`` or ``unknown``.

- :enabled:`Compiler`: The compiler name. It should be one of
  gcc, clang, msvc, icc, iccw or unix-like.

**CPU baseline**:

- :enabled:`Requested`: The specific features and options to ``cpu-baseline`` as-is.
- :enabled:`Enabled`: The final set of enabled CPU features.
- :enabled:`Flags`: The compiler flags that were used to all NumPy C/C++ sources
  during the compilation except for temporary sources that have been used for generating
  the binary objects of dispatched features.
- :enabled:`Extra checks`: list of internal checks that activate certain functionality
  or intrinsics related to the enabled features, useful for debugging when it comes
  to developing SIMD kernels.

**CPU dispatch**:

- :enabled:`Requested`: The specific features and options to ``cpu-dispatch`` as-is.
- :enabled:`Enabled`: The final set of enabled CPU features.
- :enabled:`Generated`: At the beginning of the next row of this property,
  the features for which optimizations have been generated are shown in the
  form of several sections with similar properties explained as follows:

  - :enabled:`One or multiple dispatched feature`: The implied CPU features.
  - :enabled:`Flags`: The compiler flags that been used for these features.
  - :enabled:`Extra checks`: Similar to the baseline but for these dispatched features.
  - :enabled:`Detect`: Set of CPU features that need be detected in runtime in order to
    execute the generated optimizations.
  - The lines that come after the above property and end with a ':' on a separate line,
    represent the paths of c/c++ sources that define the generated optimizations.

.. _runtime-simd-dispatch:

Runtime dispatch
----------------
Importing NumPy triggers a scan of the available CPU features from the set
of dispatchable features. This can be further restricted by setting the
environment variable ``NPY_DISABLE_CPU_FEATURES`` to a comma-, tab-, or
space-separated list of features to disable. This will raise an error if
parsing fails or if the feature was not enabled. For instance, on ``x86_64``
this will disable ``AVX2`` and ``FMA3``::

    NPY_DISABLE_CPU_FEATURES="AVX2,FMA3"

If the feature is not available, a warning will be emitted.

Tracking dispatched functions
-----------------------------
Discovering which CPU targets are enabled for different optimized functions is achievable
through the Python function ``numpy.lib.introspect.opt_func_info``.
This function offers the flexibility of applying filters using two optional arguments:
one for refining function names and the other for specifying data types in the signatures.

For example::

   >> func_info = numpy.lib.introspect.opt_func_info(func_name='add|abs', signature='float64|complex64')
   >> print(json.dumps(func_info, indent=2))
   {
     "absolute": {
       "dd": {
         "current": "SSE41",
         "available": "SSE41 baseline(SSE SSE2 SSE3)"
       },
       "Ff": {
         "current": "FMA3__AVX2",
         "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
       },
       "Dd": {
         "current": "FMA3__AVX2",
         "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
       }
     },
     "add": {
       "ddd": {
         "current": "FMA3__AVX2",
         "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
       },
       "FFF": {
         "current": "FMA3__AVX2",
         "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
       }
    }
  }

