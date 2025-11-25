*****************
CPU Build Options
*****************

Overview
--------

NumPy provides configuration options to optimize performance based on CPU capabilities.
These options allow you to specify which CPU features to support, balancing performance, compatibility, and binary size.
This document explains how to use these options effectively across various CPU architectures.

Key Configuration Options
-------------------------

NumPy uses several build options to control CPU optimizations:

- ``cpu-baseline``: The minimum set of CPU features required to run the compiled NumPy.
  
  * Default: ``min`` (provides compatibility across a wide range of platforms)
  * If your target CPU doesn't support all specified baseline features, NumPy will fail to load with a Python runtime error

- ``cpu-baseline-detect``: controls detection of CPU baseline based on compiler
  flags. Default value is ``auto`` that enables detection if ``-march=``
  or a similar compiler flag is used. The other possible values are ``enabled``
  and ``disabled`` to respective enable or disable it unconditionally.

- ``cpu-dispatch``: Additional CPU features for which optimized code paths will be generated.
  
  * Default: ``max`` (enables all available optimizations)
  * At runtime, NumPy will automatically select the fastest available code path based on your CPU's capabilities

- ``disable-optimization``: Completely disables all CPU optimizations.
  
  * Default: ``false`` (optimizations are enabled)
  * When set to ``true``, disables all CPU optimized code including dispatch, SIMD, and loop unrolling
  * Useful for debugging, testing, or in environments where optimization causes issues

These options are specified at build time via meson-python arguments::

  pip install . -Csetup-args=-Dcpu-baseline="min" -Csetup-args=-Dcpu-dispatch="max"
  # or through spin
  spin build -- -Dcpu-baseline="min" -Dcpu-dispatch="max"

``cpu-baseline`` and ``cpu-dispatch`` can be set to specific :ref:`CPU groups, features<opt-supported-features>`, or :ref:`special options <opt-special-options>`
that perform specific actions. The following sections describe these options in detail.

Common Usage Scenarios
----------------------

Building for Local Use Only
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building for your machine only and not planning to distribute::

  python -m build --wheel -Csetup-args=-Dcpu-baseline="native" -Csetup-args=-Dcpu-dispatch="none"

This automatically detects and uses all CPU features available on your machine.

.. note::
  A fatal error will be raised if ``native`` isn't supported by the host platform.

Excluding Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to exclude certain CPU features from the dispatched features::

  # For x86-64: exclude all AVX-512 features
  python -m build --wheel -Csetup-args=-Dcpu-dispatch="max -X86_V4"

  # For ARM64: exclude SVE
  python -m build --wheel -Csetup-args=-Dcpu-dispatch="max -SVE"

.. note::
  Excluding a feature will also exclude any successor features that are
  implied by the excluded feature. For example, excluding ``X86_V4`` will
  exclude ``AVX512_ICL`` and ``AVX512_SPR`` as well.

Targeting Older CPUs
~~~~~~~~~~~~~~~~~~~~

On ``x86-64``, by default the baseline is set to ``min`` which maps to ``X86_V2``.
This unsuitable for older CPUs (before 2009) or old virtual machines.
To address this, set the baseline to ``none``::

  python -m build --wheel -Csetup-args=-Dcpu-baseline="none"

This will create a build that is compatible with all x86 CPUs, but 
without any manual optimizations or SIMD code paths for the baseline.
The build will rely only on dispatched code paths for optimization.

Targeting Newer CPUs
~~~~~~~~~~~~~~~~~~~~

Raising the baseline improves performance for two main reasons:

1. Dispatched kernels don't cover all code paths
2. A higher baseline leads to smaller binary size as the compiler won't generate code paths for excluded dispatched features

For CPUs from 2015 and newer, setting the baseline to ``X86_V3`` may be suitable::

  python -m build --wheel -Csetup-args=-Dcpu-baseline="min+X86_V3"

.. _opt-supported-features:

Supported CPU Features By Architecture
--------------------------------------

NumPy supports optimized code paths for multiple CPU architectures. Below are the supported feature groups for each architecture.
The name of the feature group can be used in the build options ``cpu-baseline`` and ``cpu-dispatch``.

X86
~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
     - Includes
   * - ``X86_V2``
     - 
     - ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE4_1`` ``SSE4_2`` ``POPCNT`` ``CX16`` ``LAHF``
   * - ``X86_V3``
     - ``X86_V2``
     - ``AVX`` ``AVX2`` ``FMA3`` ``BMI`` ``BMI2`` ``LZCNT`` ``F16C`` ``MOVBE``
   * - ``X86_V4``
     - ``X86_V3``
     - ``AVX512F`` ``AVX512CD`` ``AVX512VL`` ``AVX512BW`` ``AVX512DQ``
   * - ``AVX512_ICL``
     - ``X86_V4``
     - ``AVX512VBMI`` ``AVX512VBMI2`` ``AVX512VNNI`` ``AVX512BITALG`` ``AVX512VPOPCNTDQ`` ``AVX512IFMA`` ``VAES`` ``GFNI`` ``VPCLMULQDQ``
   * - ``AVX512_SPR``
     - ``AVX512_ICL``
     - ``AVX512FP16``

These groups correspond to CPU generations:

- ``X86_V2``: x86-64-v2 microarchitectures (CPUs since 2009)
- ``X86_V3``: x86-64-v3 microarchitectures (CPUs since 2015)
- ``X86_V4``: x86-64-v4 microarchitectures (AVX-512 capable CPUs)
- ``AVX512_ICL``: Intel Ice Lake and similar CPUs
- ``AVX512_SPR``: Intel Sapphire Rapids and newer CPUs

.. note::
    On 32-bit x86, ``cx16`` is excluded from ``X86_V2``.

On IBM/POWER big-endian
~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
   * - ``VSX``
     - 
   * - ``VSX2``
     - ``VSX``
   * - ``VSX3``
     - ``VSX`` ``VSX2``
   * - ``VSX4``
     - ``VSX`` ``VSX2`` ``VSX3``

On IBM/POWER little-endian
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
   * - ``VSX``
     - ``VSX2``
   * - ``VSX2``
     - ``VSX``
   * - ``VSX3``
     - ``VSX`` ``VSX2``
   * - ``VSX4``
     - ``VSX`` ``VSX2`` ``VSX3``

On ARMv7/A32
~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
   * - ``NEON``
     - 
   * - ``NEON_FP16``
     - ``NEON``
   * - ``NEON_VFPV4``
     - ``NEON`` ``NEON_FP16``
   * - ``ASIMD``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4``
   * - ``ASIMDHP``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``
   * - ``ASIMDDP``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``
   * - ``ASIMDFHM``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD`` ``ASIMDHP``

On ARMv8/A64
~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
   * - ``NEON``
     - ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``
   * - ``NEON_FP16``
     - ``NEON`` ``NEON_VFPV4`` ``ASIMD``
   * - ``NEON_VFPV4``
     - ``NEON`` ``NEON_FP16`` ``ASIMD``
   * - ``ASIMD``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4``
   * - ``ASIMDHP``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``
   * - ``ASIMDDP``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``
   * - ``ASIMDFHM``
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD`` ``ASIMDHP``

On IBM/ZSYSTEM(S390X)
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
   * - ``VX``
     - 
   * - ``VXE``
     - ``VX``
   * - ``VXE2``
     - ``VX`` ``VXE``

On RISCV64
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :align: left

   * - Name
     - Implies
   * - ``RVV``
     - 

.. _opt-special-options:

Special Options
---------------

Beyond specific feature names, you can use these special values:

``NONE``
~~~~~~~~

Enables no features (equivalent to an empty string).

``NATIVE``
~~~~~~~~~~

Enables all features supported by the host CPU.

``DETECT``
~~~~~~~~~~

Detects the features enabled by the compiler. This option is appended by default
to ``cpu-baseline`` if ``-march``, ``-mcpu``, ``-xhost``, or ``/QxHost`` is set in 
the environment variable ``CFLAGS`` unless ``cpu-baseline-detect`` is ``disabled``.

``MIN``
~~~~~~~

Enables the minimum CPU features for each architecture:

.. list-table::
   :header-rows: 1
   :align: left

   * - For Arch
     - Implies
   * - x86 (32-bit)
     - ``X86_V2``
   * - x86-64
     - ``X86_V2``
   * - IBM/POWER (big-endian)
     - ``NONE``
   * - IBM/POWER (little-endian)
     - ``VSX`` ``VSX2``
   * - ARMv7/ARMHF
     - ``NONE``
   * - ARMv8/AArch64
     - ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``
   * - IBM/ZSYSTEM(S390X)
     - ``NONE``
   * - riscv64
     - ``NONE``


``MAX``
~~~~~~~

Enables all features supported by the compiler and platform. 

Operator Operators (``-``/``+``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove or add specific features, useful with ``MAX``, ``MIN``, and ``NATIVE``:

- Adding a feature (``+``) includes all implied features
- Removing a feature (``-``) excludes all successor features that imply the removed feature

Examples::

  python -m build --wheel -Csetup-args=-Dcpu-dispatch="max-X86_V4"
  python -m build --wheel -Csetup-args=-Dcpu-baseline="min+X86_V4"

Usage And Behaviors
-------------------

Case Insensitivity
~~~~~~~~~~~~~~~~~~

CPU features and options are case-insensitive::

  python -m build --wheel -Csetup-args=-Dcpu-dispatch="X86_v4"

Mixing Features across Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can mix features from different architectures::

  python -m build --wheel -Csetup-args=-Dcpu-baseline="X86_V4 VSX4 SVE"

Order Independence
~~~~~~~~~~~~~~~~~~

The order of specified features doesn't matter::

  python -m build --wheel -Csetup-args=-Dcpu-dispatch="SVE X86_V4 x86_v3"

Separators
~~~~~~~~~~

You can use spaces or commas as separators::

  # All of these are equivalent
  python -m build --wheel -Csetup-args=-Dcpu-dispatch="X86_V2 X86_V4"
  python -m build --wheel -Csetup-args=-Dcpu-dispatch=X86_V2,X86_V4

Feature Combination
~~~~~~~~~~~~~~~~~~~

Features specified in options are automatically combined with all implied features::

  python -m build --wheel -Csetup-args=-Dcpu-baseline=X86_V4

Equivalent to::

  python -m build --wheel -Csetup-args=-Dcpu-baseline="X86_V2 X86_V3 X86_V4"

Baseline Overlapping 
~~~~~~~~~~~~~~~~~~~~

Features specified in ``cpu-baseline`` will be excluded from the ``cpu-dispatch`` features,
along with their implied features, but without excluding successor features that imply them.

For instance, if you specify ``cpu-baseline="X86_V4"``, it will exclude ``X86_V4`` and its
implied features ``X86_V2`` and ``X86_V3`` from the ``cpu-dispatch`` features.

Compile-time Detection
~~~~~~~~~~~~~~~~~~~~~~

Specifying features to ``cpu-dispatch`` or ``cpu-baseline`` doesn't explicitly enable them.
Features are detected at compile time, and the maximum available features based on your
specified options will be enabled according to toolchain and platform support.

This detection occurs by testing feature availability in the compiler through compile-time
source files containing common intrinsics for the specified features. If both the compiler
and assembler support the feature, it will be enabled.

For example, if you specify ``cpu-dispatch="AVX512_ICL"`` but your compiler doesn't support it,
the feature will be excluded from the build. However, any implied features will still be
enabled if they're supported.


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
  first generation that supports little-endian mode is ``Power-8(ISA 2.07)``
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

.. _opt-build-report:

Build report
------------

In most cases, the CPU build options do not produce any fatal errors that lead to hanging the build.
Most of the errors that may appear in the build log serve as heavy warnings due to the lack of some
expected CPU features by the compiler.

So we strongly recommend checking the final report log, to be aware of what kind of CPU features
are enabled and what are not.

You can find the final report of CPU optimizations by tracing meson build log,
and here is how it looks on x86_64/gcc:

.. raw:: html

    <style>#build-report .highlight-bash pre{max-height:450px; overflow-y: scroll;}</style>

.. literalinclude:: log_example.txt
   :language: bash


.. _runtime-simd-dispatch:

Runtime Dispatch
----------------

Importing NumPy triggers a scan of the available CPU features from the set
of dispatchable features. You can restrict this scan by setting the
environment variable ``NPY_DISABLE_CPU_FEATURES`` to a comma-, tab-, or
space-separated list of features to disable. 

For instance, on ``x86_64`` this will disable ``X86_V4``::

    NPY_DISABLE_CPU_FEATURES="X86_V4"

This will raise an error if parsing fails or if the feature was not enabled through the ``cpu-dispatch`` build option.
If the feature is supported by the build but not available on the current CPU, a warning will be emitted instead.

Tracking Dispatched Functions
-----------------------------

You can discover which CPU targets are enabled for different optimized functions using 
the Python function ``numpy.lib.introspect.opt_func_info``.

This function offers two optional arguments for filtering results:

1. ``func_name`` - For refining function names
2. ``signature`` - For specifying data types in the signatures

For example::

   >> func_info = numpy.lib.introspect.opt_func_info(func_name='add|abs', signature='float64|complex64')
   >> print(json.dumps(func_info, indent=2))
   {
    "absolute": {
      "dd": {
        "current": "baseline(X86_V2)",
        "available": "baseline(X86_V2)"
      },
      "Ff": {
        "current": "X86_V3",
        "available": "X86_V3 baseline(X86_V2)"
      },
      "Dd": {
        "current": "X86_V3",
        "available": "X86_V3 baseline(X86_V2)"
      }
    },
    "add": {
      "ddd": {
        "current": "X86_V3",
        "available": "X86_V3 baseline(X86_V2)"
      },
      "FFF": {
        "current": "X86_V3",
        "available": "X86_V3 baseline(X86_V2)"
      }
    }
  }

