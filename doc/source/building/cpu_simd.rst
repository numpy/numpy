CPU support & SIMD
==================

NumPy supports a wide range of platforms and CPUs, and includes a significant
amount of code optimized for specific CPUs. By default, NumPy targets a
baseline with the minimum required SIMD instruction sets that are needed
(e.g., SSE4.2 on x86-64 CPUs) and uses dynamic dispatch to use newer instruction
sets (e.g., AVX2 and AVX512 on x86-64) when those are detected at runtime.

There are a number of build options that can be used to modify that behavior.
The default build settings are chosen for both portability and performance, and
should be reasonably close to optimal for creating redistributable binaries as
well as local installs. That said, there are reasons one may want to change the
default behavior, for example to obtain smaller binaries, to install on very old
hardware, to work around bugs, or for testing.

To detect and use all CPU features available on your local machine::

    $ python -m pip install . -Csetup-args=-Dcpu-baseline="native" -Csetup-args=-Dcpu-dispatch="none"

To use a lower baseline without any SIMD optimizations, useful for very old CPUs::

    $ python -m pip install . -Csetup-args=-Dcpu-baseline="none"

For more usage scenarios and more in-depth information about NumPy's SIMD support,
see :ref:`cpu-build-options`.
