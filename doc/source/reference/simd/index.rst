.. _numpysimd:
.. currentmodule:: numpysimd

***********************
CPU/SIMD Optimizations
***********************

NumPy comes with a flexible working mechanism that allows it to harness the SIMD
features that CPUs own, in order to provide faster and more stable performance
on all popular platforms. Currently, NumPy supports the X86, IBM/Power, ARM7 and ARM8
architectures.

The optimization process in NumPy is carried out in three layers:

- Code is *written* using the universal intrinsics which is a set of types, macros and
  functions that are mapped to each supported instruction-sets by using guards that
  will enable use of the them only when the compiler recognizes them.
  This allow us to generate multiple kernels for the same functionality,
  in which each generated kernel represents a set of instructions that related one
  or multiple certain CPU features. The first kernel represents the minimum (baseline)
  CPU features, and the other kernels represent the additional (dispatched) CPU features.

- At *compile* time, CPU build options are used to define the minimum and
  additional features to support, based on user choice and compiler support. The
  appropriate intrinsics are overlaid with the platform / architecture intrinsics,
  and multiple kernels are compiled.

- At *runtime import*, the CPU is probed for the set of supported CPU
  features. A mechanism is used to grab the pointer to the most appropriate
  kernel, and this will be the one called for the function.

.. note::

   NumPy community had a deep discussion before implementing this work,
   please check `NEP-38`_ for more clarification.

.. toctree::

    build-options
    how-it-works

.. _`NEP-38`: https://numpy.org/neps/nep-0038-SIMD-optimizations.html

