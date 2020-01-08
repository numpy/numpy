.. _numpyrandom:

.. py:module:: numpy.random

.. currentmodule:: numpy.random

Random sampling (:mod:`numpy.random`)
=====================================

Numpy's random number routines produce pseudo random numbers using
combinations of a `BitGenerator` to create sequences and a `Generator`
to use those sequences to sample from different statistical distributions:

* BitGenerators: Objects that generate random numbers. These are typically
  unsigned integer words filled with sequences of either 32 or 64 random bits.
* Generators: Objects that transform sequences of random bits from a
  BitGenerator into sequences of numbers that follow a specific probability
  distribution (such as uniform, Normal or Binomial) within a specified
  interval.

Since Numpy version 1.17.0 the Generator can be initialized with a
number of different BitGenerators. It exposes many different probability
distributions. See `NEP 19 <https://www.numpy.org/neps/
nep-0019-rng-policy.html>`_ for context on the updated random Numpy number
routines. The legacy `RandomState` random number routines are still
available, but limited to a single BitGenerator.

For convenience and backward compatibility, a single `RandomState`
instance's methods are imported into the numpy.random namespace, see
:ref:`legacy` for the complete list.

.. _random-quick-start:

Quick Start
-----------

Call `default_rng` to get a new instance of a `Generator`, then call its
methods to obtain samples from different distributions.  By default,
`Generator` uses bits provided by `PCG64` which has better statistical
properties than the legacy `MT19937` used in `RandomState`.

.. code-block:: python

  # Do this
  from numpy.random import default_rng
  rng = default_rng()
  vals = rng.standard_normal(10)
  more_vals = rng.standard_normal(10)

  # instead of this
  from numpy import random
  vals = random.standard_normal(10)
  more_vals = random.standard_normal(10)

`Generator` can be used as a replacement for `RandomState`. Both class
instances hold a internal `BitGenerator` instance to provide the bit
stream, it is accessible as ``gen.bit_generator``. Some long-overdue API
cleanup means that legacy and compatibility methods have been removed from
`Generator`

=================== ============== ============
`RandomState`       `Generator`    Notes
------------------- -------------- ------------
``random_sample``,  ``random``     Compatible with `random.random`
``rand``
------------------- -------------- ------------
``randint``,        ``integers``   Add an ``endpoint`` kwarg
``random_integers``
------------------- -------------- ------------
``tomaxint``        removed        Use ``integers(0, np.iinfo(np.int_).max,``
                                   ``endpoint=False)``
------------------- -------------- ------------
``seed``            removed        Use `SeedSequence.spawn`
=================== ============== ============

See :ref:`new-or-different` for more information

Something like the following code can be used to support both ``RandomState``
and ``Generator``, with the understanding that the interfaces are slightly
different

.. code-block:: python

    try:
        rg_integers = rg.integers
    except AttributeError:
        rg_integers = rg.randint
    a = rg_integers(1000)

Seeds can be passed to any of the BitGenerators. The provided value is mixed
via `SeedSequence` to spread a possible sequence of seeds across a wider
range of initialization states for the BitGenerator. Here `PCG64` is used and
is wrapped with a `Generator`.

.. code-block:: python

  from numpy.random import Generator, PCG64
  rg = Generator(PCG64(12345))
  rg.standard_normal()

Introduction
------------
The new infrastructure takes a different approach to producing random numbers
from the `RandomState` object.  Random number generation is separated into
two components, a bit generator and a random generator.

The `BitGenerator` has a limited set of responsibilities. It manages state
and provides functions to produce random doubles and random unsigned 32- and
64-bit values.

The `random generator <Generator>` takes the
bit generator-provided stream and transforms them into more useful
distributions, e.g., simulated normal random values. This structure allows
alternative bit generators to be used with little code duplication.

The `Generator` is the user-facing object that is nearly identical to
`RandomState`. The canonical method to initialize a generator passes a
`PCG64` bit generator as the sole argument.

.. code-block:: python

  from numpy.random import default_rng
  rg = default_rng(12345)
  rg.random()

One can also instantiate `Generator` directly with a `BitGenerator` instance.
To use the older `MT19937` algorithm, one can instantiate it directly
and pass it to `Generator`.

.. code-block:: python

  from numpy.random import Generator, MT19937
  rg = Generator(MT19937(12345))
  rg.random()

What's New or Different
~~~~~~~~~~~~~~~~~~~~~~~
.. warning::

  The Box-Muller method used to produce NumPy's normals is no longer available
  in `Generator`.  It is not possible to reproduce the exact random
  values using Generator for the normal distribution or any other
  distribution that relies on the normal such as the `RandomState.gamma` or
  `RandomState.standard_t`. If you require bitwise backward compatible
  streams, use `RandomState`.

* The Generator's normal, exponential and gamma functions use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's Box-Muller or inverse CDF
  implementations.
* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double prevision uniform random variables for
  select distributions
* Optional ``out`` argument that allows existing arrays to be filled for
  select distributions
* All BitGenerators can produce doubles, uint64s and uint32s via CTypes
  (`PCG64.ctypes`) and CFFI (`PCG64.cffi`). This allows the bit generators
  to be used in numba.
* The bit generators can be used in downstream projects via
  :ref:`Cython <random_cython>`.
* `Generator.integers` is now the canonical way to generate integer
  random numbers from a discrete uniform distribution. The ``rand`` and
  ``randn`` methods are only available through the legacy `RandomState`.
  The ``endpoint`` keyword can be used to specify open or closed intervals.
  This replaces both ``randint`` and the deprecated ``random_integers``.
* `Generator.random` is now the canonical way to generate floating-point
  random numbers, which replaces `RandomState.random_sample`,
  `RandomState.sample`, and `RandomState.ranf`. This is consistent with
  Python's `random.random`.
* All BitGenerators in numpy use `SeedSequence` to convert seeds into
  initialized states.

See :ref:`new-or-different` for a complete list of improvements and
differences from the traditional ``Randomstate``.

Parallel Generation
~~~~~~~~~~~~~~~~~~~

The included generators can be used in parallel, distributed applications in
one of three ways:

* :ref:`seedsequence-spawn`
* :ref:`independent-streams`
* :ref:`parallel-jumped`

Concepts
--------
.. toctree::
   :maxdepth: 1

   generator
   Legacy Generator (RandomState) <legacy>
   BitGenerators, SeedSequences <bit_generators/index>

Features
--------
.. toctree::
   :maxdepth: 2

   Parallel Applications <parallel>
   Multithreaded Generation <multithreading>
   new-or-different
   Comparing Performance <performance>
   c-api
   Examples of using Numba, Cython, CFFI <extending>

Original Source of the Generator and BitGenerators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This package was developed independently of NumPy and was integrated in version
1.17.0. The original repo is at https://github.com/bashtage/randomgen.

