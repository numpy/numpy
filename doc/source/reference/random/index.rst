.. _numpyrandom:

.. py:module:: numpy.random

.. currentmodule:: numpy.random

Random sampling (:mod:`numpy.random`)
=====================================

NumPy facilitates various tools for generating pseudo-random numbers. This
also includes providing means to create repeatable sequences for reproducible
pseudo-random processes. As NumPy random module has undergone substantial changes
first the quick reference for the major differences are laid out to demonstrate
what the usage style used to be pre NumPy v1.17 and what currently is for
the impatient reader.

.. _random-quick-start:

Quick Reference
---------------

If you want to draw samples from various distributions, below is probably what you
see in the old docs and elsewhere, roughly speaking, import ``random`` module and then
use functions from the :ref:`legacy` section, such as

.. code-block:: python

  # pre-NumPy 1.17, nostalgic style,
  import numpy as np
  my_array = np.random.rand(5, 3)
  my_bivar = np.random.multivariate_normal((1,2), [[1, 0], [0, 1]], (3, 3))
  my_uniform = np.random.uniform(-0.5, 0.5, 100)
  # and many other legacy functions
  # see above Legacy link for the complete list

Instead, you can create a random number `Generator` instance (not to be confused
with a Python generator of ``(n for n in range(5))`` variety) and use the methods
of this generator. For a regular user, a generator with the default settings should
be quite sufficient. And for that, NumPy has a shorthand to create it.

.. code-block:: python
  
  # the current, recommended style
  import numpy as np
  rng = np.random.default_rng()  # (r)andom (n)umber (g)enerator
  my_array = rng.random([5, 3])
  my_bivar = rng.multivariate_normal(mean=(1, 2), cov=[[1, 0], [0, 1]], size=(3, 3))
  my_uniform = rng.uniform(-0.5, 0.5, 100)
  # and so on
  
  # Create another rng with a seed which only initializes this particular
  # generator and the rest remains unaltered
  rng0 = np.default_rng(19400707)
  sgt_p = rng0.uniform(-60, 4, 50)

.. note:: **Seed and State Management**

   Historically, it was possible to influence the random state through
   initializing/resetting the random number generator via setting an integer,
   known as a "seed", such that the randomized procedures in the existing
   scope can be reproduced. While this worked in many scenarios, the seed
   setting caused some nontrivial challenges such as leaking to other processes
   that suppoesed to use their own random processes or affecting other packages
   that relied on NumPy and many other interesting bugs.

   This situation changed when NumPy started offering a rather finer control
   to initialize and contain these processes. As you can see from the last
   example, you don't need to limit the scope of the seeding or implement more
   hacks to contain the effects of the (deprecated) `seed`.
   
   Moreover, the current infrastructure takes a different approach to producing
   random numbers from the legacy `RandomState` object.

Introduction
------------

Numpy's random number routines produce pseudo random numbers using
combinations of a `BitGenerator` to create sequences and a `Generator`
to use those sequences to sample from different statistical distributions:

* BitGenerators: Objects that generate random numbers. These are typically
  unsigned integer words filled with sequences of either 32 or 64 random bits.
* Generators: Objects that transform sequences of random bits from a
  BitGenerator into sequences of numbers that follow a specific probability
  distribution (such as uniform, Normal or Binomial) within a specified
  interval.

In short, BitGenerators are the parts responsible for the randomization
and the state, whereas Generators are responsible for utilizing these
randomized inputs.

The Generators can be initialized with a number of different BitGenerators.
They expose many different probability distributions. For the context
and the design decisions, we refer to `NEP 19 <https://www.numpy.org/neps/
nep-0019-rng-policy.html>`_ . The legacy `RandomState` random number routines
are still available, but limited to a single BitGenerator. See :ref:`new-or-different` 
for a complete list of improvements and differences from the legacy
``RandomState``.

For convenience and backward compatibility, a single `RandomState`
instance's methods are imported into the numpy.random namespace, see
:ref:`legacy` for the complete list. At this point, we have established
the `Generator` based way-of-working still the legacy API is available.
While this might seem like, we offer two options, it is quite the
contrary. We sincerely hope that we minimized the annoyance during
these changes, however we also hope that users will eventually stop
using the legacy API which would allow us deprecate and remove the
legacy API. 


Supporting legacy API
---------------------

The most important part to use the Generator-based API is to use a better
BitGenerator by default; `Generator` uses bits provided by `PCG64` which
has better statistical properties than the legacy `MT19937` used in 
`RandomState`.

`Generator` can be used as a replacement for `RandomState`. Both class
instances hold an internal `BitGenerator` instance to provide the bit
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

See :ref:`new-or-different` for more information.

Something like the following code can be used to support both ``RandomState``
and ``Generator``, with the understanding that the interfaces are slightly
different

.. code-block:: python

    try:
        rng_integers = rng.integers
    except AttributeError:
        rng_integers = rng.randint
    a = rng_integers(1000)

Seeds can be passed to any of the BitGenerators. The provided value is mixed
via `SeedSequence` to spread a possible sequence of seeds across a wider
range of initialization states for the BitGenerator. Here `PCG64` is used and
is wrapped with a `Generator`.

.. code-block:: python

  from numpy.random import Generator, PCG64
  rng = Generator(PCG64(12345))
  rng.standard_normal()
  
Here we use `default_rng` to create an instance of `Generator` to generate a 
random float:
 
>>> import numpy as np
>>> rng = np.random.default_rng(12345)
>>> print(rng)
Generator(PCG64)
>>> rfloat = rng.random()
>>> rfloat
0.22733602246716966
>>> type(rfloat)
<class 'float'>
 
Here we use `default_rng` to create an instance of `Generator` to generate 3 
random integers between 0 (inclusive) and 10 (exclusive):
    
>>> import numpy as np
>>> rng = np.random.default_rng(12345)
>>> rints = rng.integers(low=0, high=10, size=3)
>>> rints
array([6, 2, 7])
>>> type(rints[0])
<class 'numpy.int64'> 


The `BitGenerator` has a limited set of responsibilities. It manages state
and provides functions to produce random doubles and random unsigned 32- and
64-bit values.

The `random generator <Generator>` takes the
bit generator-provided stream and transforms them into more useful
distributions, e.g., simulated normal random values. This structure allows
alternative bit generators to be used with little code duplication.

The `Generator` is the user-facing object that is nearly identical to the
legacy `RandomState`. It accepts a bit generator instance as an argument.
The default is currently `PCG64` but this may change in future versions. 
As a convenience NumPy  provides the `default_rng` function to hide these 
details:
  
>>> from numpy.random import default_rng
>>> rng = default_rng(12345)
>>> print(rng)
Generator(PCG64)
>>> print(rng.random())
0.22733602246716966
  
One can also instantiate `Generator` directly with a `BitGenerator` instance.

To use the default `PCG64` bit generator, one can instantiate it directly and 
pass it to `Generator`:

>>> from numpy.random import Generator, PCG64
>>> rng = Generator(PCG64(12345))
>>> print(rng)
Generator(PCG64)

Similarly to use the older `MT19937` bit generator (not recommended), one can
instantiate it directly and pass it to `Generator`:

>>> from numpy.random import Generator, MT19937
>>> rng = Generator(MT19937(12345))
>>> print(rng)
Generator(MT19937)

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
  to produce either single or double precision uniform random variables for
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
* The addition of an ``axis`` keyword argument to methods such as 
  `Generator.choice`, `Generator.permutation`,  and `Generator.shuffle` 
  improves support for sampling from and shuffling multi-dimensional arrays.

See :ref:`new-or-different` for a complete list of improvements and
differences from the traditional ``Randomstate``.

Parallel Generation
~~~~~~~~~~~~~~~~~~~

The included generators can be used in parallel, distributed applications in
a number of ways:

* :ref:`seedsequence-spawn`
* :ref:`sequence-of-seeds`
* :ref:`independent-streams`
* :ref:`parallel-jumped`

Users with a very large amount of parallelism will want to consult
:ref:`upgrading-pcg64`.

Concepts
--------
.. toctree::
   :maxdepth: 1

   generator
   Legacy Generator (RandomState) <legacy>
   BitGenerators, SeedSequences <bit_generators/index>
   Upgrading PCG64 with PCG64DXSM <upgrading-pcg64>

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
