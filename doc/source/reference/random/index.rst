.. _numpyrandom:

.. py:module:: numpy.random

.. currentmodule:: numpy.random

Random sampling (:mod:`numpy.random`)
=====================================

.. _random-quick-start:

Quick Start
-----------

The :mod:`numpy.random` module implements pseudo-random number generators
(PRNGs) with the ability to draw samples from a variety of probability
distributions. In general, users will create a `Generator` instance with
`default_rng` and call the various methods on it to obtain samples from
different distributions.

::

  >>> import numpy as np
  >>> rng = np.random.default_rng()
  # Generate one random float uniformly distributed over the range [0, 1)
  >>> rng.random()  #doctest: +SKIP
  0.06369197489564249  # may vary
  # Generate an array of 10 numbers according to a unit Gaussian distribution.
  >>> rng.standard_normal(10)  #doctest: +SKIP
  array([-0.31018314, -1.8922078 , -0.3628523 , -0.63526532,  0.43181166,  # may vary
          0.51640373,  1.25693945,  0.07779185,  0.84090247, -2.13406828])
  # Generate an array of 5 integers uniformly over the range [0, 10).
  >>> rng.integers(low=0, high=10, size=5)  #doctest: +SKIP
  array([8, 7, 6, 2, 0])  # may vary

PRNGs are deterministic sequences and can be reproduced by specifying a seed to
control its initial state. By default, with no seed, `default_rng` will create
the PRNG using nondeterministic data from the operating system and therefore
generate different numbers each time. The pseudorandom sequences will be
practically independent.

::

    >>> rng1 = np.random.default_rng()
    >>> rng1.random()  #doctest: +SKIP
    0.6596288841243357  # may vary
    >>> rng2 = np.random.default_rng()
    >>> rng2.random()  #doctest: +SKIP
    0.11885628817151628  # may vary

Seeds are usually large positive integers. `default_rng` can take positive
integers of any size. We recommend using very large, unique numbers to ensure
that your seed is different from anyone else's. This is good practice to ensure
that your results are statistically independent from theirs unless if you are
intentionally *trying* to reproduce their result. A convenient way to get
such seed number is to use :py:func:`secrets.randbits` to get an
arbitrary 128-bit integer.

::

    >>> import secrets
    >>> import numpy as np
    >>> secrets.randbits(128)  #doctest: +SKIP
    122807528840384100672342137672332424406  # may vary
    >>> rng1 = np.random.default_rng(122807528840384100672342137672332424406)
    >>> rng1.random()
    0.5363922081269535
    >>> rng2 = np.random.default_rng(122807528840384100672342137672332424406)
    >>> rng2.random()
    0.5363922081269535

See the documentation on `default_rng` and `SeedSequence` for more advanced
options for controlling the seed in specialized scenarios.

`Generator` and its associated infrastructure was introduced in NumPy version
1.17.0. There is still a lot of code that uses the older `RandomState` and the
functions in `numpy.random`. While there are no plans to remove them at this
time, we do recommend transitioning to `Generator` as you can. The algorithms
are faster, more flexible, and will receive more improvements in the future.
For the most part, `Generator` can be used as a replacement for `RandomState`.
See :ref:`legacy` for information on the legacy infrastructure,
:ref:`new-or-different` for information on transitioning, and :ref:`NEP 19
<NEP19>` for some of the reasoning for the transition.

Design
------

Users primarily interact with `Generator` instances. Each `Generator` instance
owns a `BitGenerator` instance that implements the core PRNG algorithm. The
`BitGenerator` has a limited set of responsibilities. It manages state and
provides functions to produce random doubles and random unsigned 32- and 64-bit
values.

The `Generator` takes the bit generator-provided stream and transforms them
into more useful distributions, e.g., simulated normal random values. This
structure allows alternative bit generators to be used with little code
duplication.

Numpy implements several different `BitGenerator` classes implementing
different PRNG algorithms. `default_rng` currently uses `~PCG64` as the
default `BitGenerator`. It has better statistical properties and performance
over the `~MT19937` algorithm used in the legacy `RandomState`. See
:ref:`random-bit-generators` for more details on the supported BitGenerators.

`default_rng` and BitGenerators delegate the conversion of seeds into PRNG
states to `SeedSequence` internally. `SeedSequence` implements a sophisticated
algorithm that intermediates between the user's input and the internal
implementation details of each `BitGenerator` algorithm, each of which can
require different amounts of bits for its state. Importantly, it lets you use
arbitrary-sized integers and arbitrary sequences of such integers to mix
together into the PRNG state. This is a useful primitive for constructing
a `flexible pattern for parallel PRNG streams <seedsequence-spawn>`_.

For backward compatibility, we still maintain the legacy `RandomState` class.
It continues to use the `~MT19937` algorithm by default, and old seeds continue
to reproduce the same results. The convenience :ref:`functions-in-numpy-random`
are still aliases to the methods on a single global `RandomState` instance. See
:ref:`legacy` for the complete details.

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

What's New or Different
~~~~~~~~~~~~~~~~~~~~~~~
.. warning::

  The Box-Muller method used to produce NumPy's normals is no longer available
  in `Generator`.  It is not possible to reproduce the exact random
  values using Generator for the normal distribution or any other
  distribution that relies on the normal such as the `RandomState.gamma` or
  `RandomState.standard_t`. If you require bitwise backward compatible
  streams, use `RandomState`.

`Generator` can be used as a replacement for `RandomState`. Both class
instances hold an internal `BitGenerator` instance to provide the bit
stream, it is accessible as ``gen.bit_generator``. Some long-overdue API
cleanup means that legacy and compatibility methods have been removed from
`Generator`.

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

Something like the following code can be used to support both ``RandomState``
and ``Generator``, with the understanding that the interfaces are slightly
different.

.. code-block:: python

    try:
        rng_integers = rng.integers
    except AttributeError:
        rng_integers = rng.randint
    a = rng_integers(1000)

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

.. _random-compatibility:

Compatibility Policy
--------------------

`numpy.random` has a somewhat stricter compatibility policy than the rest of
Numpy. Users of pseudorandomness often have use cases for being able to
reproduce runs in fine detail given the same seed (so-called "stream
compatibility"), and so we try to balance those needs with the flexibility to
enhance our algorithms. :ref:`NEP 19 <NEP19>` describes the evolution of this
policy.

The main kind of compatibility that we enforce is stream-compatibility from run
to run under certain conditions. If you create a `Generator` with the same
`BitGenerator`, with the same seed, perform the same sequence of method calls
with the same arguments, on the same build of ``numpy``, in the same
environment, on the same machine, you should get the same stream of numbers.
Note that these conditions are very strict. There are a number of factors
outside of Numpy's control that limit our ability to guarantee much more than
this. For example, different CPUs implement floating point arithmetic
differently, and this can cause differences in certain edge cases that cascade
to the rest of the stream. `Generator.multivariate_normal`, for another
example, uses a matrix decomposition from ``numpy.linalg``. Even on the same
platform, a different build of ``numpy`` may use a different version of this
matrix decomposition algorithm from the LAPACK that it links to, causing
`Generator.multivariate_normal` to return completely different (but equally
valid!) results. We strive to prefer algorithms that are more resistant to
these effects, but this is always imperfect.

.. note::

   Most of the `Generator` methods allow you to draw multiple values from
   a distribution as arrays. The requested size of this array is a parameter,
   for the purposes of the above policy. Calling ``rng.random()`` 5 times is
   not *guaranteed* to give the same numbers as ``rng.random(5)``. We reserve
   the ability to decide to use different algorithms for different-sized
   blocks. In practice, this happens rarely.

Like the rest of Numpy, we generally maintain API source
compatibility from version to version. If we *must* make an API-breaking
change, then we will only do so with an appropriate deprecation period and
warnings, according to :ref:`general Numpy policy <NEP23>`.

Breaking stream-compatibility in order to introduce new features or
improve performance in `Generator` or `default_rng` will be *allowed* with
*caution*. Such changes will be considered features, and as such will be no
faster than the standard release cadence of features (i.e. on ``X.Y`` releases,
never ``X.Y.Z``).  Slowness will not be considered a bug for this purpose.
Correctness bug fixes that break stream-compatibility can happen on bugfix
releases, per usual, but developers should consider if they can wait until the
next feature release. We encourage developers to strongly weight user’s pain
from the break in stream-compatibility against the improvements. One example
of a worthwhile improvement would be to change algorithms for a significant
increase in performance, for example, moving from the `Box-Muller transform
<https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>`_ method of
Gaussian variate generation to the faster `Ziggurat algorithm
<https://en.wikipedia.org/wiki/Ziggurat_algorithm>`_. An example of
a discouraged improvement would be tweaking the Ziggurat tables just a little
bit for a small performance improvement.

.. note::

    In particular, `default_rng` is allowed to change the default
    `BitGenerator` that it uses (again, with *caution* and plenty of advance
    warning).

In general, `BitGenerator` classes have stronger guarantees of
version-to-version stream compatibility. This allows them to be a firmer
building block for downstream users that need it. Their limited API surface
makes them easier to maintain this compatibility from version to version. See
the docstrings of each `BitGenerator` class for their individual compatibility
guarantees.

The legacy `RandomState` and the `associated convenience functions
<random-functions-in-numpy-random>`_ have a stricter version-to-version
compatibility guarantee. For reasons outlined in :ref:`NEP 19 <NEP19>`, we had made
stronger promises about their version-to-version stability early in Numpy's
development. There are still some limited use cases for this kind of
compatibility (like generating data for tests), so we maintain as much
compatibility as we can. There will be no more modifications to `RandomState`,
not even to fix correctness bugs. There are a few gray areas where we can make
minor fixes to keep `RandomState` working without segfaulting as Numpy's
internals change, and some docstring fixes. However, the previously-mentioned
caveats about the variability from machine to machine and build to build still
apply to `RandomState` just as much as it does to `Generator`.

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
