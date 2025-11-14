.. _numpyrandom:

.. py:module:: numpy.random

.. currentmodule:: numpy.random

Random sampling
===============

.. _random-quick-start:

Quick start
-----------

The :mod:`numpy.random` module implements pseudo-random number generators
(PRNGs or RNGs, for short) with the ability to draw samples from a variety of
probability distributions. In general, users will create a `Generator` instance
with `default_rng` and call the various methods on it to obtain samples from
different distributions.

.. try_examples::

  >>> import numpy as np
  >>> rng = np.random.default_rng()

  Generate one random float uniformly distributed over the range :math:`[0, 1)`:

  >>> rng.random()  #doctest: +SKIP
  0.06369197489564249  # may vary

  Generate an array of 10 numbers according to a unit Gaussian distribution:

  >>> rng.standard_normal(10)  #doctest: +SKIP
  array([-0.31018314, -1.8922078 , -0.3628523 , -0.63526532,  0.43181166,  # may vary
          0.51640373,  1.25693945,  0.07779185,  0.84090247, -2.13406828])

  Generate an array of 5 integers uniformly over the range :math:`[0, 10)`:

  >>> rng.integers(low=0, high=10, size=5)  #doctest: +SKIP
  array([8, 7, 6, 2, 0])  # may vary

Our RNGs are deterministic sequences and can be reproduced by specifying a seed integer to
derive its initial state. By default, with no seed provided, `default_rng` will
seed the RNG from nondeterministic data from the operating system and therefore
generate different numbers each time. The pseudo-random sequences will be
independent for all practical purposes, at least those purposes for which our
pseudo-randomness was good for in the first place.

.. try_examples::

  >>> import numpy as np
  >>> rng1 = np.random.default_rng()
  >>> rng1.random()  #doctest: +SKIP
  0.6596288841243357  # may vary
  >>> rng2 = np.random.default_rng()
  >>> rng2.random()  #doctest: +SKIP
  0.11885628817151628  # may vary

.. warning::

  The pseudo-random number generators implemented in this module are designed
  for statistical modeling and simulation. They are not suitable for security
  or cryptographic purposes. See the :py:mod:`secrets` module from the
  standard library for such use cases.

.. _recommend-secrets-randbits:

Seeds should be large positive integers. `default_rng` can take positive
integers of any size. We recommend using very large, unique numbers to ensure
that your seed is different from anyone else's. This is good practice to ensure
that your results are statistically independent from theirs unless you are
intentionally *trying* to reproduce their result. A convenient way to get
such a seed number is to use :py:func:`secrets.randbits` to get an
arbitrary 128-bit integer.


.. try_examples::

  >>> import numpy as np
  >>> import secrets
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
owns a `BitGenerator` instance that implements the core RNG algorithm. The
`BitGenerator` has a limited set of responsibilities. It manages state and
provides functions to produce random doubles and random unsigned 32- and 64-bit
values.

The `Generator` takes the bit generator-provided stream and transforms them
into more useful distributions, e.g., simulated normal random values. This
structure allows alternative bit generators to be used with little code
duplication.

NumPy implements several different `BitGenerator` classes implementing
different RNG algorithms. `default_rng` currently uses `~PCG64` as the
default `BitGenerator`. It has better statistical properties and performance
than the `~MT19937` algorithm used in the legacy `RandomState`. See
:ref:`random-bit-generators` for more details on the supported BitGenerators.

`default_rng` and BitGenerators delegate the conversion of seeds into RNG
states to `SeedSequence` internally. `SeedSequence` implements a sophisticated
algorithm that intermediates between the user's input and the internal
implementation details of each `BitGenerator` algorithm, each of which can
require different amounts of bits for its state. Importantly, it lets you use
arbitrary-sized integers and arbitrary sequences of such integers to mix
together into the RNG state. This is a useful primitive for constructing
a :ref:`flexible pattern for parallel RNG streams <seedsequence-spawn>`.

For backward compatibility, we still maintain the legacy `RandomState` class.
It continues to use the `~MT19937` algorithm by default, and old seeds continue
to reproduce the same results. The convenience :ref:`functions-in-numpy-random`
are still aliases to the methods on a single global `RandomState` instance. See
:ref:`legacy` for the complete details. See :ref:`new-or-different` for
a detailed comparison between `Generator` and `RandomState`.

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
   compatibility

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
