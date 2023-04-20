.. _new-or-different:

.. currentmodule:: numpy.random

What's New or Different
-----------------------

NumPy 1.17.0 introduced `Generator` as an improved replacement for
the :ref:`legacy <legacy>` `RandomState`. Here is a quick comparison of the two
implementations.

================== ==================== =============
Feature            Older Equivalent     Notes
------------------ -------------------- -------------
`~.Generator`      `~.RandomState`      ``Generator`` requires a stream
                                        source, called a `BitGenerator`
                                        A number of these are provided.
                                        ``RandomState`` uses
                                        the Mersenne Twister `~.MT19937` by
                                        default, but can also be instantiated
                                        with any BitGenerator.
------------------ -------------------- -------------
``random``         ``random_sample``,   Access the values in a BitGenerator,
                   ``rand``             convert them to ``float64`` in the
                                        interval ``[0.0.,`` `` 1.0)``.
                                        In addition to the ``size`` kwarg, now
                                        supports ``dtype='d'`` or ``dtype='f'``,
                                        and an ``out`` kwarg to fill a user-
                                        supplied array.

                                        Many other distributions are also
                                        supported.
------------------ -------------------- -------------
``integers``       ``randint``,         Use the ``endpoint`` kwarg to adjust
                   ``random_integers``  the inclusion or exclusion of the
                                        ``high`` interval endpoint
================== ==================== =============

* The normal, exponential and gamma generators use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's default implementation in
  `~.Generator.standard_normal`, `~.Generator.standard_exponential` or
  `~.Generator.standard_gamma`. Because of the change in algorithms, it is not
  possible to reproduce the exact random values using ``Generator`` for these
  distributions or any distribution method that relies on them.

.. ipython:: python

  import numpy.random
  rng = np.random.default_rng()
  %timeit -n 1 rng.standard_normal(100000)
  %timeit -n 1 numpy.random.standard_normal(100000)

.. ipython:: python

  %timeit -n 1 rng.standard_exponential(100000)
  %timeit -n 1 numpy.random.standard_exponential(100000)

.. ipython:: python

  %timeit -n 1 rng.standard_gamma(3.0, 100000)
  %timeit -n 1 numpy.random.standard_gamma(3.0, 100000)


* `~.Generator.integers` is now the canonical way to generate integer
  random numbers from a discrete uniform distribution. This replaces both
  ``randint`` and the deprecated ``random_integers``.
* The ``rand`` and ``randn`` methods are only available through the legacy
  `~.RandomState`.
* `Generator.random` is now the canonical way to generate floating-point
  random numbers, which replaces `RandomState.random_sample`,
  `sample`, and `ranf`, all of which were aliases. This is consistent with
  Python's `random.random`.
* All bit generators can produce doubles, uint64s and
  uint32s via CTypes (`~PCG64.ctypes`) and CFFI (`~PCG64.cffi`).
  This allows these bit generators to be used in numba.
* The bit generators can be used in downstream projects via
  Cython.
* All bit generators use `SeedSequence` to :ref:`convert seed integers to
  initialized states <seeding_and_entropy>`.
* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double precision uniform random variables for
  select distributions. `~.Generator.integers` accepts a ``dtype`` argument
  with any signed or unsigned integer dtype.

  * Uniforms (`~.Generator.random` and `~.Generator.integers`)
  * Normals (`~.Generator.standard_normal`)
  * Standard Gammas (`~.Generator.standard_gamma`)
  * Standard Exponentials (`~.Generator.standard_exponential`)

.. ipython:: python

  rng = np.random.default_rng()
  rng.random(3, dtype=np.float64)
  rng.random(3, dtype=np.float32)
  rng.integers(0, 256, size=3, dtype=np.uint8)

* Optional ``out`` argument that allows existing arrays to be filled for
  select distributions

  * Uniforms (`~.Generator.random`)
  * Normals (`~.Generator.standard_normal`)
  * Standard Gammas (`~.Generator.standard_gamma`)
  * Standard Exponentials (`~.Generator.standard_exponential`)

  This allows multithreading to fill large arrays in chunks using suitable
  BitGenerators in parallel.

.. ipython:: python

  rng = np.random.default_rng()
  existing = np.zeros(4)
  rng.random(out=existing[:2])
  print(existing)

* Optional ``axis`` argument for methods like `~.Generator.choice`,
  `~.Generator.permutation` and `~.Generator.shuffle` that controls which
  axis an operation is performed over for multi-dimensional arrays.

.. ipython:: python

  rng = np.random.default_rng()
  a = np.arange(12).reshape((3, 4))
  a
  rng.choice(a, axis=1, size=5)
  rng.shuffle(a, axis=1)        # Shuffle in-place
  a

* Added a method to sample from the complex normal distribution
  (`~.Generator.complex_normal`)
