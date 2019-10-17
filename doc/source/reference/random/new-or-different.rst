.. _new-or-different:

.. currentmodule:: numpy.random

What's New or Different
-----------------------

.. warning::

  The Box-Muller method used to produce NumPy's normals is no longer available
  in `Generator`.  It is not possible to reproduce the exact random
  values using ``Generator`` for the normal distribution or any other
  distribution that relies on the normal such as the `Generator.gamma` or
  `Generator.standard_t`. If you require bitwise backward compatible
  streams, use `RandomState`, i.e., `RandomState.gamma` or
  `RandomState.standard_t`.

Quick comparison of legacy `mtrand <legacy>`_ to the new `Generator`

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
                   ``random_integers``  the inclusion or exclution of the
                                        ``high`` interval endpoint
================== ==================== =============

And in more detail:

* Simulate from the complex normal distribution
  (`~.Generator.complex_normal`)
* The normal, exponential and gamma generators use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's default implementation in
  `~.Generator.standard_normal`, `~.Generator.standard_exponential` or
  `~.Generator.standard_gamma`.
* `~.Generator.integers` is now the canonical way to generate integer
  random numbers from a discrete uniform distribution. The ``rand`` and
  ``randn`` methods are only available through the legacy `~.RandomState`.
  This replaces both ``randint`` and the deprecated ``random_integers``.
* The Box-Muller method used to produce NumPy's normals is no longer available.
* All bit generators can produce doubles, uint64s and
  uint32s via CTypes (`~PCG64.ctypes`) and CFFI (`~PCG64.cffi`).
  This allows these bit generators to be used in numba.
* The bit generators can be used in downstream projects via
  Cython.


.. ipython:: python

  from  numpy.random import Generator, PCG64
  import numpy.random
  rg = Generator(PCG64())
  %timeit rg.standard_normal(100000)
  %timeit numpy.random.standard_normal(100000)

.. ipython:: python

  %timeit rg.standard_exponential(100000)
  %timeit numpy.random.standard_exponential(100000)

.. ipython:: python

  %timeit rg.standard_gamma(3.0, 100000)
  %timeit numpy.random.standard_gamma(3.0, 100000)

* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double prevision uniform random variables for
  select distributions

  * Uniforms (`~.Generator.random` and `~.Generator.integers`)
  * Normals (`~.Generator.standard_normal`)
  * Standard Gammas (`~.Generator.standard_gamma`)
  * Standard Exponentials (`~.Generator.standard_exponential`)

.. ipython:: python

  rg = Generator(PCG64(0))
  rg.random(3, dtype='d')
  rg.random(3, dtype='f')

* Optional ``out`` argument that allows existing arrays to be filled for
  select distributions

  * Uniforms (`~.Generator.random`)
  * Normals (`~.Generator.standard_normal`)
  * Standard Gammas (`~.Generator.standard_gamma`)
  * Standard Exponentials (`~.Generator.standard_exponential`)

  This allows multithreading to fill large arrays in chunks using suitable
  BitGenerators in parallel.

.. ipython:: python

  existing = np.zeros(4)
  rg.random(out=existing[:2])
  print(existing)

