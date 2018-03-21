.. _new-or-different:

What's New or Different
-----------------------

.. warning::

  The Box-Muller method used to produce NumPy's normals is no longer available.
  It is not possible to exactly reproduce the random values produced from NumPy
  for the normal distribution or any other distribution that relies on the
  normal such as the gamma or student's t.


* :func:`~randomgen.entropy.random_entropy` provides access to the system
  source of randomness that is used in cryptographic applications (e.g.,
  ``/dev/urandom`` on Unix).
* Simulate from the complex normal distribution
  (:meth:`~randomgen.generator.RandomGenerator.complex_normal`)
* The normal, exponential and gamma generators use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's default implementation in
  :meth:`~randomgen.generator.RandomGenerator.standard_normal`,
  :meth:`~randomgen.generator.RandomGenerator.standard_exponential` or
  :meth:`~randomgen.generator.RandomGenerator.standard_gamma`.
* The Box-Muller used to produce NumPy's normals is no longer available.
* All basic random generators functions to produce doubles, uint64s and
  uint32s via CTypes (:meth:`~randomgen.xoroshiro128.Xoroshiro128.ctypes`)
  and CFFI (:meth:`~randomgen.xoroshiro128.Xoroshiro128.cffi`).  This allows
  these basic RNGs to be used in numba.
* The basic random number generators can be used in downstream projects via
  Cython.


.. ipython:: python

  from randomgen import Xoroshiro128
  import numpy.random
  rg = Xoroshiro128().generator
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

  * Uniforms (:meth:`~randomgen.generator.RandomGenerator.random_sample` and
    :meth:`~randomgen.generator.RandomGenerator.rand`)
  * Normals (:meth:`~randomgen.generator.RandomGenerator.standard_normal` and
    :meth:`~randomgen.generator.RandomGenerator.randn`)
  * Standard Gammas (:meth:`~randomgen.generator.RandomGenerator.standard_gamma`)
  * Standard Exponentials (:meth:`~randomgen.generator.RandomGenerator.standard_exponential`)

.. ipython:: python

  rg.seed(0)
  rg.random_sample(3, dtype='d')
  rg.seed(0)
  rg.random_sample(3, dtype='f')

* Optional ``out`` argument that allows existing arrays to be filled for
  select distributions

  * Uniforms (:meth:`~randomgen.generator.RandomGenerator.random_sample`)
  * Normals (:meth:`~randomgen.generator.RandomGenerator.standard_normal`)
  * Standard Gammas (:meth:`~randomgen.generator.RandomGenerator.standard_gamma`)
  * Standard Exponentials (:meth:`~randomgen.generator.RandomGenerator.standard_exponential`)

  This allows multithreading to fill large arrays in chunks using suitable
  PRNGs in parallel.

.. ipython:: python

  existing = np.zeros(4)
  rg.random_sample(out=existing[:2])
  print(existing)

..   * For changes since the previous release, see the :ref:`change-log`
