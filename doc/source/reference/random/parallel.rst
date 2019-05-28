Parallel Random Number Generation
=================================

There are three strategies implemented that can be used to produce
repeatable pseudo-random numbers across multiple processes (local
or distributed).

.. _independent-streams:

.. currentmodule:: numpy.random

Independent Streams
-------------------

:class:`~pcg64.PCG64`, :class:`~threefry.ThreeFry`
and :class:`~philox.Philox` support independent streams.  This
example shows how many streams can be created by passing in different index
values in the second input while using the same seed in the first.

.. code-block:: python

  from numpy.random.entropy import random_entropy
  from numpy.random import PCG64

  entropy = random_entropy(4)
  # 128-bit number as a seed
  seed = sum([int(entropy[i]) * 2 ** (32 * i) for i in range(4)])
  streams = [PCG64(seed, stream) for stream in range(10)]


:class:`~philox.Philox` and :class:`~threefry.ThreeFry` are
counter-based RNGs which use a counter and key.  Different keys can be used
to produce independent streams.

.. code-block:: python

  import numpy as np
  from numpy.random import ThreeFry

  key = random_entropy(8)
  key = key.view(np.uint64)
  key[0] = 0
  step = np.zeros(4, dtype=np.uint64)
  step[0] = 1
  streams = [ThreeFry(key=key + stream * step) for stream in range(10)]

.. _jump-and-advance:

Jump/Advance the BitGenerator state
-----------------------------------

Jumped
******

``jumped`` advances the state of the BitGenerator *as-if* a large number of
random numbers have been drawn, and returns a new instance with this state.
The specific number of draws varies by BitGenerator, and ranges from
:math:`2^{64}` to :math:`2^{512}`.  Additionally, the *as-if* draws also depend
on the size of the default random number produced by the specific BitGenerator.
The BitGenerator that support ``jumped``, along with the period of the
BitGenerator, the size of the jump and the bits in the default unsigned random
are listed below.

+-----------------+-------------------------+-------------------------+-------------------------+
| BitGenerator    | Period                  |  Jump Size              | Bits                    |
+=================+=========================+=========================+=========================+
| DSFMT           | :math:`2^{19937}`       | :math:`2^{128}`         | 53                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| MT19937         | :math:`2^{19937}`       | :math:`2^{128}`         | 32                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| PCG64           | :math:`2^{128}`         | :math:`2^{64}`          | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| Philox          | :math:`2^{256}`         | :math:`2^{128}`         | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| ThreeFry        | :math:`2^{256}`         | :math:`2^{128}`         | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| Xoshiro256**    | :math:`2^{256}`         | :math:`2^{128}`         | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| Xoshiro512**    | :math:`2^{512}`         | :math:`2^{256}`         | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+

``jumped`` can be used to produce long blocks which should be long enough to not
overlap.

.. code-block:: python

  from numpy.random.entropy import random_entropy
  from numpy.random import Xoshiro256

  entropy = random_entropy(2).astype(np.uint64)
  # 64-bit number as a seed
  seed = entropy[0] * 2**32 + entropy[1]
  blocked_rng = []
  rng = Xoshiro256(seed)
  for i in range(10):
      blocked_rng.append(rng.jumped(i))

Advance
*******
``advance`` can be used to jump the state an arbitrary number of steps, and so
is a more general approach than ``jumped``.  :class:`~pcg64.PCG64`,
:class:`~threefry.ThreeFry` and :class:`~philox.Philox`
support ``advance``, and since these also support
independent streams, it is not usually necessary to use ``advance``.

Advancing a BitGenerator updates the underlying state as-if a given number of
calls to the BitGenerator have been made. In general there is not a
one-to-one relationship between the number output random values from a
particular distribution and the number of draws from the core BitGenerator.
This occurs for two reasons:

* The random values are simulated using a rejection-based method
  and so more than one value from the underlying BitGenerator can be required
  to generate an single draw.
* The number of bits required to generate a simulated value differs from the
  number of bits generated by the underlying BitGenerator. For example, two
  16-bit integer values can be simulated from a single draw of a 32-bit value.

Advancing the BitGenerator state resets any pre-computed random numbers. This
is required to ensure exact reproducibility.

This example uses ``advance`` to advance a :class:`~pcg64.PCG64`
generator 2 ** 127 steps to set a sequence of random number generators.

.. code-block:: python

   from numpy.random import PCG64
   bit_generator = PCG64()
   bit_generator_copy = PCG64()
   bit_generator_copy.state = bit_generator.state

   advance = 2**127
   bit_generators = [bit_generator]
   for _ in range(9):
       bit_generator_copy.advance(advance)
       bit_generator = PCG64()
       bit_generator.state = bit_generator_copy.state
       bit_generators.append(bit_generator)

.. end block

