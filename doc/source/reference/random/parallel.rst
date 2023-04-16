Parallel Random Number Generation
=================================

There are four main strategies implemented that can be used to produce
repeatable pseudo-random numbers across multiple processes (local
or distributed).

.. currentmodule:: numpy.random

.. _seedsequence-spawn:

`~SeedSequence` spawning
------------------------

NumPy allows you to spawn new (with very high probability) independent
`~BitGenerator` and `~Generator` instances via their ``spawn()`` method.
This spawning is implemented by the `~SeedSequence` used for initializing
the bit generators random stream.

`~SeedSequence` `implements an algorithm`_ to process a user-provided seed,
typically as an integer of some size, and to convert it into an initial state for
a `~BitGenerator`. It uses hashing techniques to ensure that low-quality seeds
are turned into high quality initial states (at least, with very high
probability).

For example, `MT19937` has a state consisting of 624
`uint32` integers. A naive way to take a 32-bit integer seed would be to just set
the last element of the state to the 32-bit seed and leave the rest 0s. This is
a valid state for `MT19937`, but not a good one. The Mersenne Twister
algorithm `suffers if there are too many 0s`_. Similarly, two adjacent 32-bit
integer seeds (i.e. ``12345`` and ``12346``) would produce very similar
streams.

`~SeedSequence` avoids these problems by using successions of integer hashes
with good `avalanche properties`_ to ensure that flipping any bit in the input
has about a 50% chance of flipping any bit in the output. Two input seeds that
are very close to each other will produce initial states that are very far
from each other (with very high probability). It is also constructed in such
a way that you can provide arbitrary-sized integers or lists of integers.
`~SeedSequence` will take all of the bits that you provide and mix them
together to produce however many bits the consuming `~BitGenerator` needs to
initialize itself.

These properties together mean that we can safely mix together the usual
user-provided seed with simple incrementing counters to get `~BitGenerator`
states that are (to very high probability) independent of each other. We can
wrap this together into an API that is easy to use and difficult to misuse.

.. code-block:: python

  from numpy.random import SeedSequence, default_rng

  ss = SeedSequence(12345)

  # Spawn off 10 child SeedSequences to pass to child processes.
  child_seeds = ss.spawn(10)
  streams = [default_rng(s) for s in child_seeds]

.. end_block

For convenience the direct use of `~SeedSequence` is not necessary.
The above ``streams`` can be spawned directly from a parent generator
via `~Generator.spawn`:

.. code-block:: python

  parent_rng = default_rng(12345)
  streams = parent_rng.spawn(10)

.. end_block

Child objects can also spawn to make grandchildren, and so on.
Each child has a `~SeedSequence` with its position in the tree of spawned
child objects mixed in with the user-provided seed to generate independent
(with very high probability) streams.

.. code-block:: python

  grandchildren = streams[0].spawn(4)

.. end_block

This feature lets you make local decisions about when and how to split up
streams without coordination between processes. You do not have to preallocate
space to avoid overlapping or request streams from a common global service. This
general "tree-hashing" scheme is `not unique to numpy`_ but not yet widespread.
Python has increasingly-flexible mechanisms for parallelization available, and
this scheme fits in very well with that kind of use.

Using this scheme, an upper bound on the probability of a collision can be
estimated if one knows the number of streams that you derive. `~SeedSequence`
hashes its inputs, both the seed and the spawn-tree-path, down to a 128-bit
pool by default. The probability that there is a collision in
that pool, pessimistically-estimated ([1]_), will be about :math:`n^2*2^{-128}` where
`n` is the number of streams spawned. If a program uses an aggressive million
streams, about :math:`2^{20}`, then the probability that at least one pair of
them are identical is about :math:`2^{-88}`, which is in solidly-ignorable
territory ([2]_).

.. [1] The algorithm is carefully designed to eliminate a number of possible
       ways to collide. For example, if one only does one level of spawning, it
       is guaranteed that all states will be unique. But it's easier to
       estimate the naive upper bound on a napkin and take comfort knowing
       that the probability is actually lower.

.. [2] In this calculation, we can mostly ignore the amount of numbers drawn from each
       stream. See :ref:`upgrading-pcg64` for the technical details about
       `PCG64`. The other PRNGs we provide have some extra protection built in
       that avoids overlaps if the `~SeedSequence` pools differ in the
       slightest bit. `PCG64DXSM` has :math:`2^{127}` separate cycles
       determined by the seed in addition to the position in the
       :math:`2^{128}` long period for each cycle, so one has to both get on or
       near the same cycle *and* seed a nearby position in the cycle.
       `Philox` has completely independent cycles determined by the seed.
       `SFC64` incorporates a 64-bit counter so every unique seed is at
       least :math:`2^{64}` iterations away from any other seed. And
       finally, `MT19937` has just an unimaginably huge period. Getting
       a collision internal to `SeedSequence` is the way a failure would be
       observed.

.. _`implements an algorithm`: http://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html
.. _`suffers if there are too many 0s`: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
.. _`avalanche properties`: https://en.wikipedia.org/wiki/Avalanche_effect
.. _`not unique to numpy`: https://www.iro.umontreal.ca/~lecuyer/myftp/papers/parallel-rng-imacs.pdf


.. _sequence-of-seeds:

Sequence of Integer Seeds
-------------------------

As discussed in the previous section, `~SeedSequence` can not only take an
integer seed, it can also take an arbitrary-length sequence of (non-negative)
integers. If one exercises a little care, one can use this feature to design
*ad hoc* schemes for getting safe parallel PRNG streams with similar safety
guarantees as spawning.

For example, one common use case is that a worker process is passed one
root seed integer for the whole calculation and also an integer worker ID (or
something more granular like a job ID, batch ID, or something similar). If
these IDs are created deterministically and uniquely, then one can derive
reproducible parallel PRNG streams by combining the ID and the root seed
integer in a list.

.. code-block:: python

  # default_rng() and each of the BitGenerators use SeedSequence underneath, so
  # they all accept sequences of integers as seeds the same way.
  from numpy.random import default_rng

  def worker(root_seed, worker_id):
      rng = default_rng([worker_id, root_seed])
      # Do work ...

  root_seed = 0x8c3c010cb4754c905776bdac5ee7501
  results = [worker(root_seed, worker_id) for worker_id in range(10)]

.. end_block

This can be used to replace a number of unsafe strategies that have been used
in the past which try to combine the root seed and the ID back into a single
integer seed value. For example, it is common to see users add the worker ID to
the root seed, especially with the legacy `~RandomState` code.

.. code-block:: python

  # UNSAFE! Do not do this!
  worker_seed = root_seed + worker_id
  rng = np.random.RandomState(worker_seed)

.. end_block

It is true that for any one run of a parallel program constructed this way,
each worker will have distinct streams. However, it is quite likely that
multiple invocations of the program with different seeds will get overlapping
sets of worker seeds. It is not uncommon (in the author's self-experience) to
change the root seed merely by an increment or two when doing these repeat
runs. If the worker seeds are also derived by small increments of the worker
ID, then subsets of the workers will return identical results, causing a bias
in the overall ensemble of results.

Combining the worker ID and the root seed as a list of integers eliminates this
risk. Lazy seeding practices will still be fairly safe.

This scheme does require that the extra IDs be unique and deterministically
created. This may require coordination between the worker processes. It is
recommended to place the varying IDs *before* the unvarying root seed.
`~SeedSequence.spawn` *appends* integers after the user-provided seed, so if
you might be mixing both this *ad hoc* mechanism and spawning, or passing your
objects down to library code that might be spawning, then it is a little bit
safer to prepend your worker IDs rather than append them to avoid a collision.

.. code-block:: python

  # Good.
  worker_seed = [worker_id, root_seed]

  # Less good. It will *work*, but it's less flexible.
  worker_seed = [root_seed, worker_id]

.. end_block

With those caveats in mind, the safety guarantees against collision are about
the same as with spawning, discussed in the previous section. The algorithmic
mechanisms are the same.


.. _independent-streams:

Independent Streams
-------------------

`Philox` is a counter-based RNG based which generates values by
encrypting an incrementing counter using weak cryptographic primitives. The
seed determines the key that is used for the encryption. Unique keys create
unique, independent streams. `Philox` lets you bypass the
seeding algorithm to directly set the 128-bit key. Similar, but different, keys
will still create independent streams.

.. code-block:: python

  import secrets
  from numpy.random import Philox

  # 128-bit number as a seed
  root_seed = secrets.getrandbits(128)
  streams = [Philox(key=root_seed + stream_id) for stream_id in range(10)]

.. end_block

This scheme does require that you avoid reusing stream IDs. This may require
coordination between the parallel processes.


.. _parallel-jumped:

Jumping the BitGenerator state
------------------------------

``jumped`` advances the state of the BitGenerator *as-if* a large number of
random numbers have been drawn, and returns a new instance with this state.
The specific number of draws varies by BitGenerator, and ranges from
:math:`2^{64}` to :math:`2^{128}`.  Additionally, the *as-if* draws also depend
on the size of the default random number produced by the specific BitGenerator.
The BitGenerators that support ``jumped``, along with the period of the
BitGenerator, the size of the jump and the bits in the default unsigned random
are listed below.

+-----------------+-------------------------+-------------------------+-------------------------+
| BitGenerator    | Period                  |  Jump Size              | Bits per Draw           |
+=================+=========================+=========================+=========================+
| MT19937         | :math:`2^{19937}-1`     | :math:`2^{128}`         | 32                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| PCG64           | :math:`2^{128}`         | :math:`~2^{127}` ([3]_) | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| PCG64DXSM       | :math:`2^{128}`         | :math:`~2^{127}` ([3]_) | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| Philox          | :math:`2^{256}`         | :math:`2^{128}`         | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+

.. [3] The jump size is :math:`(\phi-1)*2^{128}` where :math:`\phi` is the
       golden ratio. As the jumps wrap around the period, the actual distances
       between neighboring streams will slowly grow smaller than the jump size,
       but using the golden ratio this way is a classic method of constructing
       a low-discrepancy sequence that spreads out the states around the period
       optimally. You will not be able to jump enough to make those distances
       small enough to overlap in your lifetime.

``jumped`` can be used to produce long blocks which should be long enough to not
overlap.

.. code-block:: python

  import secrets
  from numpy.random import PCG64

  seed = secrets.getrandbits(128)
  blocked_rng = []
  rng = PCG64(seed)
  for i in range(10):
      blocked_rng.append(rng.jumped(i))

.. end_block

When using ``jumped``, one does have to take care not to jump to a stream that
was already used. In the above example, one could not later use
``blocked_rng[0].jumped()`` as it would overlap with ``blocked_rng[1]``. Like
with the independent streams, if the main process here wants to split off 10
more streams by jumping, then it needs to start with ``range(10, 20)``,
otherwise it would recreate the same streams. On the other hand, if you
carefully construct the streams, then you are guaranteed to have streams that
do not overlap.
