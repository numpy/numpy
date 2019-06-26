.. _bit_generator:

.. currentmodule:: numpy.random

Bit Generators
--------------

The random values produced by :class:`~Generator`
orignate in a BitGenerator.  The BitGenerators do not directly provide
random numbers and only contains methods used for seeding, getting or
setting the state, jumping or advancing the state, and for accessing
low-level wrappers for consumption by code that can efficiently
access the functions provided, e.g., `numba <https://numba.pydata.org>`_.

.. toctree::
   :maxdepth: 1

   BitGenerator <bitgenerators>
   MT19937 <mt19937>
   PCG64 <pcg64>
   Philox <philox>
   SFC64 <sfc64>

Seeding and Entropy
-------------------

A BitGenerator provides a stream of random values. In order to generate
reproducableis streams, BitGenerators support setting their initial state via a
seed. But how best to seed the BitGenerator? On first impulse one would like to
do something like ``[bg(i) for i in range(12)]`` to obtain 12 non-correlated,
independent BitGenerators. However using a highly correlated set of seeds could
generate BitGenerators that are correlated or overlap within a few samples.

NumPy uses a `SeedSequence` class to mix the seed in a reproducible way that
introduces the necessary entropy to produce independent and largely non-
overlapping streams. Small seeds may still be unable to reach all possible
initialization states, which can cause biases among an ensemble of small-seed
runs. For many cases, that doesn't matter. If you just want to hold things in
place while you debug something, biases aren't a concern.  For actual
simulations whose results you care about, let ``SeedSequence(None)`` do its
thing and then log/print the `SeedSequence.entropy` for repeatable
`BitGenerator` streams.

.. autosummary::
   :toctree: generated/

    bit_generator.ISeedSequence
    bit_generator.ISpawnableSeedSequence
    SeedSequence
    bit_generator.SeedlessSeedSequence
