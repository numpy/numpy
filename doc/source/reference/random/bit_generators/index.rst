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

Supported BitGenerators
=======================

The included BitGenerators are:

* MT19937 - The standard Python BitGenerator. Adds a `~mt19937.MT19937.jumped`
  function that returns a new generator with state as-if ``2**128`` draws have
  been made.
* PCG-64 - Fast generator that support many parallel streams and
  can be advanced by an arbitrary amount. See the documentation for
  :meth:`~.PCG64.advance`. PCG-64 has a period of
  :math:`2^{128}`. See the `PCG author's page`_ for more details about
  this class of PRNG.
* Philox - a counter-based generator capable of being advanced an
  arbitrary number of steps or generating independent streams. See the
  `Random123`_ page for more details about this class of bit generators.

.. _`PCG author's page`: http://www.pcg-random.org/
.. _`Random123`: https://www.deshawresearch.com/resources_random123.html


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
overlapping streams. Small seeds are unable to fill the complete range of
initializaiton states, and lead to biases among an ensemble of small-seed
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
