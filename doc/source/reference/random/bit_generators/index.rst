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

* PCG-64 - The default. A fast generator that supports many parallel streams
  and can be advanced by an arbitrary amount. See the documentation for
  :meth:`~.PCG64.advance`. PCG-64 has a period of :math:`2^{128}`. See the `PCG
  author's page`_ for more details about this class of PRNG.
* MT19937 - The standard Python BitGenerator. Adds a `MT19937.jumped`
  function that returns a new generator with state as-if :math:`2^{128}` draws have
  been made.
* Philox - A counter-based generator capable of being advanced an
  arbitrary number of steps or generating independent streams. See the
  `Random123`_ page for more details about this class of bit generators.
* SFC64 - A fast generator based on random invertible mappings. Usually the
  fastest generator of the four. See the `SFC author's page`_ for (a little)
  more detail.

.. _`PCG author's page`: http://www.pcg-random.org/
.. _`Random123`: https://www.deshawresearch.com/resources_random123.html
.. _`SFC author's page`: http://pracrand.sourceforge.net/RNG_engines.txt

.. autosummary::
    :toctree: generated/

    BitGenerator

.. toctree::
    :maxdepth: 1

    MT19937 <mt19937>
    PCG64 <pcg64>
    Philox <philox>
    SFC64 <sfc64>

Seeding and Entropy
-------------------

A BitGenerator provides a stream of random values. In order to generate
reproducible streams, BitGenerators support setting their initial state via a
seed. All of the provided BitGenerators will take an arbitrary-sized
non-negative integer, or a list of such integers, as a seed. BitGenerators
need to take those inputs and process them into a high-quality internal state
for the BitGenerator. All of the BitGenerators in numpy delegate that task to
`SeedSequence`, which uses hashing techniques to ensure that even low-quality
seeds generate high-quality initial states.

.. code-block:: python

    from numpy.random import PCG64

    bg = PCG64(12345678903141592653589793)

.. end_block

`~SeedSequence` is designed to be convenient for implementing best practices.
We recommend that a stochastic program defaults to using entropy from the OS so
that each run is different. The program should print out or log that entropy.
In order to reproduce a past value, the program should allow the user to
provide that value through some mechanism, a command-line argument is common,
so that the user can then re-enter that entropy to reproduce the result.
`~SeedSequence` can take care of everything except for communicating with the
user, which is up to you.

.. code-block:: python

    from numpy.random import PCG64, SeedSequence

    # Get the user's seed somehow, maybe through `argparse`.
    # If the user did not provide a seed, it should return `None`.
    seed = get_user_seed()
    ss = SeedSequence(seed)
    print('seed = {}'.format(ss.entropy))
    bg = PCG64(ss)

.. end_block

We default to using a 128-bit integer using entropy gathered from the OS. This
is a good amount of entropy to initialize all of the generators that we have in
numpy. We do not recommend using small seeds below 32 bits for general use.
Using just a small set of seeds to instantiate larger state spaces means that
there are some initial states that are impossible to reach. This creates some
biases if everyone uses such values.

There will not be anything *wrong* with the results, per se; even a seed of
0 is perfectly fine thanks to the processing that `~SeedSequence` does. If you
just need *some* fixed value for unit tests or debugging, feel free to use
whatever seed you like. But if you want to make inferences from the results or
publish them, drawing from a larger set of seeds is good practice.

If you need to generate a good seed "offline", then ``SeedSequence().entropy``
or using ``secrets.randbits(128)`` from the standard library are both
convenient ways.

.. autosummary::
    :toctree: generated/

    SeedSequence
