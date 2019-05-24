.. _bit_generator:

Bit Generators
--------------

.. currentmodule:: numpy.random

The random values produced by :class:`~Generator`
orignate in a BitGenerator.  The BitGenerators do not directly provide
random numbers and only contains methods used for seeding, getting or
setting the state, jumping or advancing the state, and for accessing
low-level wrappers for consumption by code that can efficiently
access the functions provided, e.g., `numba <https://numba.pydata.org>`_.

Stable RNGs
===========

.. toctree::
   :maxdepth: 1

   DSFMT <dsfmt>
   MT19937 <mt19937>
   PCG32 <pcg32>
   PCG64 <pcg64>
   Philox <philox>
   ThreeFry <threefry>
   Xoshiro256** <xoshiro256>
   Xoshiro512** <xoshiro512>

