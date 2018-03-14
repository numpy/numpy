Basic Random Number Generators
------------------------------

The random values produced by :class:`~randomgen.generator.RandomGenerator` 
are produced by a basic RNG.  These basic RNGs do not directly provide
random numbers and only contains methods used for seeding, getting or
setting the state, jumping or advancing the state, and for accessing 
low-level wrappers for consumption by code that can efficiently 
access the functions provided, e.g., `numba <https://numba.pydata.org>`_.

Stable RNGs
===========
These RNGs will be included in future releases.


.. toctree::
   :maxdepth: 1

   DSFMT <dsfmt>
   MT19937 <mt19937>
   PCG64 <pcg64>
   Philox <philox>
   ThreeFry <threefry>
   XoroShiro128+ <xoroshiro128>
   Xorshift1024*φ <xorshift1024>


Experimental RNGs
=================

These RNGs are currently included for testing but are may not be
permanent.

.. toctree::
   :maxdepth: 1

   PCG32 <pcg32>
   ThreeFry32 <threefry32>
