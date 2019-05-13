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
   Philox <philox>
   ThreeFry <threefry>
   XoroShiro128+ <xoroshiro128>
   Xorshift1024*φ <xorshift1024>
   Xoshiro256** <xoshiro256>
   Xoshiro512** <xoshiro512>


Experimental RNGs
=================

These BitGenerators are currently included but are may not be permanent.

.. toctree::
   :maxdepth: 1

   ThreeFry32 <threefry32>
