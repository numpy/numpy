:orphan:

.. _bitgenerators:

BitGenerators
-------------

.. currentmodule:: numpy.random

The random values produced by :class:`~Generator`
orignate in a BitGenerator.  The BitGenerators do not directly provide
random numbers and only contains methods used for seeding, getting or
setting the state, jumping or advancing the state, and for accessing
low-level wrappers for consumption by code that can efficiently
access the functions provided, e.g., `numba <https://numba.pydata.org>`_.


The included BitGenerators are:

* `MT19937 <mt19937>` - The standard Python BitGenerator. Produces identical results to
  Python using the same seed/state. Adds a `~mt19937.MT19937.jumped` function
  that returns a new generator with state as-if ``2**128`` draws have been made.
* `DSFMT <dsfmt>` - SSE2 enabled versions of the MT19937 generator. Widely used
  across many software packages as the default generator. Probably behind as
  many papers as any other generator. Good performance on any CPU with SSE2 or
  Altivec. See the `dSFMT authors' page`_.
* `Xoshiro256 <xoshiro256>` and `Xoshiro512 <xoshiro512>` - The most recently
  introduced XOR, shift, and rotate generator. Fast and popular bit generator,
  despite some reservations in rare corner case.More information about these bit
  generators is available at the `xorshift, xoroshiro and xoshiro authors'
  page`_.
* `ThreeFry <threefry>` and `Philox <philox>` - counter-based generators
  capable of being advanced an arbitrary number of steps or generating
  independent streams. Very popular in machine learning. See the `Random123`_
  page for more details about this class of bit generators.
* `PCG32 <pcg32>` and `PCG64 <pcg64>` are permutation-congruential generators
  with very good statistical properties.  More information is available on the
  `PCG authors' page`_.

.. _`dSFMT authors' page`: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/
.. _`PCG authors' page`: http://www.pcg-random.org/
.. _`xorshift, xoroshiro and xoshiro authors' page`:  http://xoroshiro.di.unimi.it/
.. _`Random123`: https://www.deshawresearch.com/resources_random123.html

Summary:

============== ==================== ================ ======= =======
BitGenerator   period               jumped distance  speed relative
                                                     to MT19937 [1]_
-------------- -------------------- ---------------- ---------------
\                                                    linux64 win32
============== ==================== ================ ======= =======
``MT19937``                         :math:`2^{128}`  1x      1x
``dSFMT``                           :math:`2^{128}`  1.2x    1.3x
``Xoshiro256`` :math:`2^{256} - 1`  :math:`2^{128}`  1.6x    0.8x
``Xoshiro512`` :math:`2^{1024} - 1` :math:`2^{512}`
``ThreeFry``   :math:`2^{256} - 1`  :math:`2^{128}`  0.6x    0.3x
``PHilox``     :math:`2^{256} - 1`  :math:`2^{128}`  0.9x    0.3x
``PCG32``      :math:`2^{64}`       :math:`2^{32}`
``PCG64``      :math:`2^{128}`      :math:`2^{64}`   1.4x    0.3x
============== ==================== ================ ======= =======

.. rubric:: Footnotes

.. [1] Each platform measured separately. More is better. As always with
       benchmarks, these are rough guides, your experience may vary. For more
       details see :ref:`numpy-random-performance`


Recommendations
===============
The recommended generator for single use is :class:`~.xoshiro256.Xoshiro256`.
The recommended generator for use in large-scale parallel applications is
:class:`~.xoshiro512.Xoshiro512` where the `~.xoshiro512.Xoshiro512.jumped`
method is used to advance the state. For very large scale applications --
requiring 1,000+ independent streams, :class:`~pcg64.PCG64` or
:class:`~.philox.Philox` are the best choices.


Details
=======

.. toctree::
   :maxdepth: 1

   DSFMT <dsfmt>
   MT19937 <mt19937>
   PCG32 <pcg32>
   PCG64 <pcg64>
   Philox <philox>
   ThreeFry <threefry>
   Xoshiro256 <xoshiro256>
   Xoshiro512 <xoshiro512>

