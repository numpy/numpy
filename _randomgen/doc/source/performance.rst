Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is
:class:`~randomgen.xoroshiro128.Xoroshiro128`.  The recommended generator
for use in large-scale parallel applications is
:class:`~randomgen.xorshift1024.Xorshift1024`
where the `jump` method is used to advance the state. For very large scale
applications -- requiring 1,000+ independent streams,
:class:`~randomgen.pcg64.PCG64` or :class:`~randomgen.threefry.ThreeFry` are
the best choices.

Timings
*******

The timings below are the time in ms to produce 1,000,000 random values from a
specific distribution.  :class:`~randomgen.xoroshiro128.Xoroshiro128` is the
fastest, followed by :class:`~randomgen.xorshift1024.Xorshift1024` and
:class:`~randomgen.pcg64.PCG64`.  The original :class:`~randomgen.mt19937.MT19937`
generator is much slower since it requires 2 32-bit values to equal the output
of the faster generators.

Integer performance has a similar ordering although `dSFMT` is slower since
it generates 53-bit floating point values rather than integer values. On the
other hand, it is very fast for uniforms, although slower than `xoroshiro128+`.

The pattern is similar for other, more complex generators. The normal
performance of NumPy's MT19937 is much lower than the other since it
uses the Box-Muller transformation rather than the Ziggurat generator. The
performance gap for Exponentials is also large due to the cost of computing
the log function to invert the CDF.

.. csv-table::
    :header: ,Xoroshiro128,Xorshift1024,PCG64,DSFMT,MT19937,Philox,ThreeFry,NumPy
    :widths: 14,14,14,14,14,14,14,14,14

    32-bit Unsigned Ints,3.0,3.0,3.0,3.5,3.7,6.8,6.6,3.3
    64-bit Unsigned Ints,2.6,3.0,3.1,3.4,3.8,6.9,6.6,8.8
    Uniforms,3.2,3.8,4.4,5.0,7.4,8.9,9.9,8.8
    Normals,11.0,13.9,13.7,15.8,16.9,17.8,18.8,63.0
    Exponentials,7.0,8.4,9.0,11.2,12.5,14.1,15.0,102.2
    Binomials,20.9,22.6,22.0,21.2,26.7,27.7,29.2,26.5
    Complex Normals,23.2,28.7,29.1,33.2,35.4,37.6,38.6,
    Gammas,35.3,38.6,39.2,41.3,46.7,49.4,51.2,98.8
    Laplaces,97.8,99.9,99.8,96.2,104.1,104.6,104.8,104.1
    Poissons,104.8,113.2,113.3,107.6,129.7,135.6,138.1,131.9


The next table presents the performance relative to `xoroshiro128+` in
percentage. The overall performance was computed using a geometric mean.

.. csv-table::
    :header: ,Xorshift1024,PCG64,DSFMT,MT19937,Philox,ThreeFry,NumPy
    :widths: 14,14,14,14,14,14,14,14
    
    32-bit Unsigned Ints,102,99,118,125,229,221,111
    64-bit Unsigned Ints,114,116,129,143,262,248,331
    Uniforms,116,137,156,231,275,306,274
    Normals,126,124,143,153,161,170,572
    Exponentials,121,130,161,179,203,215,1467
    Binomials,108,105,101,128,133,140,127
    Complex Normals,124,125,143,153,162,166,
    Gammas,109,111,117,132,140,145,280
    Laplaces,102,102,98,106,107,107,106
    Poissons,108,108,103,124,129,132,126
    Overall,113,115,125,144,172,177,251


.. note::

   All timings were taken using Linux on a i5-3570 processor.
