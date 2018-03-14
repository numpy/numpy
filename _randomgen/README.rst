RandomGen
=========

|Travis Build Status| |Appveyor Build Status|

Random Number Generator using settable Basic RNG interface for future
NumPy RandomState evolution.

This is a library and generic interface for alternative random
generators in Python and NumPy.

Compatibility Warning
~~~~~~~~~~~~~~~~~~~~~

RandomGen no longer supports Box-Muller normal variates and so it not
100% compatible with NumPy (or randomstate). Box-Muller normals are slow
to generate and all functions which previously relied on Box-Muller
normals now use the faster Ziggurat implementation.

Features
--------

-  Replacement for NumPy's RandomState

.. code:: python

    # import numpy.random as rnd
    from randomgen import RandomGenerator, MT19937
    rnd = RandomGenerator(MT19937())
    x = rnd.standard_normal(100)
    y = rnd.random_sample(100)
    z = rnd.randn(10,10)

-  Default random generator is a fast generator called Xoroshiro128plus
-  Support for random number generators that support independent streams
   and jumping ahead so that sub-streams can be generated
-  Faster random number generation, especially for normal, standard
   exponential and standard gamma using the Ziggurat method

.. code:: python

    from randomgen import RandomGenerator
    # Use Xoroshiro128
    rnd = RandomGenerator()
    w = rnd.standard_normal(10000, method='zig')
    x = rnd.standard_exponential(10000, method='zig')
    y = rnd.standard_gamma(5.5, 10000, method='zig')

-  Support for 32-bit floating randoms for core generators. Currently
   supported:

   -  Uniforms (``random_sample``)
   -  Exponentials (``standard_exponential``, both Inverse CDF and
      Ziggurat)
   -  Normals (``standard_normal``)
   -  Standard Gammas (via ``standard_gamma``)

**WARNING**: The 32-bit generators are **experimental** and subject to
change.

**Note**: There are *no* plans to extend the alternative precision
generation to all distributions.

-  Support for filling existing arrays using ``out`` keyword argument.
   Currently supported in (both 32- and 64-bit outputs)

   -  Uniforms (``random_sample``)
   -  Exponentials (``standard_exponential``)
   -  Normals (``standard_normal``)
   -  Standard Gammas (via ``standard_gamma``)

Included Pseudo Random Number Generators
----------------------------------------

This module includes a number of alternative random number generators in
addition to the MT19937 that is included in NumPy. The RNGs include:

-  `MT19937 <https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/>`__,
   the NumPy rng
-  `dSFMT <http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/>`__ a
   SSE2-aware version of the MT19937 generator that is especially fast
   at generating doubles
-  `xoroshiro128+ <http://xoroshiro.di.unimi.it/>`__ and
   `xorshift1024\*φ <http://xorshift.di.unimi.it/>`__
-  `PCG64 <http:w//www.pcg-random.org/>`__
-  ThreeFry and Philox from
   `Random123 <https://www.deshawrsearch.com/resources_random123.html>`__
   ## Differences from ``numpy.random.RandomState``

New Features
~~~~~~~~~~~~

-  ``standard_normal``, ``normal``, ``randn`` and
   ``multivariate_normal`` all use the much faster (100%+) Ziggurat
   method.
-  ``standard_gamma`` and ``gamma`` both use the much faster Ziggurat
   method.
-  ``standard_exponential`` ``exponential`` both support an additional
   ``method`` keyword argument which can be ``inv`` or ``zig`` where
   ``inv`` corresponds to the current method using the inverse CDF and
   ``zig`` uses the much faster (100%+) Ziggurat method.
-  Core random number generators can produce either single precision
   (``np.float32``) or double precision (``np.float64``, the default)
   using the optional keyword argument ``dtype``
-  Core random number generators can fill existing arrays using the
   ``out`` keyword argument
-  Standardizes integer-values random values as int64 for all platforms.

New Functions
~~~~~~~~~~~~~

-  ``random_entropy`` - Read from the system entropy provider, which is
   commonly used in cryptographic applications
-  ``random_raw`` - Direct access to the values produced by the
   underlying PRNG. The range of the values returned depends on the
   specifics of the PRNG implementation.
-  ``random_uintegers`` - unsigned integers, either 32-
   (``[0, 2**32-1]``) or 64-bit (``[0, 2**64-1]``)
-  ``jump`` - Jumps RNGs that support it. ``jump`` moves the state a
   great distance. *Only available if supported by the RNG.*
-  ``advance`` - Advanced the RNG 'as-if' a number of draws were made,
   without actually drawing the numbers. *Only available if supported by
   the RNG.*

Status
------

-  Builds and passes all tests on:
-  Linux 32/64 bit, Python 2.7, 3.4, 3.5, 3.6 (probably works on 2.6 and
   3.3)
-  PC-BSD (FreeBSD) 64-bit, Python 2.7
-  OSX 64-bit, Python 3.6
-  Windows 32/64 bit (only tested on Python 2.7, 3.5 and 3.6, but should
   work on 3.3/3.4)

Version
-------

The version matched the latest version of NumPy where
``RandoMGenerator(MT19937())`` passes all NumPy test.

Documentation
-------------

| Documentation for the latest release is available on `my GitHub
  pages <http://bashtage.github.io/randomgen/>`__. Documentation for the
  latest commit (unreleased) is available under
| `devel <http://bashtage.github.io/randomgen/devel/>`__.

Plans
-----

This module is essentially complete. There are a few rough edges that
need to be smoothed.

-  Creation of additional streams from where supported (i.e. a
   ``next_stream()`` method)

Requirements
------------

Building requires:

-  Python (2.7, 3.4, 3.5, 3.6)
-  NumPy (1.10, 1.11, 1.12, 1.13, 1.14)
-  Cython (0.25+)
-  tempita (0.5+), if not provided by Cython

Testing requires pytest (3.0+).

**Note:** it might work with other versions but only tested with these
versions.

Development and Testing
-----------------------

All development has been on 64-bit Linux, and it is regularly tested on
Travis-CI (Linux/OSX) and Appveyor (Windows). The library is
occasionally tested on Linux 32-bit and Free BSD 11.1.

Basic tests are in place for all RNGs. The MT19937 is tested against
NumPy's implementation for identical results. It also passes NumPy's
test suite where still relevant.

Installing
----------

.. code:: bash

    python setup.py install

SSE2
~~~~

``dSFTM`` makes use of SSE2 by default. If you have a very old computer
or are building on non-x86, you can install using:

.. code:: bash

    python setup.py install --no-sse2

Windows
~~~~~~~

Either use a binary installer, or if building from scratch, use Python
3.6 with Visual Studio 2015 Community Edition. It can also be build
using Microsoft Visual C++ Compiler for Python 2.7 and Python 2.7,
although some modifications may be needed to ``distutils`` to find the
compiler.

Using
-----

The separate generators are importable from ``randomgen``

.. code:: python

    from randomgen import RandomGenerator, ThreeFry, PCG64, MT19937
    rg = RandomGenerator(ThreeFry())
    rg.random_sample(100)

    rg = RandomGenerator(PCG64())
    rg.random_sample(100)

    # Identical to NumPy
    rg = RandomGenerator(MT19937())
    rg.random_sample(100)

License
-------

Standard NCSA, plus sub licenses for components.

Performance
-----------

Performance is promising, and even the mt19937 seems to be faster than
NumPy's mt19937.

::

    Speed-up relative to NumPy (Uniform Doubles)
    ************************************************************
    DSFMT           137.1%
    MT19937          21.0%
    PCG32           101.2%
    PCG64           110.7%
    Philox           -2.7%
    ThreeFry        -11.4%
    ThreeFry32      -62.3%
    Xoroshiro128    181.4%
    Xorshift1024    141.8%

    Speed-up relative to NumPy (64-bit unsigned integers)
    ************************************************************
    DSFMT            24.8%
    MT19937          15.0%
    PCG32            92.6%
    PCG64            99.0%
    Philox          -20.4%
    ThreeFry        -21.7%
    ThreeFry32      -64.4%
    Xoroshiro128    164.2%
    Xorshift1024    120.8%

    Speed-up relative to NumPy (Standard normals)
    ************************************************************
    DSFMT           299.4%
    MT19937         271.2%
    PCG32           364.5%
    PCG64           364.2%
    Philox          256.9%
    ThreeFry        236.0%
    ThreeFry32       97.0%
    Xoroshiro128    477.4%
    Xorshift1024    360.7%

.. |Travis Build Status| image:: https://travis-ci.org/bashtage/randomgen.svg?branch=master
   :target: https://travis-ci.org/bashtage/randomgen
.. |Appveyor Build Status| image:: https://ci.appveyor.com/api/projects/status/odc5c4ukhru5xicl/branch/master?svg=true
   :target: https://ci.appveyor.com/project/bashtage/randomgen/branch/master
