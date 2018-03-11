# Core PRNG

[![Travis Build Status](https://travis-ci.org/bashtage/ng-numpy-randomstate.svg?branch=master)](https://travis-ci.org/bashtage/core-prng) 
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/odc5c4ukhru5xicl/branch/master?svg=true)](https://ci.appveyor.com/project/bashtage/core-prng/branch/master)

Experimental Core Pseudo Random Number Generator interface for future
NumPy RandomState evolution.

This is a library and generic interface for alternative random 
generators in Python and NumPy. 

## Features

* Replacement for NumPy's RandomState

```python
# import numpy.random as rnd
from core_prng import RandomGenerator, MT19937
rnd = RandomGenerator(MT19937())
x = rnd.standard_normal(100)
y = rnd.random_sample(100)
z = rnd.randn(10,10)
```

* Default random generator is a fast generator called Xoroshiro128plus
* Support for random number generators that support independent streams 
  and jumping ahead so that sub-streams can be generated
* Faster random number generation, especially for normal, standard
  exponential and standard gamma using the Ziggurat method

```python
from core_prng import RandomGenerator
# Use Xoroshiro128
rnd = RandomGenerator()
w = rnd.standard_normal(10000, method='zig')
x = rnd.standard_exponential(10000, method='zig')
y = rnd.standard_gamma(5.5, 10000, method='zig')
```

* Support for 32-bit floating randoms for core generators. 
  Currently supported:

    * Uniforms (`random_sample`)
    * Exponentials (`standard_exponential`, both Inverse CDF and Ziggurat)
    * Normals (`standard_normal`, both Box-Muller and Ziggurat)
    * Standard Gammas (via `standard_gamma`, both Inverse CDF and Ziggurat)
  
  **WARNING**: The 32-bit generators are **experimental** and subject 
  to change.
  
  **Note**: There are _no_ plans to extend the alternative precision 
  generation to all random number types.

* Support for filling existing arrays using `out` keyword argument. Currently
  supported in (both 32- and 64-bit outputs)

    * Uniforms (`random_sample`)
    * Exponentials (`standard_exponential`)
    * Normals (`standard_normal`)
    * Standard Gammas (via `standard_gamma`)

## Included Pseudo Random Number Generators

This modules includes a number of alternative random 
number generators in addition to the MT19937 that is included in NumPy. 
The RNGs include:

* [MT19937](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/),
 the NumPy rng
* [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/) a 
  SSE2-aware version of the MT19937 generator that is especially fast at 
  generating doubles
* [xoroshiro128+](http://xoroshiro.di.unimi.it/) and
  [xorshift1024*](http://xorshift.di.unimi.it/)
* [PCG64](http:w//www.pcg-random.org/)
* ThreeFry and Philox implementationf from [Random123](https://www.deshawrsearch.com/resources_random123.html)
## Differences from `numpy.random.RandomState`

### New Features
* `standard_normal`, `normal`, `randn` and `multivariate_normal` all 
  support an additional `method` keyword argument which can be `bm` or
  `zig` where `bm` corresponds to the current method using the Box-Muller
  transformation and `zig` uses the much faster (100%+) Ziggurat method.
* `standard_exponential` and `standard_gamma` both support an additional
  `method` keyword argument which can be `inv` or
  `zig` where `inv` corresponds to the current method using the inverse
  CDF and `zig` uses the much faster (100%+) Ziggurat method.
* Core random number generators can produce either single precision
  (`np.float32`) or double precision (`np.float64`, the default) using
  an the optional keyword argument `dtype`
* Core random number generators can fill existing arrays using the
  `out` keyword argument


### New Functions

* `random_entropy` - Read from the system entropy provider, which is 
commonly used in cryptographic applications
* `random_raw` - Direct access to the values produced by the underlying 
PRNG. The range of the values returned depends on the specifics of the 
PRNG implementation.
* `random_uintegers` - unsigned integers, either 32- (`[0, 2**32-1]`)
or 64-bit (`[0, 2**64-1]`)
* `jump` - Jumps RNGs that support it.  `jump` moves the state a great 
distance. _Only available if supported by the RNG._
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, 
without actually drawing the numbers. _Only available if supported by 
the RNG._

## Status

* Replacement for `numpy.random.RandomState`. The 
  `MT19937` generator is identical to `numpy.random.RandomState`, and 
  will produce an identical sequence of random numbers for a given seed.   
* Builds and passes all tests on:
  * Linux 32/64 bit, Python 2.7, 3.4, 3.5, 3.6 (probably works on 2.6 and 3.3)
  * PC-BSD (FreeBSD) 64-bit, Python 2.7
  * OSX 64-bit, Python 2.7
  * Windows 32/64 bit (only tested on Python 2.7, 3.5 and 3.6, but
    should work on 3.3/3.4)

## Version
The version matched the latest version of NumPy where 
`RandoMGenerator(MT19937())` passes all NumPy test.

## Documentation

An occasionally updated build of the documentation is available on
[my github pages](http://bashtage.github.io/core-prng/).

## Plans
This module is essentially complete.  There are a few rough edges that 
need to be smoothed.
  
* Creation of additional streams from a RandomState where supported 
  (i.e. a `next_stream()` method)
  
## Requirements
Building requires:

  * Python (2.7, 3.4, 3.5, 3.6)
  * NumPy (1.9, 1.10, 1.11, 1.12)
  * Cython (0.22, **not** 0.23, 0.24, 0.25)
  * tempita (0.5+), if not provided by Cython
 
Testing requires pytest (3.0+).

**Note:** it might work with other versions but only tested with these 
versions. 

## Development and Testing

All development has been on 64-bit Linux, and it is regularly tested on 
Travis-CI (Linux) and Appveyor (Windows). The library is occasionally 
tested on Linux 32-bit, OSX 10.13, Free BSD 11.1.

Basic tests are in place for all RNGs. The MT19937 is tested against 
NumPy's implementation for identical results. It also passes NumPy's 
test suite.

## Installing

```bash
python setup.py install
```

### SSE2
`dSFTM` makes use of SSE2 by default.  If you have a very old computer 
or are building on non-x86, you can install using:

```bash
python setup.py install --no-sse2
```

### Windows
Either use a binary installer, or if building from scratch, use 
Python 3.6 with Visual Studio 2015 Community Edition. It can also be 
build using Microsoft Visual C++ Compiler for Python 2.7 and Python 2.7, 
although some modifications may be needed to `distutils` to find the 
compiler.

## Using

The separate generators are importable from `core_prng`

```python
from core_prng import RandomGenerator, ThreeFry, PCG64, MT19937
rg = RandomGenerator(ThreeFry())
rg.random_sample(100)

rg = RandomGenerator(PCG64())
rg.random_sample(100)

# Identical to NumPy
rg = RandomGenerator(MT19937())
rg.random_sample(100)
```

## License
Standard NCSA, plus sub licenses for components.

## Performance
Performance is promising, and even the mt19937 seems to be faster than 
NumPy's mt19937. 

```
Speed-up relative to NumPy (Uniform Doubles)
************************************************************
MT19937          22.9%
PCG64           109.6%
Philox           -6.2%
ThreeFry        -16.6%
Xoroshiro128    161.0%
Xorshift1024    119.9%

Speed-up relative to NumPy (64-bit unsigned integers)
************************************************************
MT19937           6.2%
PCG64            88.2%
Philox          -23.0%
ThreeFry        -26.5%
Xoroshiro128    142.4%
Xorshift1024    107.5%

Speed-up relative to NumPy (Standard normals (Box-Muller))
************************************************************
MT19937          17.7%
PCG64            35.6%
Philox          -26.2%
ThreeFry        -16.9%
Xoroshiro128     57.9%
Xorshift1024     40.9%

Speed-up relative to NumPy (Standard normals (Ziggurat))
************************************************************
MT19937         107.9%
PCG64           149.6%
Philox           11.1%
ThreeFry         78.8%
Xoroshiro128    224.7%
Xorshift1024    158.6%
```