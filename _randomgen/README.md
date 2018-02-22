# Core PRNG

Experimental Core Pseudo Random Number Generator interface for future
NumPy RandomState evolution.

## Demo

Basic POC demonstration

```bash
python setup.py develop
```

```ipython
In [1]: import core_prng.generator

# Default generator is Splitmix64
In [2]: rg = core_prng.generator.RandomGenerator()

In [3]: rg.random_integer()
Out[3]: 872337561037043212

In [4]: from core_prng.xoroshiro128 import Xoroshiro128

# Swap the generator
In [5]: rg = core_prng.generator.RandomGenerator(Xoroshiro128())

In [6]: rg.random_integer()
Out[6]: 13370384800127340062
```
