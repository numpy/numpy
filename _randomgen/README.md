# Core PRNG

Experimental Core Pseudo Random Number Generator interface for future
NumPy RandomState evolution.

## Demo

Basic POC demonstration

```bash
python setup.py develop
```

```ipython
In [13]: import core_prng.generator

In [14]: rg = core_prng.generator.RandomGenerator()

In [15]: rg.random_integer()
Out[15]: 872337561037043212
```
