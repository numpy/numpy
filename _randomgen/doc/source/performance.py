from collections import OrderedDict
from timeit import repeat

import numpy as np
import pandas as pd

from randomgen import MT19937, DSFMT, ThreeFry, PCG64, Xoroshiro128, \
    Xorshift1024, Philox, Xoshiro256StarStar, Xoshiro512StarStar

PRNGS = [DSFMT, MT19937, Philox, PCG64, ThreeFry, Xoroshiro128, Xorshift1024,
         Xoshiro256StarStar, Xoshiro512StarStar]

funcs = {'32-bit Unsigned Ints': 'random_uintegers(size=1000000,bits=32)',
         '64-bit Unsigned Ints': 'random_uintegers(size=1000000,bits=32)',
         'Uniforms': 'random_sample(size=1000000)',
         'Complex Normals': 'complex_normal(size=1000000)',
         'Normals': 'standard_normal(size=1000000)',
         'Exponentials': 'standard_exponential(size=1000000)',
         'Gammas': 'standard_gamma(3.0,size=1000000)',
         'Binomials': 'binomial(9, .1, size=1000000)',
         'Laplaces': 'laplace(size=1000000)',
         'Poissons': 'poisson(3.0, size=1000000)', }

setup = """
from randomgen import {prng}
rg = {prng}().generator
"""

test = "rg.{func}"
table = OrderedDict()
for prng in PRNGS:
    print(prng)
    col = OrderedDict()
    for key in funcs:
        t = repeat(test.format(func=funcs[key]),
                   setup.format(prng=prng().__class__.__name__),
                   number=1, repeat=3)
        col[key] = 1000 * min(t)
    col = pd.Series(col)
    table[prng().__class__.__name__] = col

npfuncs = OrderedDict()
npfuncs.update(funcs)
npfuncs['32-bit Unsigned Ints'] = 'randint(2**32,dtype="uint32",size=1000000)'
npfuncs['64-bit Unsigned Ints'] = 'tomaxint(size=1000000)'
del npfuncs['Complex Normals']
setup = """
from numpy.random import RandomState
rg = RandomState()
"""
col = {}
for key in npfuncs:
    t = repeat(test.format(func=npfuncs[key]),
               setup.format(prng=prng().__class__.__name__),
               number=1, repeat=3)
    col[key] = 1000 * min(t)
table['NumPy'] = pd.Series(col)

table = pd.DataFrame(table)
table = table.reindex(table.mean(1).sort_values().index)
order = np.log(table).mean().sort_values().index
table = table.T
table = table.reindex(order)
table = table.T
print(table.to_csv(float_format='%0.1f'))

rel = table.loc[:, ['NumPy']].values @ np.ones((1, table.shape[1])) / table
rel.pop(rel.columns[0])
rel = rel.T
rel['Overall'] = np.exp(np.log(rel).mean(1))
rel *= 100
rel = np.round(rel)
rel = rel.T
print(rel.to_csv(float_format='%0d'))
