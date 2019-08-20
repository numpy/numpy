from collections import OrderedDict
from timeit import repeat

import pandas as pd

import numpy as np
from numpy.random import MT19937, PCG64, Philox, SFC64

PRNGS = [MT19937, PCG64, Philox, SFC64]

funcs = OrderedDict()
integers = 'integers(0, 2**{bits},size=1000000, dtype="uint{bits}")'
funcs['32-bit Unsigned Ints'] = integers.format(bits=32)
funcs['64-bit Unsigned Ints'] = integers.format(bits=64)
funcs['Uniforms'] = 'random(size=1000000)'
funcs['Normals'] = 'standard_normal(size=1000000)'
funcs['Exponentials'] = 'standard_exponential(size=1000000)'
funcs['Gammas'] = 'standard_gamma(3.0,size=1000000)'
funcs['Binomials'] = 'binomial(9, .1, size=1000000)'
funcs['Laplaces'] = 'laplace(size=1000000)'
funcs['Poissons'] = 'poisson(3.0, size=1000000)'

setup = """
from numpy.random import {prng}, Generator
rg = Generator({prng}())
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
npfuncs['64-bit Unsigned Ints'] = 'randint(2**64,dtype="uint64",size=1000000)'
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
table['RandomState'] = pd.Series(col)

columns = ['MT19937','PCG64','Philox','SFC64', 'RandomState']
table = pd.DataFrame(table)
order = np.log(table).mean().sort_values().index
table = table.T
table = table.reindex(columns)
table = table.T
table = table.reindex([k for k in funcs], axis=0)
print(table.to_csv(float_format='%0.1f'))


rel = table.loc[:, ['RandomState']].values @ np.ones(
    (1, table.shape[1])) / table
rel.pop('RandomState')
rel = rel.T
rel['Overall'] = np.exp(np.log(rel).mean(1))
rel *= 100
rel = np.round(rel)
rel = rel.T
print(rel.to_csv(float_format='%0d'))

# Cross-platform table
rows = ['32-bit Unsigned Ints','64-bit Unsigned Ints','Uniforms','Normals','Exponentials']
xplat = rel.reindex(rows, axis=0)
xplat = 100 * (xplat / xplat.MT19937.values[:,None])
overall = np.exp(np.log(xplat).mean(0))
xplat = xplat.T.copy()
xplat['Overall']=overall
print(xplat.T.round(1))



