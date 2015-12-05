from __future__ import absolute_import, division, print_function

import numpy
import random

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
# while testing numpy we better not rely on numpy to produce random
# sequences
random.seed(1)
# but will seed it nevertheless
numpy.random.seed(1)

nx, ny = 1000, 1000
# reduced squares based on indexes_rand, primarily for testing more
# time-consuming functions (ufunc, linalg, etc)
nxs, nys = 100, 100

# a set of interesting types to test
TYPES1 = [
    'int16', 'float16',
    'int32', 'float32',
    'int64', 'float64',  'complex64',
    'longfloat', 'complex128',
    'complex256',
]

# values which will be used to construct our sample data matrices
# replicate 10 times to speed up initial imports of this helper
# and generate some redundancy
values = [random.uniform(0, 100) for x in range(nx*ny//10)]*10

squares = {t: numpy.array(values,
                          dtype=getattr(numpy, t)).reshape((nx, ny))
           for t in TYPES1}

# adjust complex ones to have non-degenerated imagery part -- use
# original data transposed for that
for t, v in squares.items():
    if t.startswith('complex'):
        v += v.T*1j

# smaller squares
squares_ = {t: s[:nxs, :nys] for t, s in squares.items()}
# vectors
vectors = {t: s[0] for t, s in squares.items()}

indexes = list(range(nx))
# so we do not have all items
indexes.pop(5)
indexes.pop(95)

indexes_rand = indexes[:]       # copy
random.shuffle(indexes_rand)         # in-place shuffle

# only now make them arrays
indexes = numpy.array(indexes)
indexes_rand = numpy.array(indexes_rand)
# smaller versions
indexes_ = indexes[indexes < nxs]
indexes_rand_ = indexes_rand[indexes_rand < nxs]


class Benchmark(object):
    goal_time = 0.25
