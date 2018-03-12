import os
import struct
import timeit

import numpy as np
import pandas as pd
from numpy.random import RandomState

rs = RandomState()

SETUP = '''
import numpy as np
if '{prng}' == 'numpy':
    import numpy.random
    rg = numpy.random.RandomState()
else:
    from core_prng import RandomGenerator, {prng}
    rg = RandomGenerator({prng}())
rg.random_sample()
'''

scale_32 = scale_64 = 1
if struct.calcsize('P') == 8 and os.name != 'nt':
    # 64 bit
    scale_32 = 0.5
else:
    scale_64 = 2

PRNGS = ['PCG64', 'MT19937', 'Xoroshiro128', 'Xorshift1024',
         'Philox', 'ThreeFry', 'ThreeFry32', 'numpy']


def timer(code, setup):
    return 1000 * min(timeit.Timer(code, setup=setup).repeat(10, 10)) / 10.0


def print_legend(legend):
    print('\n' + legend + '\n' + '*' * max(60, len(legend)))


def run_timer(dist, command, numpy_command=None, setup='', random_type=''):
    print('-' * 80)
    if numpy_command is None:
        numpy_command = command

    res = {}
    for prng in PRNGS:
        cmd = numpy_command if prng == 'numpy' else command
        res[prng] = timer(cmd, setup=setup.format(prng=prng))

    s = pd.Series(res)
    t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
    print_legend('Time to produce 1,000,000 ' + random_type)
    print(t.sort_index())

    p = 1000.0 / s
    p = p.apply(lambda x: '{0:0.2f} million'.format(x))
    print_legend(random_type + ' per second')
    print(p.sort_index())

    baseline = [k for k in p.index if 'numpy' in k][0]
    p = 1000.0 / s
    p = p / p[baseline] * 100 - 100
    p = p.drop(baseline, 0)
    p = p.apply(lambda x: '{0:0.1f}%'.format(x))
    print_legend('Speed-up relative to NumPy')
    print(p.sort_index())
    print('-' * 80)


def timer_raw():
    dist = 'random_raw'
    command = 'rg.random_raw(size=1000000, output=False)'
    info = np.iinfo(np.int32)
    command_numpy = 'rg.random_integers({max},size=1000000)'
    command_numpy = command_numpy.format(max=info.max)
    run_timer(dist, command, command_numpy, SETUP, 'Raw Values')


def timer_uniform():
    dist = 'random_sample'
    command = 'rg.random_sample(1000000)'
    run_timer(dist, command, None, SETUP, 'Uniforms')


def timer_32bit():
    info = np.iinfo(np.uint32)
    min, max = info.min, info.max
    dist = 'random_uintegers'
    command = 'rg.random_uintegers(1000000, 32)'
    command_numpy = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint32)'
    command_numpy = command_numpy.format(min=min, max=max)
    run_timer(dist, command, command_numpy, SETUP, '32-bit unsigned integers')


def timer_64bit():
    info = np.iinfo(np.uint64)
    min, max = info.min, info.max
    dist = 'random_uintegers'
    command = 'rg.random_uintegers(1000000)'
    command_numpy = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint64)'
    command_numpy = command_numpy.format(min=min, max=max)
    run_timer(dist, command, command_numpy, SETUP, '64-bit unsigned integers')


def timer_normal():
    dist = 'standard_normal'
    command = 'rg.standard_normal(1000000, method="bm")'
    command_numpy = 'rg.standard_normal(1000000)'
    run_timer(dist, command, command_numpy, SETUP, 'Box-Muller normals')


def timer_normal_zig():
    dist = 'standard_normal'
    command = 'rg.standard_normal(1000000, method="zig")'
    command_numpy = 'rg.standard_normal(1000000)'
    run_timer(dist, command, command_numpy, SETUP,
              'Standard normals (Ziggurat)')


if __name__ == '__main__':
    timer_raw()
    timer_uniform()
    timer_32bit()
    timer_64bit()
    timer_normal()
    timer_normal_zig()
