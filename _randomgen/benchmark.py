import os
import struct
import timeit

import numpy as np
import pandas as pd
from numpy.random import RandomState

rs = RandomState()

SETUP = '''
import numpy as np
if '{brng}' == 'numpy':
    import numpy.random
    rg = numpy.random.RandomState()
else:
    from randomgen import RandomGenerator, {brng}
    rg = RandomGenerator({brng}())
rg.random_sample()
'''

scale_32 = scale_64 = 1
if struct.calcsize('P') == 8 and os.name != 'nt':
    # 64 bit
    scale_32 = 0.5
else:
    scale_64 = 2

PRNGS = ['DSFMT', 'PCG64', 'PCG32', 'MT19937', 'Xoroshiro128', 'Xorshift1024',
         'Xoshiro256StarStar', 'Xoshiro512StarStar', 'Philox', 'ThreeFry',
         'ThreeFry32', 'numpy']


def timer(code, setup):
    return 1000 * min(timeit.Timer(code, setup=setup).repeat(10, 10)) / 10.0


def print_legend(legend):
    print('\n' + legend + '\n' + '*' * max(60, len(legend)))


def run_timer(dist, command, numpy_command=None, setup='', random_type=''):
    print('-' * 80)
    if numpy_command is None:
        numpy_command = command

    res = {}
    for brng in PRNGS:
        cmd = numpy_command if brng == 'numpy' else command
        res[brng] = timer(cmd, setup=setup.format(brng=brng))

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


def timer_bounded(bits=8, max=95, use_masked=True):
    """
    Timer for 8-bit bounded values.

    Parameters
    ----------
    bits : {8, 16, 32, 64}
        Bit width of unsigned output type
    max : int
        Upper bound for range. Lower is always 0.  Must be <= 2**bits.
    use_masked: bool
        If True, masking and rejection sampling is used to generate a random
        number in an interval. If False, Lemire's algorithm is used if
        available to generate a random number in an interval.

    Notes
    -----
    Lemire's algorithm has improved performance when {max}+1 is not a
    power of two.
    """
    if bits not in (8, 16, 32, 64):
        raise ValueError('bits must be one of 8, 16, 32, 64.')
    minimum = 0

    dist = 'random_uintegers'

    if use_masked:  # Use masking & rejection.
        command = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint{bits}, use_masked=True)'
    else:  # Use Lemire's algo.
        command = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint{bits}, use_masked=False)'

    command = command.format(min=minimum, max=max, bits=bits)

    command_numpy = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint{bits})'
    command_numpy = command_numpy.format(min=minimum, max=max, bits=bits)

    run_timer(dist, command, command_numpy, SETUP,
              '{bits}-bit bounded unsigned integers (max={max}, '
              'use_masked={use_masked})'.format(max=max, use_masked=use_masked, bits=bits))


def timer_32bit():
    info = np.iinfo(np.uint32)
    minimum, maximum = info.min, info.max
    dist = 'random_uintegers'
    command = 'rg.random_uintegers(1000000, 32)'
    command_numpy = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint32)'
    command_numpy = command_numpy.format(min=minimum, max=maximum)
    run_timer(dist, command, command_numpy, SETUP, '32-bit unsigned integers')


def timer_64bit():
    info = np.iinfo(np.uint64)
    minimum, maximum = info.min, info.max
    dist = 'random_uintegers'
    command = 'rg.random_uintegers(1000000)'
    command_numpy = 'rg.randint({min}, {max}+1, 1000000, dtype=np.uint64)'
    command_numpy = command_numpy.format(min=minimum, max=maximum)
    run_timer(dist, command, command_numpy, SETUP, '64-bit unsigned integers')


def timer_normal_zig():
    dist = 'standard_normal'
    command = 'rg.standard_normal(1000000)'
    command_numpy = 'rg.standard_normal(1000000)'
    run_timer(dist, command, command_numpy, SETUP,
              'Standard normals (Ziggurat)')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--full',
                        help='Run benchmarks for a wide range of distributions.'
                             ' If not provided, only tests the production of '
                             'uniform values.',
                        dest='full', action='store_true')
    parser.add_argument('-bi', '--bounded-ints',
                        help='Included benchmark coverage of the bounded '
                             'integer generators in a full run.',
                        dest='bounded_ints', action='store_true')
    args = parser.parse_args()

    timer_uniform()
    if args.full:
        timer_raw()
        if args.bounded_ints:
            timer_bounded(use_masked=True)
            timer_bounded(max=64, use_masked=False)  # Worst case for Numpy.
            timer_bounded(max=95, use_masked=False)  # Typ. avrg. case for Numpy.
            timer_bounded(max=127, use_masked=False)  # Best case for Numpy.

            timer_bounded(16, use_masked=True)
            timer_bounded(16, max=1024, use_masked=False)  # Worst case for Numpy.
            timer_bounded(16, max=1535, use_masked=False)  # Typ. avrg. case for Numpy.
            timer_bounded(16, max=2047, use_masked=False)  # Best case for Numpy.

        timer_32bit()

        if args.bounded_ints:
            timer_bounded(32, use_masked=True)
            timer_bounded(32, max=1024, use_masked=False)  # Worst case for Numpy.
            timer_bounded(32, max=1535, use_masked=False)  # Typ. avrg. case for Numpy.
            timer_bounded(32, max=2047, use_masked=False)  # Best case for Numpy.

        timer_64bit()

        if args.bounded_ints:
            timer_bounded(64, use_masked=True)
            timer_bounded(64, max=1024, use_masked=False)  # Worst case for Numpy.
            timer_bounded(64, max=1535, use_masked=False)  # Typ. avrg. case for Numpy.
            timer_bounded(64, max=2047, use_masked=False)  # Best case for Numpy.

        timer_normal_zig()
