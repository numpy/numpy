"""
Use cffi to access the underlying C functions from distributions.h
"""
import os
import numpy as np
import cffi
ffi = cffi.FFI()

inc_dir = os.path.join(np.get_include(), 'numpy')

# Basic numpy types
ffi.cdef('''
    typedef intptr_t npy_intp;
    typedef unsigned char npy_bool;

''')

with open(os.path.join(inc_dir, 'random', 'bitgen.h')) as fid:
    s = []
    for line in fid:
        # massage the include file
        if line.strip().startswith('#'):
            continue
        s.append(line)
    ffi.cdef('\n'.join(s))
        
with open(os.path.join(inc_dir, 'random', 'distributions.h')) as fid:
    s = []
    in_skip = 0
    for line in fid:
        # massage the include file
        if line.strip().startswith('#'):
            continue

        # skip any inlined function definition
        # which starts with 'static NPY_INLINE xxx(...) {'
        # and ends with a closing '}'
        if line.strip().startswith('static NPY_INLINE'):
            in_skip += line.count('{')
            continue
        elif in_skip > 0:
            in_skip += line.count('{')
            in_skip -= line.count('}')
            continue

        # replace defines with their value or remove them
        line = line.replace('DECLDIR', '')
        line = line.replace('NPY_INLINE', '')
        line = line.replace('RAND_INT_TYPE', 'int64_t')
        s.append(line)
    ffi.cdef('\n'.join(s))

lib = ffi.dlopen(np.random._generator.__file__)

# Compare the distributions.h random_standard_normal_fill to
# Generator.standard_random
bit_gen = np.random.PCG64()
rng = np.random.Generator(bit_gen)
state = bit_gen.state

interface = rng.bit_generator.cffi
n = 100
vals_cffi = ffi.new('double[%d]' % n)
lib.random_standard_normal_fill(interface.bit_generator, n, vals_cffi)

# reset the state
bit_gen.state = state

vals = rng.standard_normal(n)

for i in range(n):
    assert vals[i] == vals_cffi[i]
