# Build npyrandom library
# -----------------------
npyrandom_sources = [
  'src/distributions/logfactorial.c',
  'src/distributions/distributions.c',
  'src/distributions/random_mvhg_count.c',
  'src/distributions/random_mvhg_marginals.c',
  'src/distributions/random_hypergeometric.c',
]

npyrandom_lib = static_library('npyrandom',
  npyrandom_sources,
  c_args: staticlib_cflags,
  dependencies: [py_dep, np_core_dep],
  install: true,
  install_dir: np_dir / 'random/lib',
  name_prefix: name_prefix_staticlib,
  name_suffix: name_suffix_staticlib,
)

# Build Cython extensions for numpy.random
# ----------------------------------------
# pyx -> c transpile output depends on copied __init__.py and pxd files
_cython_tree_random = [
  fs.copyfile('__init__.py'),
  fs.copyfile('__init__.pxd'),
  fs.copyfile('_common.pxd'),
  fs.copyfile('bit_generator.pxd'),
  fs.copyfile('c_distributions.pxd'),
]
# Need to use `custom_target` because we need to install this .pxd file
_cython_tree_random += custom_target('_bounded_integer_pxd',
  output: '_bounded_integers.pxd',
  input: '_bounded_integers.pxd.in',
  command: [tempita_cli, '@INPUT@', '-o', '@OUTPUT@'],
  install: true,
  install_dir: np_dir / 'random'
)

_bounded_integers_pyx = custom_target('_bounded_integer_pyx',
  output: '_bounded_integers.pyx',
  input: '_bounded_integers.pyx.in',
  command: [tempita_cli, '@INPUT@', '-o', '@OUTPUT@'],
)

c_args_random = [
  cflags_large_file_support,
  '-DNPY_NO_DEPRECATED_API=0',  # Cython still uses old NumPy C API
]
if host_machine.system() == 'cygwin'
  c_args_random += ['-Wl,--export-all-symbols']
endif

# name, sources, extra c_args, extra static libs to link
random_pyx_sources = [
  ['_bounded_integers', _bounded_integers_pyx, [], [npyrandom_lib, npymath_lib]],
  ['_common', '_common.pyx', [], [npyrandom_lib]],
  ['_mt19937', ['_mt19937.pyx', 'src/mt19937/mt19937.c', 'src/mt19937/mt19937-jump.c'],
    [], [npyrandom_lib]
  ],
  ['_philox', ['_philox.pyx', 'src/philox/philox.c'], [], [npyrandom_lib]],
  ['_pcg64', ['_pcg64.pyx', 'src/pcg64/pcg64.c'], ['-U__GNUC_GNU_INLINE__'], [npyrandom_lib]],
  ['_sfc64', ['_sfc64.pyx', 'src/sfc64/sfc64.c'], [], [npyrandom_lib]],
  ['bit_generator', 'bit_generator.pyx', [], [npyrandom_lib]],
  # The `fs.copyfile` usage here is needed because these two .pyx files import
  # from _bounded_integers,and its pxd file is only present in the build directory
  ['_generator', fs.copyfile('_generator.pyx'), [], [npyrandom_lib, npymath_lib]],
  ['mtrand', [
      fs.copyfile('mtrand.pyx'),
      'src/distributions/distributions.c',
      'src/legacy/legacy-distributions.c'
    ],
    ['-DNP_RANDOM_LEGACY=1'], [npymath_lib],
  ],
]
foreach gen: random_pyx_sources
  py.extension_module(gen[0],
    [gen[1], _cython_tree, _cython_tree_random],
    c_args: [c_args_random, gen[2]],
    include_directories: 'src',
    dependencies: np_core_dep,
    link_with: gen[3],
    install: true,
    subdir: 'numpy/random',
  )
endforeach

# Install Python sources, stub files, tests, examples and license
# ---------------------------------------------------------------
py.install_sources(
  [
    '__init__.pxd',
    '__init__.py',
    '__init__.pyi',
    '_common.pxd',
    '_generator.pyi',
    '_mt19937.pyi',
    '_pcg64.pyi',
    '_pickle.py',
    '_philox.pyi',
    '_sfc64.pyi',
    'bit_generator.pxd',
    'bit_generator.pyi',
    'c_distributions.pxd',
    'LICENSE.md',
    'mtrand.pyi',
  ],
  subdir: 'numpy/random'
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_direct.py',
    'tests/test_extending.py',
    'tests/test_generator_mt19937.py',
    'tests/test_generator_mt19937_regressions.py',
    'tests/test_random.py',
    'tests/test_randomstate.py',
    'tests/test_randomstate_regression.py',
    'tests/test_regression.py',
    'tests/test_seed_sequence.py',
    'tests/test_smoke.py',
  ],
  subdir: 'numpy/random/tests'
)

py.install_sources(
  [
    'tests/data/__init__.py',
    'tests/data/mt19937-testset-1.csv',
    'tests/data/mt19937-testset-2.csv',
    'tests/data/pcg64-testset-1.csv',
    'tests/data/pcg64-testset-2.csv',
    'tests/data/pcg64dxsm-testset-1.csv',
    'tests/data/pcg64dxsm-testset-2.csv',
    'tests/data/philox-testset-1.csv',
    'tests/data/philox-testset-2.csv',
    'tests/data/sfc64-testset-1.csv',
    'tests/data/sfc64-testset-2.csv',
  ],
  subdir: 'numpy/random/tests/data'
)

py.install_sources(
  [
  '_examples/cffi/extending.py',
  '_examples/cffi/parse.py',
  ],
  subdir: 'numpy/random/_examples/cffi'
)

py.install_sources(
  [
  '_examples/cython/extending.pyx',
  '_examples/cython/extending_distributions.pyx',
  '_examples/cython/meson.build',
  ],
  subdir: 'numpy/random/_examples/cython'
)

py.install_sources(
  [
  '_examples/numba/extending.py',
  '_examples/numba/extending_distributions.py',
  ],
  subdir: 'numpy/random/_examples/numba'
)
