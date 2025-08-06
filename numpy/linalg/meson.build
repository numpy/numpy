# Note that `python_xerbla.c` was excluded on Windows in setup.py;
# unclear why and it seems needed, so unconditionally used here.
python_xerbla_sources = ['lapack_lite/python_xerbla.c']

lapack_lite_sources = []
if not have_lapack
  lapack_lite_sources = [
    'lapack_lite/f2c.c',
    'lapack_lite/f2c_c_lapack.c',
    'lapack_lite/f2c_d_lapack.c',
    'lapack_lite/f2c_s_lapack.c',
    'lapack_lite/f2c_z_lapack.c',
    'lapack_lite/f2c_blas.c',
    'lapack_lite/f2c_config.c',
    'lapack_lite/f2c_lapack.c',
  ]
endif

py.extension_module('lapack_lite',
  [
    'lapack_litemodule.c',
    python_xerbla_sources,
    lapack_lite_sources,
  ],
  dependencies: [np_core_dep, blas_dep, lapack_dep],
  install: true,
  subdir: 'numpy/linalg',
)

py.extension_module('_umath_linalg',
  [
    'umath_linalg.cpp',
    python_xerbla_sources,
    lapack_lite_sources,
  ],
  dependencies: [np_core_dep, blas_dep, lapack_dep],
  link_with: npymath_lib,
  install: true,
  subdir: 'numpy/linalg',
)

py.install_sources(
  [
    '__init__.py',
    '__init__.pyi',
    '_linalg.py',
    '_linalg.pyi',
    '_umath_linalg.pyi',
    'lapack_lite.pyi',
    'linalg.py',
    'linalg.pyi',
  ],
  subdir: 'numpy/linalg'
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_deprecations.py',
    'tests/test_linalg.py',
    'tests/test_regression.py',
  ],
  subdir: 'numpy/linalg/tests',
  install_tag: 'tests'
)
