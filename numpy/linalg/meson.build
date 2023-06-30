lapack_lite_sources = [
  'lapack_lite/f2c.c',
  'lapack_lite/f2c_c_lapack.c',
  'lapack_lite/f2c_d_lapack.c',
  'lapack_lite/f2c_s_lapack.c',
  'lapack_lite/f2c_z_lapack.c',
  'lapack_lite/f2c_blas.c',
  'lapack_lite/f2c_config.c',
  'lapack_lite/f2c_lapack.c',
  'lapack_lite/python_xerbla.c',
]

# TODO: ILP64 support

lapack_lite_module_src = ['lapack_litemodule.c']
if not have_lapack
  warning('LAPACK was not found, NumPy is using an unoptimized, naive build from sources!')
  lapack_lite_module_src += lapack_lite_sources
endif

py.extension_module('lapack_lite',
  lapack_lite_module_src,
  dependencies: [np_core_dep, blas_dep, lapack_dep],
  install: true,
  subdir: 'numpy/linalg',
)

_umath_linalg_src = ['umath_linalg.cpp'] + lapack_lite_sources

py.extension_module('_umath_linalg',
  _umath_linalg_src,
  dependencies: np_core_dep,
  link_with: npymath_lib,
  install: true,
  subdir: 'numpy/linalg',
)

py.install_sources(
  [
    '__init__.py',
    '__init__.pyi',
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
  subdir: 'numpy/linalg/tests'
)
