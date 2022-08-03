py.extension_module('_intrinsics',
  [
    '../../npy_cpu_features.c',
    'src/module.cpp',
	'src/intrinsics.dispatch.cpp',
	'src/datatypes.dispatch.cpp'
  ],
  c_args: c_args_common,
  dependencies: np_core_dep,
  install: true,
  subdir: 'numpy/core/_simd'
)

py.install_sources(
  [
    '__init__.py'
  ],
  subdir: 'numpy/core/_simd'
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_simd.py',
  ],
  subdir: 'numpy/core/_simd/tests'
)
