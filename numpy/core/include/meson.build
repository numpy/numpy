installed_headers = [
  'numpy/_neighborhood_iterator_imp.h',
  'numpy/arrayobject.h',
  'numpy/arrayscalars.h',
  'numpy/_dtype_api.h',
  'numpy/experimental_dtype_api.h',
  'numpy/halffloat.h',
  'numpy/ndarrayobject.h',
  'numpy/ndarraytypes.h',
  'numpy/npy_1_7_deprecated_api.h',
  'numpy/npy_2_complexcompat.h',
  'numpy/npy_3kcompat.h',
  'numpy/npy_common.h',
  'numpy/npy_cpu.h',
  'numpy/npy_endian.h',
  'numpy/npy_math.h',
  'numpy/npy_no_deprecated_api.h',
  'numpy/npy_os.h',
  'numpy/numpyconfig.h',
  'numpy/ufuncobject.h',
  'numpy/utils.h',
]

py.install_sources(
  installed_headers,
  subdir: 'numpy/core/include/numpy'
)

py.install_sources(
  [
    'numpy/random/bitgen.h',
    'numpy/random/distributions.h',
  ],
  subdir: 'numpy/core/include/numpy/random'
)


py.install_sources(
  [
    'numpy/libdivide/libdivide.h',
    'numpy/libdivide/LICENSE.txt',
  ],
  subdir: 'numpy/core/include/numpy/random'
)
