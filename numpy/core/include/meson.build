installed_headers = [
  'numpy/_neighborhood_iterator_imp.h',
  'numpy/arrayobject.h',
  'numpy/arrayscalars.h',
  'numpy/_dtype_api.h',
  'numpy/experimental_dtype_api.h',
  'numpy/halffloat.h',
  'numpy/ndarrayobject.h',
  'numpy/ndarraytypes.h',
  'numpy/noprefix.h',
  'numpy/npy_1_7_deprecated_api.h',
  'numpy/npy_3kcompat.h',
  'numpy/npy_common.h',
  'numpy/npy_cpu.h',
  'numpy/npy_endian.h',
  'numpy/npy_interrupt.h',
  'numpy/npy_math.h',
  'numpy/npy_no_deprecated_api.h',
  'numpy/npy_os.h',
  'numpy/numpyconfig.h',
  'numpy/old_defines.h',
  'numpy/ufuncobject.h',
  'numpy/utils.h',
]

# Note that oldnumeric.h is not installed on purpose. After investigation,
# there are only two active repos left which use it, and those got a heads up
# about removal of this header in numpy 1.25.0:
#   https://github.com/enthought/enable/issues/1005
#   https://github.com/fhs/pyhdf/issues/63

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
