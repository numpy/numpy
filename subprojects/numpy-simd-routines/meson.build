project(
  'NumPy SIMD Routines',
  'cpp', 
  version: 'master',
  license: 'BSD-3',
  meson_version: '>=1.5.2', 
  default_options: [
    'buildtype=debugoptimized',
    'cpp_std=c++17',
  ],
)
fs = import('fs')
if not fs.exists('src/npsr')
  error('Missing the `numpy-simd-routines` git submodule! Run `git submodule update --init` to fix this.')
endif
npsr_inc = include_directories('src')
npsr_dep = declare_dependency(
  include_directories: npsr_inc
)

