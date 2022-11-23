largefile_define = []
if host_machine.system() == 'aix' or host_machine.system() == 'AIX'
  largefile_define += '-D_LARGE_FILES'
endif

py.extension_module('_pocketfft_internal',
  '_pocketfft.c',
  c_args: largefile_define,
  dependencies: np_core_dep,
  install: true,
  subdir: 'numpy/fft',
)

py.install_sources(
  [
    '__init__.py',
    '__init__.pyi',
    '_pocketfft.py',
    '_pocketfft.pyi',
    'helper.py',
    'helper.pyi',
  ],
  subdir: 'numpy/fft'
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_helper.py',
    'tests/test_pocketfft.py',
  ],
  subdir: 'numpy/fft/tests'
)
