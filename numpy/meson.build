# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  if is_mingw
    add_global_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_global_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_global_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
    # Manual add of MS_WIN64 macro when not using MSVC.
    # https://bugs.python.org/issue28267
    #TODO: bitness = run_command('_build_utils/gcc_build_bitness.py').stdout().strip()
    if bitness == '64'
      add_global_arguments('-DMS_WIN64', language: ['c', 'cpp', 'fortran'])
    endif
  endif
endif

# TODO: 64-bit BLAS and LAPACK
#
# Note that this works as long as BLAS and LAPACK are detected properly via
# pkg-config. By default we look for OpenBLAS, other libraries can be configured via
# `meson configure -Dblas=blas -Dlapack=lapack` (example to build with Netlib
# BLAS and LAPACK).
# For MKL and for auto-detecting one of multiple libs, we'll need a custom
# dependency in Meson (like is done for scalapack) - see
# https://github.com/mesonbuild/meson/issues/2835
blas_name = get_option('blas')
lapack_name = get_option('lapack')
# pkg-config uses a lower-case name while CMake uses a capitalized name, so try
# that too to make the fallback detection with CMake work
if blas_name == 'openblas'
  blas_name = ['openblas', 'OpenBLAS']
endif
if lapack_name == 'openblas'
  lapack_name = ['openblas', 'OpenBLAS']
endif
blas = dependency(blas_name)
lapack = dependency(lapack_name)

# TODO: generate version.py and __config__.py (see scipy/meson.build)

cc = meson.get_compiler('c')

python_sources = [
  '_distributor_init.py',
  '_globals.py',
  '__init__.cython-30.pxd',
  '__init__.pxd',
  '__init__.py',
  '__init__.pyi',
  '_pytesttester.py',
  '_pytesttester.pyi',
  '_version.py',
  'conftest.py',
  'ctypeslib.py',
  'ctypeslib.pyi',
  'dual.py',
  'matlib.py',
  'setup.py',
  'version.py'
]

py.install_sources(
  python_sources,
  pure: false,
  subdir: 'numpy'
)

cython_cli = find_program('_build_utils/cythoner.py')

cython_gen = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@'],
  output : '@BASENAME@.c')

cython_gen_cpp = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@', '--cplus'],
  output : '@BASENAME@.cpp')

pure_subdirs = [
  'lib',
  'lib/tests',
  'lib/tests/data'
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: py.get_install_dir(pure: false) / 'numpy')
endforeach

subdir('core')
