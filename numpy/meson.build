# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

cython_c_args = []
if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  add_global_link_arguments(gcc_link_args, language: ['c', 'cpp'])
  # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
  add_global_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
  # Manual add of MS_WIN64 macro when not using MSVC.
  # https://bugs.python.org/issue28267
  bitness = run_command('_build_utils/gcc_build_bitness.py').stdout().strip()
  if bitness == '64'
    add_global_arguments('-DMS_WIN64', language: ['c', 'cpp'])
  endif
  # Silence warnings emitted by PyOS_snprintf for (%zd), see
  # https://github.com/rgommers/scipy/issues/118.
  # Use as c_args for extensions containing Cython code
  cython_c_args += ['-Wno-format-extra-args', '-Wno-format']
endif

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

c_undefined_ok = ['-Wno-maybe-uninitialized']

pure_subdirs = [
  'lib',
  'lib/tests',
  'lib/tests/data'
]

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: py.get_install_dir(pure: false) / 'numpy')
endforeach
