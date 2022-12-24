# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for numpy).
m_dep = cc.find_library('m', required : false)
mlib_linkflag = ''
if m_dep.found()
  mlib_linkflag = '-lm'
  add_project_link_arguments(mlib_linkflag, language : 'c')
endif

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  if is_mingw
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
    # Manual add of MS_WIN64 macro when not using MSVC.
    # https://bugs.python.org/issue28267
    bitness = run_command('_build_utils/gcc_build_bitness.py').stdout().strip()
    if bitness == '64'
      add_project_arguments('-DMS_WIN64', language: ['c', 'cpp'])
    endif
  endif
endif

# Enable UNIX large file support on 32-bit systems (64 bit off_t,
# lseek -> lseek64, etc.)
cflags_large_file_support = []
if host_machine.system() == 'aix'
  cflags_large_file_support += '-D_LARGE_FILES'
else
  cflags_large_file_support += [
    '-D_FILE_OFFSET_BITS=64',
    '-D_LARGEFILE_SOURCE=1',
    '-D_LARGEFILE64_SOURCE=1',
  ]
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
blas = dependency(blas_name, required: false)
lapack = dependency(lapack_name, required: false)

dependency_map = {
  'BLAS': blas,
  'LAPACK': lapack,
}

# BLAS and LAPACK are optional dependencies for NumPy. We can only use a BLAS
# which provides a CBLAS interface.
# TODO: add ILP64 support
have_blas = blas.found() # TODO: and blas.has_cblas()
have_lapack = lapack.found()

# Copy the main __init__.py|pxd files to the build dir (needed for Cython)
__init__py = fs.copyfile('__init__.py')
__init__pxd = fs.copyfile('__init__.pxd')
__init__pxd30 = fs.copyfile('__init__.cython-30.pxd')
_cython_tree = [__init__py, __init__pxd, __init__pxd30]

python_sources = [
  '__init__.cython-30.pxd',
  '__init__.pxd',
  '__init__.py',
  '__init__.pyi',
  '_distributor_init.py',
  '_globals.py',
  '_pytesttester.py',
  '_pytesttester.pyi',
  '_version.py',
  'conftest.py',
  'ctypeslib.py',
  'ctypeslib.pyi',
  'dual.py',
  'exceptions.py',
  'exceptions.pyi',
  'matlib.py',
  'py.typed',
  'version.py'
]

py.install_sources(
  python_sources,
  subdir: 'numpy'
)

src_file_cli = find_program('_build_utils/process_src_template.py')
src_file = generator(src_file_cli,
  arguments : ['@INPUT@', '--outfile', '@OUTPUT@'],
  output : '@BASENAME@'
)

tempita_cli = find_program('_build_utils/tempita.py')
tempita = generator(tempita_cli,
  arguments : ['@INPUT@', '--outfile', '@OUTPUT@'],
  output : '@BASENAME@'
)

pure_subdirs = [
  '_pyinstaller',
  '_typing',
  '_utils',
  'array_api',
  'compat',
  'distutils',
  'doc',
  'f2py',
  'lib',
  'ma',
  'matrixlib',
  'polynomial',
  'testing',
  'tests',
  'typing',
]

np_dir = py.get_install_dir() / 'numpy'

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: np_dir)
endforeach

compilers = {
  'C': cc,
  'CPP': cpp,
  'CYTHON': meson.get_compiler('cython')
}

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

if meson.is_cross_build()
  machines += { 'TARGET': target_machine }
endif

conf_data = configuration_data()

# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', compiler.cmd_array())
endforeach

# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

# Python information
conf_data.set('PYTHON_PATH', py.path())
conf_data.set('PYTHON_VERSION', py.language_version())

# Dependencies information
foreach name, dep : dependency_map
  conf_data.set(name + '_NAME', dep.name())
  conf_data.set(name + '_FOUND', dep.found())
  if dep.found()
    conf_data.set(name + '_VERSION', dep.version())
    conf_data.set(name + '_TYPE_NAME', dep.type_name())
    conf_data.set(name + '_INCLUDEDIR', dep.get_variable('includedir'))
    conf_data.set(name + '_LIBDIR', dep.get_variable('libdir'))
    conf_data.set(name + '_OPENBLAS_CONFIG', dep.get_variable('openblas_config'))
    conf_data.set(name + '_PCFILEDIR', dep.get_variable('pcfiledir'))
  endif
endforeach

configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration : conf_data,
  install_dir: np_dir,
)

subdir('core')
subdir('fft')
subdir('linalg')
subdir('random')
