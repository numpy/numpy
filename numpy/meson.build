# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for NumPy). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
mlib_linkflag = ''
if m_dep.found()
  mlib_linkflag = '-lm'
  add_project_link_arguments(mlib_linkflag, language : 'c')
endif

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_define('__MINGW32__') != ''

if is_mingw
  is_mingw_built_python = run_command(
    py, ['-c', 'import sysconfig; print(sysconfig.get_platform())'],
    check: true).stdout().strip().startswith('mingw')
  if not is_mingw_built_python
    # For mingw-w64, link statically against the UCRT.
    gcc_link_args = ['-lucrt', '-static']
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
  endif
  # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
  add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
endif

# We install libnpymath and libnpyrandom; ensure they're using a `.lib` rather
# than a `.a` file extension in order not to break including them in a
# distutils-based build (see gh-23981 and
# https://mesonbuild.com/FAQ.html#why-does-building-my-project-with-msvc-output-static-libraries-called-libfooa)
if is_windows and cc.get_id() == 'msvc'
  name_prefix_staticlib = ''
  name_suffix_staticlib = 'lib'
else
  name_prefix_staticlib = []
  name_suffix_staticlib = []
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

blas_name = get_option('blas')
lapack_name = get_option('lapack')
allow_noblas = get_option('allow-noblas')
# This is currently injected directly into CFLAGS/CXXFLAGS for wheel builds
# (see cibuildwheel settings in pyproject.toml), but used by CI jobs already
blas_symbol_suffix = get_option('blas-symbol-suffix')

use_ilp64 = get_option('use-ilp64')
if use_ilp64
  blas_interface = ['interface: ilp64']
else
  blas_interface = ['interface: lp64']
endif


blas_order = get_option('blas-order')
if blas_order == ['auto']
  blas_order = []
  if host_machine.system() == 'darwin'
    blas_order += 'accelerate'
  endif
  if host_machine.cpu_family() == 'x86_64'
    blas_order += 'mkl'
  endif
  blas_order += ['openblas', 'flexiblas', 'blis', 'blas']
endif
lapack_order = get_option('lapack-order')
if lapack_order == ['auto']
  lapack_order = []
  if host_machine.system() == 'darwin'
    lapack_order += 'accelerate'
  endif
  if host_machine.cpu_family() == 'x86_64'
    lapack_order += 'mkl'
  endif
  lapack_order += ['openblas', 'flexiblas', 'lapack']
endif

# MKL-specific options
_threading_opt = get_option('mkl-threading')
if _threading_opt == 'auto'
  # Switch default to iomp once conda-forge missing openmp.pc issue is fixed
  mkl_opts = ['threading: seq']
else
  mkl_opts = ['threading: ' + _threading_opt]
endif
blas_opts = {'mkl': mkl_opts}
mkl_version_req = '>=2023.0'  # see gh-24824
mkl_may_use_sdl = not use_ilp64 and _threading_opt in ['auto', 'iomp']

# Note that we can only use a BLAS which provides a CBLAS interface. So disable
# BLAS completely if CBLAS is not found.

# First try scipy-openblas, and if found don't look for cblas or lapack, we
# know what's inside the scipy-openblas wheels already.
if blas_name == 'openblas' or blas_name == 'auto'
  blas = dependency('scipy-openblas', method: 'pkg-config', required: false)
  if blas.found()
    blas_name = 'scipy-openblas'
  endif
endif
if blas_name == 'auto'
  foreach _name : blas_order
    if _name == 'mkl'
      blas = dependency('mkl',
        modules: ['cblas'] + blas_interface + mkl_opts,
        required: false,  # may be required, but we need to emit a custom error message
        version: mkl_version_req,
      )
      # Insert a second try with MKL, because we may be rejecting older versions
      # or missing it because no pkg-config installed. If so, we need to retry
      # with MKL SDL, and drop the version constraint (this always worked).
      if not blas.found() and mkl_may_use_sdl
        blas = dependency('mkl', modules: ['cblas', 'sdl: true'], required: false)
      endif
    else
      if _name == 'flexiblas' and use_ilp64
        _name = 'flexiblas64'
      endif
      blas = dependency(_name, modules: ['cblas'] + blas_interface, required: false)
    endif
    if blas.found()
      break
    endif
  endforeach
else
  if blas_name == 'mkl'
    blas = dependency('mkl',
      modules: ['cblas'] + blas_interface + mkl_opts,
      required: false,
      version: mkl_version_req,
    )
    # Same deal as above - try again for MKL
    if not blas.found() and mkl_may_use_sdl
      blas = dependency('mkl', modules: ['cblas', 'sdl: true'], required: false)
    endif
  else
    blas = dependency(blas_name, modules: ['cblas'] + blas_interface, required: false)
  endif
endif

have_blas = blas.found()
if have_blas
  _args_blas = ['-DHAVE_CBLAS']  # note: used for C and C++ via `blas_dep` below
  if use_ilp64
    _args_blas += ['-DHAVE_BLAS_ILP64']
    if 'openblas' in blas.name()
      _args_blas += ['-DOPENBLAS_ILP64_NAMING_SCHEME']
    endif
  endif
  if blas_symbol_suffix == 'auto'
    if blas_name == 'scipy-openblas' and use_ilp64
      blas_symbol_suffix = '64_'
    else
      blas_symbol_suffix = blas.get_variable('symbol_suffix', default_value: '')
    endif
    message(f'BLAS symbol suffix: @blas_symbol_suffix@')
  endif
  if blas_symbol_suffix != ''
    _args_blas += ['-DBLAS_SYMBOL_SUFFIX=' + blas_symbol_suffix]
  endif
  blas_dep = declare_dependency(
    dependencies: [blas],
    compile_args: _args_blas,
  )
else
  if allow_noblas
    blas_dep = []
  else
    error('No BLAS library detected! Install one, or use the ' + \
          '`allow-noblas` build option (note, this may be up to 100x slower ' + \
          'for some linear algebra operations).')
  endif
endif

if 'mkl' in blas.name() or blas.name() == 'accelerate' or blas_name == 'scipy-openblas'
  # For these libraries we know that they contain LAPACK, and it's desirable to
  # use that - no need to run the full detection twice.
  lapack = blas
else
  if lapack_name == 'auto'
    foreach _name : lapack_order
      lapack = dependency(_name, modules: ['lapack'] + blas_interface, required: false)
      if lapack.found()
        break
      endif
    endforeach
  else
    lapack = dependency(lapack_name, modules: ['lapack'] + blas_interface, required: false)
  endif
endif

have_lapack = lapack.found()
if not have_lapack and not allow_noblas
  error('No LAPACK library detected! Install one, or use the ' + \
        '`allow-noblas` build option (note, this may be up to 100x slower ' + \
        'for some linear algebra operations).')
else
  lapack_dep = declare_dependency(dependencies: [lapack, blas_dep])
endif

# Determine whether it is necessary to link libatomic with gcc. This
# could be the case on 32-bit platforms when atomic operations are used
# on 64-bit types or on RISC-V using 8-bit atomics, so we explicitly
# check for both 64 bit and 8 bit operations.  The check is adapted from
# SciPy, who copied it from Mesa.
null_dep = dependency('', required : false)
atomic_dep = null_dep
code_non_lockfree = '''
  #include <stdint.h>
  #include <stddef.h>
  int main() {
   struct {
     void *p;
     uint8_t u8v;
   } x;
   x.p = NULL;
   x.u8v = 0;
   uint8_t res = __atomic_load_n(&x.u8v, __ATOMIC_SEQ_CST);
   __atomic_store_n(&x.u8v, 1, __ATOMIC_SEQ_CST);
   void *p = __atomic_load_n((void **)x.p, __ATOMIC_SEQ_CST);
   __atomic_store_n((void **)x.p, NULL, __ATOMIC_SEQ_CST);
   return 0;
   }
'''
if cc.get_id() != 'msvc'
  if not cc.links(
      code_non_lockfree,
      name : 'Check atomic builtins without -latomic'
    )
    atomic_dep = cc.find_library('atomic', required: false)
    if atomic_dep.found()
      # We're not sure that with `-latomic` things will work for all compilers,
      # so verify and only keep libatomic as a dependency if this works. It is
      # possible the build will fail later otherwise - unclear under what
      # circumstances (compilers, runtimes, etc.) exactly and this may need to
      # be extended when support is added for new CPUs
      if not cc.links(
          code_non_lockfree,
          dependencies: atomic_dep,
          name : 'Check atomic builtins with -latomic'
        )
        atomic_dep = null_dep
      endif
    endif
  endif
endif

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
  '__config__.pyi',
  '_array_api_info.py',
  '_array_api_info.pyi',
  '_configtool.py',
  '_configtool.pyi',
  '_distributor_init.py',
  '_distributor_init.pyi',
  '_globals.py',
  '_globals.pyi',
  '_pytesttester.py',
  '_pytesttester.pyi',
  '_expired_attrs_2_0.py',
  '_expired_attrs_2_0.pyi',
  'conftest.py',
  'exceptions.py',
  'exceptions.pyi',
  'dtypes.py',
  'dtypes.pyi',
  'matlib.py',
  'matlib.pyi',
  'py.typed',
  'version.pyi',
]

if fs.exists('_distributor_init_local.py')
  python_sources += ['_distributor_init_local.py']
endif

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

pure_subdirs = [
  '_pyinstaller',
  '_typing',
  '_utils',
  'ctypeslib',
  'doc',
  'f2py',
  'lib',
  'ma',
  'matrixlib',
  'polynomial',
  'testing',
  'typing',
  'rec',
  'char',
  'core',
  'strings',
]
if py.version().version_compare('<3.12')
  pure_subdirs += 'distutils'
endif

np_dir = py.get_install_dir() / 'numpy'

# Generate version.py for sdist
meson.add_dist_script(
   ['_build_utils/gitversion.py', '--meson-dist', '--write',
     'numpy/version.py']
)
if not fs.exists('version.py')
  generate_version = custom_target(
    'generate-version',
    install: true,
    build_always_stale: true,
    build_by_default: true,
    output: 'version.py',
    input: '_build_utils/gitversion.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install_dir: np_dir,
    install_tag: 'python-runtime'
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir : 'numpy'
  )
endif

foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: np_dir, install_tag: 'python-runtime', exclude_directories: ['tests'])
  if fs.is_dir(subdir/'tests')
    install_subdir(subdir/'tests', install_dir: np_dir/subdir, install_tag: 'tests')
  endif
endforeach

install_subdir('tests', install_dir: np_dir, install_tag: 'tests')

compilers = {
  'C': cc,
  'CPP': cpp,
  'CYTHON': cy,
}

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

conf_data = configuration_data()

# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py.full_path())
conf_data.set('PYTHON_VERSION', py.language_version())

# BLAS/LAPACK dependency info. Ensure we report dependencies correctly for
# `np.show_config()`; needs some special handling for the case BLAS was found
# but CBLAS not (and hence BLAS was also disabled)
dependency_map = {
  'LAPACK': lapack,
}
if have_blas
  dependency_map += {'BLAS': blas}
else
  conf_data.set('BLAS_NAME', blas_name)
  conf_data.set('BLAS_FOUND', false)
endif


foreach name, dep : dependency_map
  conf_data.set(name + '_NAME', dep.name())
  conf_data.set(name + '_FOUND', dep.found())
  if dep.found()
    conf_data.set(name + '_VERSION', dep.version())
    conf_data.set(name + '_TYPE_NAME', dep.type_name())
    # get_variable() results may be missing for a variety of reasons
    conf_data.set(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
    conf_data.set(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
    conf_data.set(name + '_OPENBLAS_CONFIG', dep.get_variable('openblas_config', default_value: 'unknown'))
    conf_data.set(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
  endif
endforeach

configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration : conf_data,
  install_dir: np_dir,
  install_tag: 'python-runtime'
)

subdir('_core')
subdir('fft')
subdir('linalg')
subdir('random')
