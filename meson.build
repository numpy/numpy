project(
  'NumPy',
  'c', 'cpp', 'cython',
  # Note that the git commit hash cannot be added dynamically here
  # It is dynamically added upon import by versioneer
  # See `numpy/__init__.py`
  version: '1.24.0.dev0',
  license: 'BSD-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c99',
    'cpp_std=c++17',
    'blas=openblas',
    'lapack=openblas',
    'pkgconfig.relocatable=true',
  ],
)

fs = import('fs')

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')

# Check compiler is recent enough (see the SciPy Toolchain Roadmap for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.4')
    error('NumPy requires GCC >= 8.4')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('NumPy requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif
if not cy.version().version_compare('>=0.29.34')
  error('NumPy requires Cython >= 0.29.34')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

if not cc.has_header('Python.h', dependencies: py_dep)
  error('Cannot compile `Python.h`. Perhaps you need to install python-dev|python-devel')
endif

# Add default compile flags for any compiler that supports them.
# Note that MSVC does not support strict aliasing at all, and neither do the
# Intel compilers on Windows, so the `-fno` flavor of the flag should be fine.
add_project_arguments(
  cc.get_supported_arguments( '-fno-strict-aliasing'), language : 'c'
)

# Generate version number. Note that this will not (yet) update the version
# number seen by pip or reflected in wheel filenames. See
# https://github.com/mesonbuild/meson-python/issues/159 for that.
versioneer = files('generate_version.py')
if fs.exists('_version_meson.py')
    py.install_sources('_version_meson.py', subdir: 'numpy')
else
    custom_target('write_version_file',
        output: '_version_meson.py',
        command: [py, versioneer, '-o', '@OUTPUT@'],
        build_by_default: true,
        build_always_stale: true,
        install: true,
        install_dir: py.get_install_dir() / 'numpy'
    )
    meson.add_dist_script(py, versioneer, '-o', '_version_meson.py')
endif

subdir('numpy')
