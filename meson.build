project(
  'NumPy',
  'c', 'cpp', 'cython',
  # Note that the git commit hash cannot be added dynamically here
  # It is dynamically added upon import by versioneer
  # See `numpy/__init__.py`
  version: '1.24.0.dev0',
  license: 'BSD-3',
  meson_version: '>= 0.64.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++11',
    'blas=openblas',
    'lapack=openblas'
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Check compiler is recent enough (see the SciPy Toolchain Roadmap for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.4')
    error('NumPy requires GCC >= 8.4')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=1910')
    error('NumPy requires at least vc141 (default with Visual Studio 2017) ' + \
          'when building with MSVC')
  endif
endif

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for numpy).
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# Hide symbols when building on Linux with GCC. For Python extension modules,
# we only need `PyInit_*` to be public, anything else may cause problems. So we
# use a linker script to avoid exporting those symbols (this is in addition to
# Meson using `-fvisibility=hidden` for C and `-fvisibility-inlines-hidden` for
# C++ code. See gh-15996 for details.
_linker_script = meson.project_source_root() / 'numpy/_build_utils/link-version-pyinit.map'
version_link_args = ['-Wl,--version-script=' + _linker_script]
# Note that FreeBSD only accepts version scripts when -shared is passed,
# hence we need to pass that to `cc.links` explicitly (flag is already
# present for `extension_module` invocations).
if not cc.links('', name: '-Wl,--version-script', args: ['-shared', version_link_args])
  version_link_args = []
endif

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

subdir('numpy')
