project(
  'NumPy',
  'c', 'cpp', 'cython',
  version: run_command(
    # This should become `numpy/_version.py` in NumPy 2.0
    ['numpy/_build_utils/gitversion.py'],
    check: true).stdout().strip(),
  license: 'BSD-3',
  meson_version: '>=1.5.2',  # version in vendored-meson is 1.5.2
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c11',
    'cpp_std=c++17',
    'pkgconfig.relocatable=true',
  ],
)

fs = import('fs')

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')

# Check compiler is recent enough (see the SciPy Toolchain Roadmap for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=9.3')
    error('NumPy requires GCC >= 9.3')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('NumPy requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif
if not cy.version().version_compare('>=3.0.6')
  error('NumPy requires Cython >= 3.0.6')
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
#
# Clang defaults to a non-strict floating error point model, but we need strict
# behavior. `-ftrapping-math` is equivalent to `-ffp-exception-behavior=strict`.
# This flag is also required to prevent the activation of SIMD partial load workarounds.
# For further clarification, refer to gh-24461.
cc_id = cc.get_id()
if cc_id.startswith('clang')
  # Determine the compiler flags for trapping math exceptions.
  trapping_math = {
    'clang-cl': '/clang:-ftrapping-math',
  }.get(cc_id, '-ftrapping-math')
  # Check if the compiler supports the trapping math flag.
  if cc.has_argument(trapping_math)
    # TODO: Consider upgrading the vendored Meson to 1.3.0 to support the parameter `werror`
    # Detect whether the compiler actually supports strict handling of floating-point exceptions
    # by treating warnings as errors.
    if cc.compiles('int main() { return 0; }', args: [trapping_math, '-Werror'])
      trapping_math = [trapping_math, '-DNPY_HAVE_CLANG_FPSTRICT']
    else
      # Suppress warnings about unsupported floating-point optimization.
      trapping_math = [trapping_math, '-Wno-unsupported-floating-point-opt']
      # Inform the user about the workaround.
      message(
        'NumPy is being built against a version of Clang that does not strictly enforce ' +
        'floating-point exception handling. Workarounds will be used, which may impact performance.\n' +
        'Consider upgrading Clang to the latest version.'
      )
    endif
    add_project_arguments(trapping_math, language: ['c', 'cpp'])
  endif
endif

if host_machine.system() == 'darwin' and cc.has_link_argument('-Wl,-ld_classic')
  # New linker introduced in macOS 14 not working yet with at least OpenBLAS in Spack,
  # see gh-24964 (and linked scipy issue from there).
  add_project_link_arguments('-Wl,-ld_classic', language : ['c', 'cpp'])
endif

subdir('meson_cpu')
subdir('numpy')
