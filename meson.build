project(
  'NumPy',
  'c', 'cpp', 'cython',
  version: run_command(
    # This should become `numpy/_version.py` in NumPy 2.0
    ['numpy/_build_utils/gitversion.py'],
    check: true).stdout().strip(),
  license: 'BSD-3',
  meson_version: '>=1.2.99',  # version in vendored-meson is 1.2.99
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
  if not cc.version().version_compare('>=8.4')
    error('NumPy requires GCC >= 8.4')
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

# Intel compilers default to fast-math, so disable it if we detect Intel
# compilers. A word of warning: this may not work with the conda-forge
# compilers, because those have the annoying habit of including lots of flags
# that are gcc-specific in CFLAGS/CXXFLAGS/FFLAGS, which throws off the
# detection logic below. You have to remove the wrong flags (only `-isystem`
# is actually needed, everything else shouldn't be there).
_intel_cflags = []
if cc.get_id() == 'intel'
  _intel_cflags += cc.get_supported_arguments('-fp-model=strict')
elif cc.get_id() == 'intel-cl'
  _intel_cflags += cc.get_supported_arguments('/fp:strict')
elif cc.get_id() == 'intel-llvm-cl'
  _intel_cflags += cc.get_supported_arguments('/fp:strict')
  _intel_cflags += cc.get_supported_arguments('/source-charset:utf-8')
endif
add_project_arguments(_intel_cflags, language: ['c', 'cpp'])
if cc.get_id() == 'intel-llvm-cl'
  _intel_c99_flag = cc.get_supported_arguments('/Qstd=c99')
  add_project_arguments(_intel_c99_flag, language: 'c')
  _intel_c17 = cpp.get_supported_arguments('/Qstd=c++17')
  add_project_arguments(_intel_c17, language: 'cpp')
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
