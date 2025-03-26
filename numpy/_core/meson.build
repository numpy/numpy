# Potential issues to address or keep track of:
#   - sincos detection incorrect on NetBSD: https://github.com/mesonbuild/meson/issues/10641

# Versioning support
#-------------------
#
# How to change C_API_VERSION ?
#   - increase C_API_VERSION value
#   - record the hash for the new C API with the cversions.py script
#   and add the hash to cversions.txt
# The hash values are used to remind developers when the C API number was not
# updated - generates a MismatchCAPIWarning warning which is turned into an
# exception for released version.

# Binary compatibility version number. This number is increased whenever the
# C-API is changed such that binary compatibility is broken, i.e. whenever a
# recompile of extension modules is needed.
# NOTE: This is the version against which an extension module was compiled.
#       As of now compiling against version 2 (0x02000000) yields version 1
#       compatible binaries (subset).  See also NEP 53.
C_ABI_VERSION = '0x02000000'

# Minor API version.  This number is increased whenever a change is made to the
# C-API -- whether it breaks binary compatibility or not.  Some changes, such
# as adding a function pointer to the end of the function table, can be made
# without breaking binary compatibility.  In this case, only the C_API_VERSION
# (*not* C_ABI_VERSION) would be increased.  Whenever binary compatibility is
# broken, both C_API_VERSION and C_ABI_VERSION should be increased.
#
# The version needs to be kept in sync with that in cversions.txt.
#
# 0x00000008 - 1.7.x
# 0x00000009 - 1.8.x
# 0x00000009 - 1.9.x
# 0x0000000a - 1.10.x
# 0x0000000a - 1.11.x
# 0x0000000a - 1.12.x
# 0x0000000b - 1.13.x
# 0x0000000c - 1.14.x
# 0x0000000c - 1.15.x
# 0x0000000d - 1.16.x
# 0x0000000d - 1.19.x
# 0x0000000e - 1.20.x
# 0x0000000e - 1.21.x
# 0x0000000f - 1.22.x
# 0x00000010 - 1.23.x
# 0x00000010 - 1.24.x
# 0x00000011 - 1.25.x
# 0x00000012 - 2.0.x
# 0x00000013 - 2.1.x
# 0x00000013 - 2.2.x
# 0x00000014 - 2.3.x
C_API_VERSION = '0x00000014'

# Check whether we have a mismatch between the set C API VERSION and the
# actual C API VERSION. Will raise a MismatchCAPIError if so.
r = run_command(
  'code_generators/verify_c_api_version.py', '--api-version', C_API_VERSION,
  check: true
)

if r.returncode() != 0
  error('verify_c_api_version.py failed with output:\n' + r.stderr().strip())
endif


# Generate config.h and _numpyconfig.h
# ------------------------------------
#
# There are two generated headers:
#   - config.h: a private header, which is not installed and used only at
#               build time, mostly to determine whether or not optional
#               headers, compiler attributes, etc. are available
#   - _numpyconfig.h: a header with public `NPY_` symbols which get made
#                     available via numpyconfig.h
#
# Note that `HAVE_*` symbols indicate the presence or absence of a checked
# property of the build environment. When available, these symbols get defined
# to `1`; when not available they must not be defined at all. This is
# important, because code in NumPy typically does not check the value but only
# whether the symbol is defined. So `#define HAVE_SOMETHING 0` is wrong.

cdata = configuration_data()

cdata.set('NPY_ABI_VERSION', C_ABI_VERSION)
cdata.set('NPY_API_VERSION', C_API_VERSION)

cpu_family = host_machine.cpu_family()
use_svml = (
  host_machine.system() == 'linux' and
  cpu_family == 'x86_64' and
  ('AVX512_SKX' in CPU_DISPATCH_NAMES or 'AVX512_SKX' in CPU_BASELINE_NAMES) and
  not get_option('disable-svml')
)
if use_svml
  cdata.set10('NPY_CAN_LINK_SVML', true)
  if not fs.exists('src/umath/svml')
    error('Missing the `SVML` git submodule! Run `git submodule update --init` to fix this.')
  endif
endif

if host_machine.cpu_family() == 'loongarch64'
  add_project_arguments(['-DHWY_COMPILE_ONLY_SCALAR'], language: ['cpp'])
endif

use_highway = not get_option('disable-highway')
if use_highway and not fs.exists('src/highway/README.md')
  error('Missing the `highway` git submodule! Run `git submodule update --init` to fix this.')
endif

if use_highway
  highway_lib = static_library('highway',
    [
      # required for hwy::Abort symbol
      'src/highway/hwy/abort.cc'
    ],
    cpp_args: '-DTOOLCHAIN_MISS_ASM_HWCAP_H',
    include_directories: ['src/highway'],
    install: false,
    gnu_symbol_visibility: 'hidden',
  )
else
  highway_lib = []
endif

use_intel_sort = not get_option('disable-intel-sort') and cpu_family in ['x86', 'x86_64']
if use_intel_sort and not fs.exists('src/npysort/x86-simd-sort/README.md')
  error('Missing the `x86-simd-sort` git submodule! Run `git submodule update --init` to fix this.')
endif

if not fs.exists('src/common/pythoncapi-compat')
  error('Missing the `pythoncapi-compat` git submodule! ' +
        'Run `git submodule update --init` to fix this.')
endif

# Check sizes of types. Note, some of these landed in config.h before, but were
# unused. So clean that up and only define the NPY_SIZEOF flavors rather than
# the SIZEOF ones
types_to_check = [
  ['NPY_SIZEOF_SHORT', 'short'],
  ['NPY_SIZEOF_INT', 'int'],
  ['NPY_SIZEOF_LONG', 'long'],
  ['NPY_SIZEOF_LONGLONG', 'long long'],
  ['NPY_SIZEOF_FLOAT', 'float'],
  ['NPY_SIZEOF_DOUBLE', 'double'],
  ['NPY_SIZEOF_LONGDOUBLE', 'long double'],
  ['NPY_SIZEOF_INTP', 'size_t'],
  ['NPY_SIZEOF_UINTP', 'size_t'],
]
foreach symbol_type: types_to_check
  cdata.set(symbol_type[0], cc.sizeof(symbol_type[1]))
endforeach
cdata.set('NPY_SIZEOF_WCHAR_T', cc.sizeof('wchar_t', prefix: '#include <wchar.h>'))
cdata.set('NPY_SIZEOF_OFF_T', cc.sizeof('off_t', prefix: '#include <sys/types.h>'))
cdata.set('NPY_SIZEOF_PY_INTPTR_T',
  cc.sizeof('Py_intptr_t', dependencies: py_dep, prefix: '#include <Python.h>'))
cdata.set('NPY_SIZEOF_PY_LONG_LONG',
  cc.sizeof('PY_LONG_LONG', dependencies: py_dep, prefix: '#include <Python.h>'))

# Check for complex support
if not cc.has_header('complex.h')
  error('"complex.h" header not found')
endif

if cc.get_argument_syntax() == 'msvc'
  complex_types_to_check = [
    ['NPY_SIZEOF_COMPLEX_FLOAT', '_Fcomplex'],
    ['NPY_SIZEOF_COMPLEX_DOUBLE', '_Dcomplex'],
    ['NPY_SIZEOF_COMPLEX_LONGDOUBLE', '_Lcomplex'],
  ]
else
  complex_types_to_check = [
    ['NPY_SIZEOF_COMPLEX_FLOAT', 'complex float'],
    ['NPY_SIZEOF_COMPLEX_DOUBLE', 'complex double'],
    ['NPY_SIZEOF_COMPLEX_LONGDOUBLE', 'complex long double'],
  ]
endif
foreach symbol_type: complex_types_to_check
  if not cc.has_type(symbol_type[1], prefix: '#include <complex.h>')
    t = symbol_type[1]
    error(f'"complex.h" header does not include complex type @t@')
  endif
  cdata.set(symbol_type[0], cc.sizeof(symbol_type[1], prefix: '#include <complex.h>'))
endforeach

# Mandatory functions: if not found, fail the build
# Some of these can still be blocklisted if the C99 implementation
# is buggy, see numpy/_core/src/common/npy_config.h
mandatory_math_funcs = [
  'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs',
  'floor', 'ceil', 'sqrt', 'log10', 'log', 'exp', 'asin',
  'acos', 'atan', 'fmod', 'modf', 'frexp', 'ldexp',
  'expm1', 'log1p', 'acosh', 'asinh', 'atanh',
  'rint', 'trunc', 'exp2', 'copysign', 'nextafter', 'cbrt',
  'log2', 'pow', 'hypot', 'atan2',
]
foreach func: mandatory_math_funcs
  if not cc.has_function(func, prefix: '#include <math.h>', dependencies: m_dep)
    error(f'Function `@func@` not found')
  endif
endforeach

mandatory_complex_math_funcs = [
  'csin', 'csinh', 'ccos', 'ccosh', 'ctan', 'ctanh', 'creal', 'cimag', 'conj'
]
foreach func: mandatory_complex_math_funcs
  if not cc.has_function(func, prefix: '#include <complex.h>', dependencies: m_dep)
    error(f'Function `@func@` not found')
  endif
endforeach

foreach func: ['strtoll', 'strtoull']
  if not cc.has_function(func, prefix: '#include <stdlib.h>')
    error(f'Function `@func@` not found')
  endif
endforeach

c99_complex_funcs = [
  'cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan',
  'catanh', 'cexp', 'clog', 'cpow', 'csqrt',
  # The long double variants (like csinl)  should be mandatory on C11,
  # but are missing in FreeBSD. Issue gh-22850
  'csin', 'csinh', 'ccos', 'ccosh', 'ctan', 'ctanh',
]
foreach func: c99_complex_funcs
  func_single = func + 'f'
  func_longdouble = func + 'l'
  if cc.has_function(func, prefix: '#include <complex.h>', dependencies: m_dep)
    cdata.set10('HAVE_' + func.to_upper(), true)
  endif
  if cc.has_function(func_single, prefix: '#include <complex.h>', dependencies: m_dep)
    cdata.set10('HAVE_' + func_single.to_upper(), true)
  endif
  if cc.has_function(func_longdouble, prefix: '#include <complex.h>', dependencies: m_dep)
    cdata.set10('HAVE_' + func_longdouble.to_upper(), true)
  endif
endforeach

# We require C99 so these should always be found at build time. But for
# libnpymath as a C99 compat layer, these may still be relevant.
c99_macros = ['isfinite', 'isinf', 'isnan', 'signbit']
foreach macro: c99_macros
  if cc.has_function(macro, prefix: '#include <math.h>', dependencies: m_dep)
    cdata.set10('NPY_HAVE_DECL_' + macro.to_upper(), true)
    if not cc.has_header_symbol('Python.h', macro, dependencies: py_dep)
      # Add in config.h as well, except if it will clash with Python's config.h content
      cdata.set10('HAVE_DECL_' + macro.to_upper(), true)
    endif
  endif
endforeach

# variable attributes tested via "int %s a" % attribute
optional_variable_attributes = [
  ['thread_local', 'HAVE_THREAD_LOCAL'],    # C23
  ['_Thread_local', 'HAVE__THREAD_LOCAL'],  # C11/C17
  ['__thread', 'HAVE__THREAD'],
  ['__declspec(thread)', 'HAVE___DECLSPEC_THREAD_']
]
foreach optional_attr: optional_variable_attributes
  attr = optional_attr[0]
  code = f'''
    #pragma GCC diagnostic error "-Wattributes"
    #pragma clang diagnostic error "-Wattributes"

    int @attr@ foo;
  '''
  code += '''
    int
    main()
    {
      return 0;
    }
  '''
  if cc.compiles(code, name: optional_attr[0])
    cdata.set10(optional_attr[1], true)
  endif
endforeach

inc_curdir = include_directories('.')
optional_file_funcs = ['fallocate', 'ftello', 'fseeko']
foreach filefunc_maybe: optional_file_funcs
  config_value = 'HAVE_' + filefunc_maybe.to_upper()
  # Some functions may already have HAVE_* defined by `Python.h`. Python puts
  # its config.h in the public Python.h namespace, so we have a possible clash
  # for the common functions we test. Hence we skip those checks.
  if (filefunc_maybe == 'fallocate' or
      not cc.has_header_symbol('Python.h', config_value, dependencies: py_dep)
    )
    if cc.has_function(filefunc_maybe,
                       include_directories: inc_curdir,
                       prefix: '#include "feature_detection_stdio.h"'
      )
      cdata.set10(config_value, true)
    endif
  endif
endforeach

# Other optional functions
optional_misc_funcs = [
  'backtrace',
  'madvise',
]
foreach func: optional_misc_funcs
  if cc.has_function(func,
      include_directories: inc_curdir,
      prefix: '#include "feature_detection_misc.h"'
    )
    cdata.set10('HAVE_' + func.to_upper(), true)
  endif
endforeach

# SSE headers only enabled automatically on amd64/x32 builds
optional_headers = [
  'features.h',   # for glibc version linux
  'xlocale.h',    # removed in glibc 2.26, but may still be useful - see gh-8367
  'dlfcn.h',      # dladdr
  'execinfo.h',   # backtrace
  'libunwind.h',  # backtrace for LLVM/Clang using libunwind
  'sys/mman.h',   # madvise
]
foreach header: optional_headers
  if cc.has_header(header)
    cdata.set10('HAVE_' + header.to_upper().replace('.', '_').replace('/', '_'), true)
  endif
endforeach

# Optional locale function - GNU-specific
_strtold_prefix = '''
#define _GNU_SOURCE
#include <locale.h>
#include <stdlib.h>
'''
if cdata.get('HAVE_XLOCALE_H', 0) == 1
  _strtold_prefix += '#include <xlocale.h>'
endif
if cc.has_function('strtold_l', include_directories: inc_curdir, prefix: _strtold_prefix)
  cdata.set10('HAVE_STRTOLD_L', true)
endif

# Optional compiler attributes
# TODO: this doesn't work with cc.has_function_attribute, see
#       https://github.com/mesonbuild/meson/issues/10732
optional_function_attributes = [
  ['optimize("unroll-loops")', 'OPTIMIZE_UNROLL_LOOPS'],
  ['optimize("O3")', 'OPTIMIZE_OPT_3'],
  ['optimize("O2")', 'OPTIMIZE_OPT_2'],
  ['optimize("nonnull (1)")', 'NONNULL'],
]
#foreach attr: optional_function_attributes
#  if cc.has_function_attribute(attr[0])
#    cdata.set10('HAVE_ATTRIBUTE_' + attr[1], true)
#  endif
#endforeach

# Max possible optimization flags. We pass this flags to all our dispatch-able
# (multi_targets) sources.
compiler_id = cc.get_id()
max_opt = {
  'msvc': ['/O2'],
  'intel-cl': ['/O3'],
}.get(compiler_id, ['-O3'])
max_opt = cc.has_multi_arguments(max_opt) and get_option('buildtype') != 'debug' ? max_opt : []

# Optional GCC compiler builtins and their call arguments.
# If given, a required header and definition name (HAVE_ prepended)
# Call arguments are required as the compiler will do strict signature checking
optional_intrinsics = [
  ['__builtin_isnan', '5.', [], []],
  ['__builtin_isinf', '5.', [], []],
  ['__builtin_isfinite', '5.', [], []],
  ['__builtin_bswap32', '5u', [], []],
  ['__builtin_bswap64', '5u', [], []],
  ['__builtin_expect', '5, 0', [], []],
  # Test `long long` for arm+clang 13 (gh-22811, but we use all versions):
  ['__builtin_mul_overflow', '(long long)5, 5, (int*)5', [], []],
  ['__builtin_prefetch', '(float*)0, 0, 3', [], []],
]
foreach intrin: optional_intrinsics
  func = intrin[0]
  func_args = intrin[1]
  header = intrin[2]
  define = intrin[3]
  code = ''
  if header.length() == 1
    header_name = header[0]
    code += f'#include <@header_name@>'
  endif
  code += f'''
    #ifdef _MSC_VER
    #pragma function(@func@)
    #endif
    int main(void) {
      @func@(@func_args@);
      return 0;
    };
    '''
  if define.length() == 1
    define_name = define[0]
  else
    define_name = 'HAVE_' + func.to_upper()
  endif
  if cc.links(code)
    cdata.set10(define_name, true)
  endif
endforeach

# This is a port of the old python code for identifying the long double
# representation to C.  The old Python code is in this range:
# https://github.com/numpy/numpy/blob/eead09a3d02c09374942cdc787c0b5e4fe9e7472/numpy/core/setup_common.py#L264-L434
# This port is in service of solving gh-23972
# as well as https://github.com/mesonbuild/meson/issues/11068
longdouble_format = meson.get_external_property('longdouble_format', 'UNKNOWN')
if longdouble_format == 'UNKNOWN'
  longdouble_format = meson.get_compiler('c').run(
'''
#include <stdio.h>
#include <string.h>

#define repcmp(z) (memcmp((const char *)&foo.x, z, sizeof(foo.x)) == 0)

const struct {
  char before[16];
  long double x;
  char after[8];
} foo = {{'\0'}, -123456789.0, {'\0'}};

int main(void) {
  switch (sizeof(foo.x)) {
  case 8: {
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0124, 0064, 0157, 0235, 0301}))) {
      fprintf(stdout, "IEEE_DOUBLE_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0301, 0235, 0157, 0064, 0124, 0000, 0000, 0000}))) {
      fprintf(stdout, "IEEE_DOUBLE_BE");
      return 0;
    }
    fprintf(stdout, "UNKNOWN");
    return 1;
  }
  case 12: {
    if (repcmp(((const char[]){0000, 0000, 0000, 0000, 0240, 0242, 0171, 0353,
                               0031, 0300, 0000, 0000}))) {
      fprintf(stdout, "INTEL_EXTENDED_12_BYTES_LE");
      return 0;
    }
    if (repcmp(((const char[]){0300, 0031, 0000, 0000, 0353, 0171, 0242, 0240,
                               0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "MOTOROLA_EXTENDED_12_BYTES_BE");
      return 0;
    }
    fprintf(stdout, "UNKNOWN");
    return 1;
  }
  case 16: {
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0000, 0240, 0242, 0171, 0353,
                            0031, 0300, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "INTEL_EXTENDED_16_BYTES_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0300, 0031, 0326, 0363, 0105, 0100, 0000, 0000,
                            0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "IEEE_QUAD_BE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000,
                            0000, 0000, 0100, 0105, 0363, 0326, 0031, 0300}))) {
      fprintf(stdout, "IEEE_QUAD_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0124, 0064, 0157, 0235, 0301,
                            0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "IBM_DOUBLE_DOUBLE_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0301, 0235, 0157, 0064, 0124, 0000, 0000, 0000,
                            0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "IBM_DOUBLE_DOUBLE_BE");
      return 0;
    }
    fprintf(stdout, "UNKNOWN");
    return 1;
  }
  }
}
  ''').stdout()
endif
if longdouble_format == 'UNKNOWN' or longdouble_format == 'UNDEFINED'
  error('Unknown long double format of size: ' + cc.sizeof('long double').to_string())
endif
cdata.set10('HAVE_LDOUBLE_' + longdouble_format, true)

if cc.has_header('endian.h')
  cdata.set10('NPY_HAVE_ENDIAN_H', true)
endif
if cc.has_header('sys/endian.h')
  cdata.set10('NPY_HAVE_SYS_ENDIAN_H', true)
endif
# Build-time option to disable threading is stored and exposed in numpyconfig.h
# Note: SMP is an old acronym for threading (Symmetric/Shared-memory MultiProcessing)
cdata.set10('NPY_NO_SMP', get_option('disable-threading'))

visibility_hidden = ''
if cc.has_function_attribute('visibility:hidden') and host_machine.system() != 'cygwin'
  visibility_hidden = '__attribute__((visibility("hidden")))'
endif
cdata.set('NPY_VISIBILITY_HIDDEN', visibility_hidden)

# if not set, we're using lapack_lite
if have_lapack
  cdata.set10('HAVE_EXTERNAL_LAPACK', have_lapack)
endif

config_h = configure_file(
  input: 'config.h.in',
  output: 'config.h',
  configuration: cdata,
  install: false
)

_numpyconfig_h = configure_file(
  input: 'include/numpy/_numpyconfig.h.in',
  output: '_numpyconfig.h',
  configuration: cdata,
  install: true,
  install_dir: np_dir / '_core/include/numpy',
  install_tag: 'devel'
)

# Build npymath static library
# ----------------------------

staticlib_cflags = []
staticlib_cppflags = []
if cc.get_id() == 'msvc'
  # Disable voltbl section for vc142 to allow link using mingw-w64; see:
  # https://github.com/matthew-brett/dll_investigation/issues/1#issuecomment-1100468171
  # Needs to be added to static libraries that are shipped for reuse (i.e.,
  # libnpymath and libnpyrandom)
  if cc.has_argument('-d2VolatileMetadata-')
     staticlib_cflags +=  '-d2VolatileMetadata-'
  endif
endif

npy_math_internal_h = custom_target(
  output: 'npy_math_internal.h',
  input: 'src/npymath/npy_math_internal.h.src',
  command: [src_file_cli, '@INPUT@', '-o', '@OUTPUT@'],
)

npymath_sources = [
  src_file.process('src/npymath/ieee754.c.src'),
  src_file.process('src/npymath/npy_math_complex.c.src'),
  npy_math_internal_h,
  'src/npymath/halffloat.cpp',
  'src/npymath/npy_math.c',
]
npymath_lib = static_library('npymath',
  npymath_sources,
  c_args: staticlib_cflags,
  cpp_args: staticlib_cppflags,
  include_directories: ['include', 'src/npymath', 'src/common'],
  dependencies: py_dep,
  install: true,
  install_dir: np_dir / '_core/lib',
  name_prefix: name_prefix_staticlib,
  name_suffix: name_suffix_staticlib,
  gnu_symbol_visibility: 'hidden',
)

dir_separator = '/'
if build_machine.system() == 'windows'
  dir_separator = '\\'
endif
configure_file(
  input: 'npymath.ini.in',
  output: 'npymath.ini',
  configuration: configuration_data({
    'pkgname' : 'numpy._core',
    'sep' : dir_separator,
  }),
  install: true,
  install_dir: np_dir / '_core/lib/npy-pkg-config',
  install_tag: 'devel'
)
configure_file(
  input: 'mlib.ini.in',
  output: 'mlib.ini',
  configuration: configuration_data({
    'posix_mathlib' : mlib_linkflag,
    'msvc_mathlib' : 'm.lib',
  }),
  install: true,
  install_dir: np_dir / '_core/lib/npy-pkg-config',
  install_tag: 'devel'
)

if false
  # This doesn't quite work (yet), it assumes we'll install headers under
  # include/, and trying to add the correct path with `extra_cflags` runs into
  # not being able to get a path relative to the prefix.
  # Note that installing numpy headers under include/ would be a good idea, but
  # that needs work (and may run into trouble with wheels perhaps, not quite
  # clear if they allow this).
  pkg = import('pkgconfig')
  pkg.generate(npymath_lib,
    name: 'npymath',
    description: 'Portable, core math library implementing C99 standard',
    url: 'https://github.com/numpy/numpy',
    requires: 'numpy',
    install_dir: np_dir / '_core/lib/npy-pkg-config',
    extra_cflags: '-I${includedir}' + np_dir / '_core' / 'include',
  )
endif

# Generate NumPy C API sources
# ----------------------------

# This is a single C file. It needs to be built before _multiarray_umath starts
# building, but we can't add it to the sources of that extension (this C file
# doesn't compile, it's only included in another r file. Hence use the slightly
# hacky --ignore argument to the next custom_target().
src_umath_api_c = custom_target('__umath_generated',
  output : '__umath_generated.c',
  input : 'code_generators/generate_umath.py',
  command: [py, '@INPUT@', '-o', '@OUTPUT@'],
)

src_umath_doc_h = custom_target('_umath_doc_generated',
  output : '_umath_doc_generated.h',
  input : ['code_generators/generate_umath_doc.py',
           'code_generators/ufunc_docstrings.py'],
  command: [py, '@INPUT0@', '-o', '@OUTPUT@'],
)

src_numpy_api = custom_target('__multiarray_api',
  output : ['__multiarray_api.c', '__multiarray_api.h'],
  input : 'code_generators/generate_numpy_api.py',
  command: [py, '@INPUT@', '-o', '@OUTDIR@', '--ignore', src_umath_api_c],
  install: true,  # NOTE: setup.py build installs all, but just need .h?
  install_dir: np_dir / '_core/include/numpy',
  install_tag: 'devel'
)

src_ufunc_api = custom_target('__ufunc_api',
  output : ['__ufunc_api.c', '__ufunc_api.h'],
  input : 'code_generators/generate_ufunc_api.py',
  command: [py, '@INPUT@', '-o', '@OUTDIR@'],
  install: true,  # NOTE: setup.py build installs all, but just need .h?
  install_dir: np_dir / '_core/include/numpy',
  install_tag: 'devel'
)

# Write out pkg-config file
# -------------------------

# Note: we can't use Meson's built-in pkgconfig module, because we have to
# install numpy.pc within site-packages rather than in its normal location.
cdata_numpy_pc = configuration_data()
cdata_numpy_pc.set('version', meson.project_version())

# Note: keep install path in sync with numpy/_configtool.py
_numpy_pc = configure_file(
  input: 'numpy.pc.in',
  output: 'numpy.pc',
  configuration: cdata_numpy_pc,
  install: true,
  install_dir: np_dir / '_core/lib/pkgconfig',
  install_tag: 'devel'
)

# Set common build flags for C and C++ code
# -----------------------------------------
# Common build flags
c_args_common = [
  '-DNPY_INTERNAL_BUILD',
  '-DHAVE_NPY_CONFIG_H',
  cflags_large_file_support,
]

# CPP exceptions are handled in the unique_hash code and therefore the `-fexceptions`
# flag.
unique_hash_cpp_args = c_args_common
if cc.get_argument_syntax() != 'msvc'
  unique_hash_cpp_args += [
    '-fexceptions',
    '-fno-rtti',  # no runtime type information
  ]
endif

# Same as NPY_CXX_FLAGS (TODO: extend for what ccompiler_opt adds)
cpp_args_common = c_args_common + [
]
if cc.get_argument_syntax() != 'msvc'
  cpp_args_common += [
    '-fno-exceptions',  # no exception support
    '-fno-rtti',  # no runtime type information
  ]
endif

# Other submodules depend on generated headers and include directories from
# core, wrap those up into a reusable dependency. Also useful for some test
# modules in this build file.
np_core_dep = declare_dependency(
  sources: [
    _numpyconfig_h,
    npy_math_internal_h,
    src_numpy_api[1],  # __multiarray_api.h
    src_ufunc_api[1],  # __ufunc_api.h
  ],
  include_directories: [
    '.',
    'include',
    'src/common',
  ]
)

# Build multiarray_tests module
# -----------------------------
py.extension_module('_multiarray_tests',
  [
    src_file.process('src/multiarray/_multiarray_tests.c.src'),
    'src/common/mem_overlap.c',
    'src/common/npy_argparse.c',
    'src/common/npy_hashtable.cpp',
    src_file.process('src/common/templ_common.h.src')
  ],
  c_args: c_args_common,
  include_directories: ['src/multiarray', 'src/npymath'],
  dependencies: np_core_dep,
  link_with: npymath_lib,
  gnu_symbol_visibility: 'default',
  install: true,
  subdir: 'numpy/_core',
  install_tag: 'tests'
)

_umath_tests_mtargets = mod_features.multi_targets(
  '_umath_tests.dispatch.h',
  'src/umath/_umath_tests.dispatch.c',
  dispatch: [
    AVX2, SSE41, SSE2,
    ASIMDHP, ASIMD, NEON,
    VSX3, VSX2, VSX,
    VXE, VX,
  ],
  baseline: CPU_BASELINE,
  prefix: 'NPY_',
  dependencies: [py_dep, np_core_dep]
)

test_modules_src = [
  ['_umath_tests', [
      src_file.process('src/umath/_umath_tests.c.src'),
      'src/common/npy_cpu_features.c',
    ],
    _umath_tests_mtargets.static_lib('_umath_tests_mtargets')
  ],
  ['_rational_tests', 'src/umath/_rational_tests.c', []],
  ['_struct_ufunc_tests', 'src/umath/_struct_ufunc_tests.c', []],
  ['_operand_flag_tests', 'src/umath/_operand_flag_tests.c', []],
]
foreach gen: test_modules_src
  py.extension_module(gen[0],
    gen[1],
    c_args: c_args_common,
    include_directories: ['src/multiarray', 'src/npymath'],
    dependencies: np_core_dep,
    install: true,
    subdir: 'numpy/_core',
    link_with: gen[2],
    install_tag: 'tests'
  )
endforeach

# Build multiarray dispatch-able sources
# --------------------------------------
multiarray_gen_headers = [
  src_file.process('src/multiarray/arraytypes.h.src'),
  src_file.process('src/common/npy_sort.h.src'),
]
foreach gen_mtargets : [
  [
    'argfunc.dispatch.h',
    src_file.process('src/multiarray/argfunc.dispatch.c.src'),
    [
      AVX512_SKX, AVX2, XOP, SSE42, SSE2,
      VSX2,
      ASIMD, NEON,
      VXE, VX
    ]
  ],
]
  mtargets = mod_features.multi_targets(
    gen_mtargets[0], multiarray_gen_headers + gen_mtargets[1],
    dispatch: gen_mtargets[2],
    baseline: CPU_BASELINE,
    prefix: 'NPY_',
    dependencies: [py_dep, np_core_dep],
    c_args: c_args_common + max_opt,
    cpp_args: cpp_args_common + max_opt,
    include_directories: [
      'include',
      'src/common',
      'src/multiarray',
      'src/multiarray/stringdtype',
      'src/npymath',
      'src/umath'
    ]
  )
  if not is_variable('multiarray_umath_mtargets')
    multiarray_umath_mtargets = mtargets
  else
    multiarray_umath_mtargets.extend(mtargets)
  endif
endforeach

# Build npysort dispatch-able sources
# -----------------------------------
foreach gen_mtargets : [
  [
    'x86_simd_argsort.dispatch.h',
    'src/npysort/x86_simd_argsort.dispatch.cpp',
    use_intel_sort ? [AVX512_SKX, AVX2] : []
  ],
  [
    'x86_simd_qsort.dispatch.h',
    'src/npysort/x86_simd_qsort.dispatch.cpp',
    use_intel_sort ? [AVX512_SKX, AVX2] : []
  ],
  [
    'x86_simd_qsort_16bit.dispatch.h',
    'src/npysort/x86_simd_qsort_16bit.dispatch.cpp',
    use_intel_sort ? [AVX512_SPR, AVX512_ICL] : []
  ],
  [
    'highway_qsort.dispatch.h',
    'src/npysort/highway_qsort.dispatch.cpp',
    use_highway ? [
      ASIMD, VSX2,  # FIXME: disable VXE due to runtime segfault
    ] : []
  ],
  [
    'highway_qsort_16bit.dispatch.h',
    'src/npysort/highway_qsort_16bit.dispatch.cpp',
    use_highway ? [
      ASIMDHP, VSX2,  # VXE FIXME: disable VXE due to runtime segfault
    ] : []
  ],
]
  mtargets = mod_features.multi_targets(
    gen_mtargets[0], multiarray_gen_headers + gen_mtargets[1],
    dispatch: gen_mtargets[2],
    # baseline: CPU_BASELINE, it doesn't provide baseline fallback
    prefix: 'NPY_',
    dependencies: [py_dep, np_core_dep],
    c_args: c_args_common + max_opt,
    cpp_args: cpp_args_common + max_opt,
    include_directories: [
      'include',
      'src/common',
      'src/multiarray',
      'src/npymath',
      'src/umath',
      'src/highway',
    ]
  )
  if not is_variable('multiarray_umath_mtargets')
    multiarray_umath_mtargets = mtargets
  else
    multiarray_umath_mtargets.extend(mtargets)
  endif
endforeach

# Build umath dispatch-able sources
# ---------------------------------
mod_features = import('features')
umath_gen_headers = [
  src_file.process('src/umath/loops.h.src'),
  src_file.process('src/umath/loops_utils.h.src'),
]

foreach gen_mtargets : [
  [
    'loops_arithm_fp.dispatch.h',
    src_file.process('src/umath/loops_arithm_fp.dispatch.c.src'),
    [
      [AVX2, FMA3], SSE2,
      ASIMD, NEON,
      VSX3, VSX2,
      VXE, VX,
      LSX,
    ]
  ],
  [
    'loops_arithmetic.dispatch.h',
    src_file.process('src/umath/loops_arithmetic.dispatch.c.src'),
    [
      AVX512_SKX, AVX512F, AVX2, SSE41, SSE2,
      NEON,
      VSX4, VSX2,
      VX,
      LSX,
    ]
  ],
  [
    'loops_comparison.dispatch.h',
    src_file.process('src/umath/loops_comparison.dispatch.c.src'),
    [
      AVX512_SKX, AVX512F, AVX2, SSE42, SSE2,
      VSX3, VSX2,
      NEON,
      VXE, VX,
      LSX,
    ]
  ],
  [
    'loops_exponent_log.dispatch.h',
    src_file.process('src/umath/loops_exponent_log.dispatch.c.src'),
    [
      AVX512_SKX, AVX512F, [AVX2, FMA3]
    ]
  ],
  [
    'loops_hyperbolic.dispatch.h',
    src_file.process('src/umath/loops_hyperbolic.dispatch.cpp.src'),
    [
      AVX512_SKX, [AVX2, FMA3],
      VSX4, VSX2,
      NEON_VFPV4,
      VXE,
      LSX,
    ]
  ],
  [
    'loops_logical.dispatch.h',
    'src/umath/loops_logical.dispatch.cpp',
    [
      ASIMD, NEON,
      AVX512_SKX, AVX2, SSE2,
      VSX2,
      VX,
      LSX,
      RVV,
    ]
  ],
  [
    'loops_minmax.dispatch.h',
    src_file.process('src/umath/loops_minmax.dispatch.c.src'),
    [
      ASIMD, NEON,
      AVX512_SKX, AVX2, SSE2,
      VSX2,
      VXE, VX,
      LSX,
    ]
  ],
  [
    'loops_modulo.dispatch.h',
    src_file.process('src/umath/loops_modulo.dispatch.c.src'),
    [
      VSX4
    ]
  ],
  [
    'loops_trigonometric.dispatch.h',
    'src/umath/loops_trigonometric.dispatch.cpp',
    [
      AVX512_SKX, [AVX2, FMA3],
      VSX4, VSX3, VSX2,
      NEON_VFPV4,
      VXE2, VXE,
      LSX,
    ]
  ],
  [
    'loops_umath_fp.dispatch.h',
    src_file.process('src/umath/loops_umath_fp.dispatch.c.src'),
    [AVX512_SKX]
  ],
  [
    'loops_unary.dispatch.h',
    src_file.process('src/umath/loops_unary.dispatch.c.src'),
    [
      ASIMD, NEON,
      AVX512_SKX, AVX2, SSE2,
      VSX2,
      VXE, VX,
      LSX,
    ]
  ],
  [
    'loops_unary_fp.dispatch.h',
    src_file.process('src/umath/loops_unary_fp.dispatch.c.src'),
    [
      SSE41, SSE2,
      VSX2,
      ASIMD, NEON,
      VXE, VX,
      LSX,
    ]
  ],
  [
    'loops_unary_fp_le.dispatch.h',
    src_file.process('src/umath/loops_unary_fp_le.dispatch.c.src'),
    [
      SSE41, SSE2,
      VSX2,
      ASIMD, NEON,
      LSX,
    ]
  ],
  [
    'loops_unary_complex.dispatch.h',
    src_file.process('src/umath/loops_unary_complex.dispatch.c.src'),
    [
      AVX512F, [AVX2, FMA3], SSE2,
      ASIMD, NEON,
      VSX3, VSX2,
      VXE, VX,
      LSX,
    ]
  ],
  [
    'loops_autovec.dispatch.h',
    src_file.process('src/umath/loops_autovec.dispatch.c.src'),
    [
      AVX2, SSE2,
      NEON,
      VSX2,
      VX,
      LSX,
    ]
  ],
]
  mtargets = mod_features.multi_targets(
    gen_mtargets[0], umath_gen_headers + gen_mtargets[1],
    dispatch: gen_mtargets[2],
    baseline: CPU_BASELINE,
    prefix: 'NPY_',
    dependencies: [py_dep, np_core_dep],
    c_args: c_args_common + max_opt,
    cpp_args: cpp_args_common + max_opt,
    include_directories: [
      'include',
      'src/common',
      'src/multiarray',
      'src/npymath',
      'src/umath',
      'src/highway',
    ]
  )
  if not is_variable('multiarray_umath_mtargets')
    multiarray_umath_mtargets = mtargets
  else
    multiarray_umath_mtargets.extend(mtargets)
  endif
endforeach

# Build _multiarray_umath module
# ------------------------------
src_multiarray_umath_common = [
  'src/common/array_assign.c',
  'src/common/gil_utils.c',
  'src/common/mem_overlap.c',
  'src/common/npy_argparse.c',
  'src/common/npy_hashtable.cpp',
  'src/common/npy_import.c',
  'src/common/npy_longdouble.c',
  'src/common/ufunc_override.c',
  'src/common/numpyos.c',
  'src/common/npy_cpu_features.c',
  'src/common/npy_cpu_dispatch.c',
  src_file.process('src/common/templ_common.h.src')
]
if have_blas
  src_multiarray_umath_common += [
    'src/common/cblasfuncs.c',
    'src/common/python_xerbla.c',
  ]
endif

src_multiarray = multiarray_gen_headers + [
  'src/multiarray/abstractdtypes.c',
  'src/multiarray/alloc.c',
  'src/multiarray/arrayobject.c',
  'src/multiarray/array_coercion.c',
  'src/multiarray/array_converter.c',
  'src/multiarray/array_method.c',
  'src/multiarray/array_api_standard.c',
  'src/multiarray/array_assign_scalar.c',
  'src/multiarray/array_assign_array.c',
  'src/multiarray/arrayfunction_override.c',
  src_file.process('src/multiarray/arraytypes.c.src'),
  'src/multiarray/arraywrap.c',
  'src/multiarray/buffer.c',
  'src/multiarray/calculation.c',
  'src/multiarray/compiled_base.c',
  'src/multiarray/common.c',
  'src/multiarray/common_dtype.c',
  'src/multiarray/convert.c',
  'src/multiarray/convert_datatype.c',
  'src/multiarray/conversion_utils.c',
  'src/multiarray/ctors.c',
  'src/multiarray/datetime.c',
  'src/multiarray/datetime_strings.c',
  'src/multiarray/datetime_busday.c',
  'src/multiarray/datetime_busdaycal.c',
  'src/multiarray/descriptor.c',
  'src/multiarray/dlpack.c',
  'src/multiarray/dtypemeta.c',
  'src/multiarray/dragon4.c',
  'src/multiarray/dtype_transfer.c',
  'src/multiarray/dtype_traversal.c',
  src_file.process('src/multiarray/einsum.c.src'),
  src_file.process('src/multiarray/einsum_sumprod.c.src'),
  'src/multiarray/public_dtype_api.c',
  'src/multiarray/flagsobject.c',
  'src/multiarray/getset.c',
  'src/multiarray/hashdescr.c',
  'src/multiarray/item_selection.c',
  'src/multiarray/iterators.c',
  'src/multiarray/legacy_dtype_implementation.c',
  src_file.process('src/multiarray/lowlevel_strided_loops.c.src'),
  'src/multiarray/mapping.c',
  'src/multiarray/methods.c',
  'src/multiarray/multiarraymodule.c',
  'src/multiarray/nditer_api.c',
  'src/multiarray/nditer_constr.c',
  'src/multiarray/nditer_pywrap.c',
  src_file.process('src/multiarray/nditer_templ.c.src'),
  'src/multiarray/npy_static_data.c',
  'src/multiarray/number.c',
  'src/multiarray/refcount.c',
  src_file.process('src/multiarray/scalartypes.c.src'),
  'src/multiarray/sequence.c',
  'src/multiarray/scalarapi.c',
  'src/multiarray/shape.c',
  'src/multiarray/strfuncs.c',
  'src/multiarray/stringdtype/casts.cpp',
  'src/multiarray/stringdtype/dtype.c',
  'src/multiarray/stringdtype/utf8_utils.c',
  'src/multiarray/stringdtype/static_string.c',
  'src/multiarray/temp_elide.c',
  'src/multiarray/usertypes.c',
  'src/multiarray/vdot.c',
  'src/npysort/quicksort.cpp',
  'src/npysort/mergesort.cpp',
  'src/npysort/timsort.cpp',
  'src/npysort/heapsort.cpp',
  'src/npysort/radixsort.cpp',
  'src/common/npy_partition.h',
  'src/npysort/selection.cpp',
  'src/common/npy_binsearch.h',
  'src/npysort/binsearch.cpp',
  'src/multiarray/textreading/conversions.c',
  'src/multiarray/textreading/field_types.c',
  'src/multiarray/textreading/growth.c',
  'src/multiarray/textreading/readtext.c',
  'src/multiarray/textreading/rows.c',
  'src/multiarray/textreading/stream_pyobject.c',
  'src/multiarray/textreading/str_to_int.c',
  'src/multiarray/textreading/tokenize.cpp',
  # Remove this `arm64_exports.c` file once scipy macos arm64 build correctly
  # links to the arm64 npymath library, see gh-22673
  'src/npymath/arm64_exports.c',
]

src_umath = umath_gen_headers + [
  src_file.process('src/umath/funcs.inc.src'),
  src_file.process('src/umath/loops.c.src'),
  src_file.process('src/umath/matmul.c.src'),
  src_file.process('src/umath/matmul.h.src'),
  'src/umath/ufunc_type_resolution.c',
  'src/umath/clip.cpp',
  'src/umath/clip.h',
  'src/umath/dispatching.cpp',
  'src/umath/extobj.c',
  'src/umath/legacy_array_method.c',
  'src/umath/override.c',
  'src/umath/reduction.c',
  src_file.process('src/umath/scalarmath.c.src'),
  'src/umath/ufunc_object.c',
  'src/umath/umathmodule.c',
  'src/umath/special_integer_comparisons.cpp',
  'src/umath/string_ufuncs.cpp',
  'src/umath/stringdtype_ufuncs.cpp',
  'src/umath/wrapping_array_method.c',
  # For testing. Eventually, should use public API and be separate:
  'src/umath/_scaled_float_dtype.c',
]

# SVML object files. If functionality is migrated to universal intrinsics and
# the object files are no longer needed, comment out the relevant object files
# here. Note that this migration is desirable; we then get the performance
# benefits for all platforms rather than only for AVX512 on 64-bit Linux, and
# may be able to avoid the accuracy regressions in SVML.
#
CPU_FEATURES_NAMES = CPU_BASELINE_NAMES + CPU_DISPATCH_NAMES
svml_file_suffix = ['d_la', 's_la', 'd_ha', 's_la']
if CPU_FEATURES_NAMES.contains('AVX512_SPR')
  svml_file_suffix += ['h_la']
endif

svml_objects = []
if use_svml
  foreach svml_func : [
    'acos', 'acosh', 'asin',
    'asinh', 'atan2',
    'atan',  'atanh',
    'cbrt', 'cos',
    'cosh', 'exp2',
    'exp', 'expm1',
    'log10', 'log1p',
    'log2', 'log',
    'pow', 'sin', 'sinh', 'tan',
    'tanh'
  ]
    foreach svml_sfx : svml_file_suffix
      svml_objects += [
        'src/umath/svml/linux/avx512/svml_z0_'+svml_func+'_'+svml_sfx+'.s'
      ]
    endforeach
  endforeach
endif

unique_hash_so = static_library(
  'unique_hash',
  ['src/multiarray/unique.cpp'],
  c_args: c_args_common,
  cpp_args: unique_hash_cpp_args,
  include_directories: [
    'include',
    'src/common',
  ],
  dependencies: [
    py_dep,
    np_core_dep,
  ],
)

py.extension_module('_multiarray_umath',
  [
    config_h,
    _numpyconfig_h,
    src_multiarray,
    src_multiarray_umath_common,
    src_umath,
    src_ufunc_api[1],  # __ufunc_api.h
    src_numpy_api[1],  # __multiarray_api.h
    src_umath_doc_h,
    npy_math_internal_h,
  ],
  objects: svml_objects,
  c_args: c_args_common,
  cpp_args: cpp_args_common,
  include_directories: [
    'include',
    'src/common',
    'src/multiarray',
    'src/npymath',
    'src/umath',
    'src/highway'
  ],
  dependencies: [blas_dep],
  link_with: [
    npymath_lib,
    unique_hash_so,
    multiarray_umath_mtargets.static_lib('_multiarray_umath_mtargets')
  ] + highway_lib,
  install: true,
  subdir: 'numpy/_core',
)

# Build SIMD module
# -----------------
_simd_dispatch = []
_simd_baseline = []
foreach target : get_option('test-simd')
  target = target.strip().to_upper().split(',')
  mfeatures = []
  foreach fet_name : target
    if fet_name == 'BASELINE'
      _simd_baseline = CPU_BASELINE
      break
    endif
    if fet_name not in CPU_FEATURES
      error('Expected a valid feature name, got('+fet_name+')')
    endif
    mfeatures += CPU_FEATURES[fet_name]
  endforeach
  _simd_dispatch += [mfeatures]
endforeach

_simd_mtargets = mod_features.multi_targets(
  '_simd.dispatch.h',
  [
    src_file.process('src/_simd/_simd_inc.h.src'),
    src_file.process('src/_simd/_simd_data.inc.src'),
    src_file.process('src/_simd/_simd.dispatch.c.src'),
  ],
  # Skip validating the order of `_simd_dispatch` because we execute all these
  # features, not just the highest interest one. The sorting doesn't matter
  # here, given the nature of this testing unit.
  keep_sort: true,
  dispatch: _simd_dispatch,
  baseline: _simd_baseline,
  prefix: 'NPY_',
  dependencies: [py_dep, np_core_dep],
  include_directories: ['src/_simd', 'src/npymath'],
  c_args: c_args_common,
  cpp_args: cpp_args_common,
)

py.extension_module('_simd',
  [
    'src/common/npy_cpu_features.c',
    'src/_simd/_simd.c',
  ],
  c_args: c_args_common,
  include_directories: ['src/_simd', 'src/npymath'],
  dependencies: np_core_dep,
  link_with: [npymath_lib, _simd_mtargets.static_lib('_simd_mtargets')],
  install: true,
  subdir: 'numpy/_core',
  install_tag: 'tests',
)

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_add_newdocs.py',
  '_add_newdocs.pyi',
  '_add_newdocs_scalars.py',
  '_add_newdocs_scalars.pyi',
  '_asarray.py',
  '_asarray.pyi',
  '_dtype.py',
  '_dtype.pyi',
  '_dtype_ctypes.py',
  '_dtype_ctypes.pyi',
  '_exceptions.py',
  '_exceptions.pyi',
  '_internal.py',
  '_internal.pyi',
  '_machar.py',
  '_machar.pyi',
  '_methods.py',
  '_methods.pyi',
  '_simd.pyi',
  '_string_helpers.py',
  '_string_helpers.pyi',
  '_type_aliases.py',
  '_type_aliases.pyi',
  '_ufunc_config.py',
  '_ufunc_config.pyi',
  'arrayprint.py',
  'arrayprint.pyi',
  'cversions.py',
  'defchararray.py',
  'defchararray.pyi',
  'einsumfunc.py',
  'einsumfunc.pyi',
  'fromnumeric.py',
  'fromnumeric.pyi',
  'function_base.py',
  'function_base.pyi',
  'getlimits.py',
  'getlimits.pyi',
  'memmap.py',
  'memmap.pyi',
  'multiarray.py',
  'multiarray.pyi',
  'numeric.py',
  'numeric.pyi',
  'numerictypes.py',
  'numerictypes.pyi',
  'overrides.py',
  'overrides.pyi',
  'printoptions.py',
  'printoptions.pyi',
  'records.py',
  'records.pyi',
  'shape_base.py',
  'shape_base.pyi',
  'strings.py',
  'strings.pyi',
  'umath.py',
  'umath.pyi',
]

py.install_sources(
  python_sources,
  subdir: 'numpy/_core'
)

subdir('include')
install_subdir('tests', install_dir: np_dir / '_core', install_tag: 'tests')
