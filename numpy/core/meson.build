# This file should contain what setup.py + setup_common.py do (WIP)
#
# Potential issues to address or keep track of:
#   - sincos detection incorrect on NetBSD: https://github.com/mesonbuild/meson/issues/10641

# Generate config.h
# -----------------

config_h = configuration_data()

if not cc.has_header('Python.h', dependencies: py_dep)
  error('Cannot compile `Python.h`. Perhaps you need to install python-dev|python-devel')
endif

fs = import ('fs')
use_svml = (
  host_machine.system() == 'linux' and
  host_machine.cpu_family() == 'x86_64' and
  not get_option('disable-svml')
)
if use_svml
  config_h.set10('NPY_CAN_LINK_SVML', true)
  if not fs.exists('src/umath/svml')
    error('Missing the `SVML` git submodule! Run `git submodule update --init` to fix this.')
  endif
endif

# Check sizes of types
config_h.set('SIZEOF_PY_INTPTR_T',
  cc.sizeof('Py_intptr_t', dependencies: py_dep, prefix: '#include <Python.h>'))
config_h.set('SIZEOF_OFF_T',
  cc.sizeof('off_t', prefix: '#include <sys/types.h>'))
config_h.set('SIZEOF_PY_LONG_LONG',
  cc.sizeof('PY_LONG_LONG', dependencies: py_dep, prefix: '#include <Python.h>'))

# Check for complex support
config_h.set10('HAVE_COMPLEX_H', cc.has_header('complex.h'))
# TODO: NPY_USE_C99_COMPLEX and on in check_complex()

# Mandatory functions: if not found, fail the build
# Some of these can still be blocklisted if the C99 implementation
# is buggy, see numpy/core/src/common/npy_config.h
mandatory_funcs = [
  'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs',
  'floor', 'ceil', 'sqrt', 'log10', 'log', 'exp', 'asin',
  'acos', 'atan', 'fmod', 'modf', 'frexp', 'ldexp',
  'expm1', 'log1p', 'acosh', 'asinh', 'atanh',
  'rint', 'trunc', 'exp2', 
  'copysign', 'nextafter', 'strtoll', 'strtoull', 'cbrt',
  'log2', 'pow', 'hypot', 'atan2',
  'csin', 'csinh', 'ccos', 'ccosh', 'ctan', 'ctanh',
  'creal', 'cimag', 'conj'
]
foreach func: mandatory_funcs
  if not cc.has_function(func)
    error('Function `{func}` not found')
  endif
endforeach

C99_COMPLEX_TYPES = [
  'complex double', 'complex float', 'complex long double'
]
c99_complex_funcs = [
  'cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan',
  'catanh', 'cexp', 'clog', 'cpow', 'csqrt'
]
foreach func: c99_complex_funcs
  func_single = func + 'f'
  func_longdouble = func + 'l'
  config_h.set10('HAVE_' + func.to_upper(), cc.has_function(func))
  config_h.set10('HAVE_' + func_single.to_upper(), cc.has_function(func_single))
  config_h.set10('HAVE_' + func_longdouble.to_upper(), cc.has_function(func_longdouble))
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
      config_h.set10(config_value,
        cc.has_function(filefunc_maybe,
          include_directories: inc_curdir,
          prefix: '#include "feature_detection_stdio.h"'
        )
      )
  endif
endforeach

# Optional locale function
result = cc.has_function('strtold_l', include_directories: inc_curdir,
  prefix:'''
  #include <stdlib.h>
  #include <xlocale.h>
  #include "feature_detection_locale.h"
''')
if not result
  # Retry with locale.h, seems to vary across Linux distros
  result = cc.has_function('strtold_l', include_directories: inc_curdir,
    prefix:'''
    #include <stdlib.h>
    #include <locale.h>
    #include "feature_detection_locale.h"
  ''')
endif
config_h.set10('HAVE_STRTOLD_L', result)

# Other optional functions
config_h.set10('HAVE_BACKTRACE',
  cc.has_function('backtrace', prefix: '#include <execinfo.h>'))
config_h.set10('HAVE_MADVISE',
  cc.has_function('madvise', prefix: '#include <sys/mman.h>'))

# SSE headers only enabled automatically on amd64/x32 builds
optional_headers = [
  'xmmintrin.h',  # SSE
  'emmintrin.h',  # SSE2
  'immintrin.h',  # AVX
  'features.h',   # for glibc version linux
  'xlocale.h',    # see GH#8367
  'dlfcn.h',      # dladdr
  'sys/mman.h',   # madvise
]
foreach header: optional_headers
  config_h.set10('HAVE_' + header.to_upper().replace('.', '_').replace('/', '_'),
	cc.has_header(header))
endforeach

# Optional compiler attributes
config_h.set10('HAVE_ATTRIBUTE_NONNULL', cc.has_function_attribute('nonnull'))
# Is this actually defined in distutils build?
has_optimize_attribute = cc.has_function_attribute('optimize')  # FIXME
#config_h.set10('HAVE_ATTRIBUTE_OPTIMIZE_OPT_3', has_optimize_attribute)
#config_h.set10('HAVE_ATTRIBUTE_OPTIMIZE_UNROLL_LOOPS', has_optimize_attribute)

#TODO: __builtin_* (see setup_common.py)
#TODO: long double representation (see setup_common.py)


configure_file(input: 'config.h.in',
  output: 'config.h',
  configuration: config_h,
  install: false
)

# Build npymath static library
# ----------------------------

staticlib_cflags = []
if cc.get_id() == 'msvc'
  # Disable voltbl section for vc142 to allow link using mingw-w64; see:
  # https://github.com/matthew-brett/dll_investigation/issues/1#issuecomment-1100468171
  # Needs to be added to static libraries that are shipped for reuse (i.e.,
  # libnpymath and libnpyrandom)
  if cc.has_argument('-d2VolatileMetadata-')
     staticlib_cflags +=  '-d2VolatileMetadata-'
   endif
endif

npymath_sources = [
  src_file.process('src/npymath/ieee754.c.src'),
  src_file.process('src/npymath/npy_math_complex.c.src'),
  src_file.process('src/npymath/npy_math_internal.h.src'),
  'src/npymath/_signbit.c',
  'src/npymath/halffloat.c',
  'src/npymath/npy_math.c',
]
npymath_lib = static_library('npymath',
  npymath_sources,
  c_args: staticlib_cflags,
  include_directories: 'include',
  dependencies: py_dep,
  install: true,
  install_dir: 'numpy/core/lib',
)
# TODO: deal with npymath.ini.in (a pkg-config file)


python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_add_newdocs.py',
  '_add_newdocs_scalars.py',
  '_asarray.py',
  '_asarray.pyi',
  '_dtype.py',
  '_dtype_ctypes.py',
  '_exceptions.py',
  '_internal.py',
  '_internal.pyi',
  '_machar.py',
  '_methods.py',
  '_string_helpers.py',
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
  'records.py',
  'records.pyi',
  'shape_base.py',
  'shape_base.pyi',
  'umath.py',
  'umath_tests.py',
]

py.install_sources(
  python_sources,
  subdir: 'numpy/core'
)

subdir('include')
install_subdir('tests', install_dir: 'numpy/core')
