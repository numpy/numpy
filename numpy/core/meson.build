# TODO: add a build option equivalent to NPY_DISABLE_SVML

# Generate config.h
cdata = configuration_data()

# Check sizes of types
cdata.set('SIZEOF_PY_INTPTR_T',
          cc.sizeof('Py_intptr_t',
		    dependencies: py3_dep,
		    prefix: '#include <Python.h>'))
cdata.set('SIZEOF_OFF_T',
          cc.sizeof('off_t', dependencies: py3_dep))  # doesn't seem to need Python.h

# Check for complex support
cdata.set10('HAVE_COMPLEX_H',
            cc.has_header('complex.h'))
# TODO: NPY_USE_C99_COMPLEX and on in check_complex()

if not cc.has_header('Python.h', dependencies: py3_dep)
  error('Cannot compile `Python.h`. Perhaps you need to install python-dev|python-devel')
endif

# Mandatory functions: if not found, fail the build
mandatory_funcs = [
  'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs',
  'floor', 'ceil', 'sqrt', 'log10', 'log', 'exp', 'asin',
  'acos', 'atan', 'fmod', 'modf', 'frexp', 'ldexp',
  'expm1', 'log1p', 'acosh', 'asinh', 'atanh',
  'rint', 'trunc', 'exp2', 
  'copysign', 'nextafter', 'strtoll', 'strtoull', 'cbrt'
]
foreach func: mandatory_funcs
  if not cc.has_function(func)
    error('Function {func} not found')
  endif
endforeach

# Standard functions which may not be available and for which we have a
# replacement implementation. Note that some of these are C99 functions.
optional_stdfuncs = ['log2', 'hypot', 'atan2', 'pow']
foreach func: optional_stdfuncs
  cdata.set10('HAVE_' + func.to_upper(), cc.has_function(func))
endforeach

# Subset of OPTIONAL_STDFUNCS which may already have HAVE_* defined by Python.h
optional_stdfuncs_maybe = ['ftello', 'fseeko']

# C99 functions: float and long double versions
C99_FUNCS = [
  'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs', 'floor', 'ceil',
  'rint', 'trunc', 'sqrt', 'log10', 'log', 'log1p', 'exp', 'expm1',
  'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh', 'hypot', 'atan2',
  'pow', 'fmod', 'modf', 'frexp', 'ldexp', 'exp2', 'log2', 'copysign',
  'nextafter', 'cbrt'
]
foreach func: optional_stdfuncs
  func_single = func + 'f'
  func_longdouble = func + 'l'
  cdata.set10('HAVE_' + func_single.to_upper(),
              cc.has_function(func_single))
  cdata.set10('HAVE_' + func_longdouble.to_upper(),
	      cc.has_function(func_longdouble))
endforeach

C99_COMPLEX_TYPES = [
  'complex double', 'complex float', 'complex long double'
]
c99_complex_funcs = [
  'cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan',
  'catanh', 'ccos', 'ccosh', 'cexp', 'cimag', 'clog', 'conj', 'cpow',
  'cproj', 'creal', 'csin', 'csinh', 'csqrt', 'ctan', 'ctanh'
]
foreach func: c99_complex_funcs
  func_single = func + 'f'
  func_longdouble = func + 'l'
  cdata.set10('HAVE_' + func.to_upper(), cc.has_function(func))
  cdata.set10('HAVE_' + func_single.to_upper(), cc.has_function(func_single))
  cdata.set10('HAVE_' + func_longdouble.to_upper(), cc.has_function(func_longdouble))
endforeach

# Optional locale function
code = '''
#pragma GCC diagnostic ignored "-Wnonnull"
long double strtold_l(const char*, char**, locale_t);
'''
result = cc.compiles(code, name: 'Check strtold_l')
cdata.set10('HAVE_STRTOLD_L', result)

# Optional file I/O related functions
code = '''
#include <stdio.h>
#include <fcntl.h>

off_t ftello(FILE *stream);
int fseeko(FILE *stream, off_t offset, int whence);
int fallocate(int, int, off_t, off_t);
'''
result = cc.compiles(code, name: 'Check file funcs')
cdata.set10('HAVE_FTELLO', result)
cdata.set10('HAVE_FSEEKO', result)
cdata.set10('HAVE_FALLOCATE', result)

# Other optional functions
cdata.set10('HAVE_BACKTRACE',
            cc.has_function('backtrace', prefix: '#include <execinfo.h>'))
cdata.set10('HAVE_MADVISE',
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
  cdata.set10('HAVE_' + header.to_upper().replace('.', '_').replace('/', '_'),
	      cc.has_header(header))
endforeach

# Optional compiler attributes
cdata.set10('HAVE_ATTRIBUTE_NONNULL', cc.has_function_attribute('nonnull'))
# Is this actually defined in distutils build?
has_optimize_attribute = cc.has_function_attribute('optimize')  # FIXME
#cdata.set10('HAVE_ATTRIBUTE_OPTIMIZE_OPT_3', has_optimize_attribute)
#cdata.set10('HAVE_ATTRIBUTE_OPTIMIZE_UNROLL_LOOPS', has_optimize_attribute)

#TODO: __builtin_* (see setup_common.py)
#TODO: long double representation (see setup_common.py)


configure_file(input: 'config.h.in',
  output: 'config.h',
  configuration: cdata)
