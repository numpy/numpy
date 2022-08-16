# TODO: add a build option equivalent to NPY_DISABLE_SVML

# Generate config.h
cdata = configuration_data()

cdata.set10('TOKEN', true)
cdata.set('A_STRING', 'Some text.')

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
  cdata.set10('HAVE_' + func, cc.has_function(func))
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
  cdata.set10('HAVE_' + func_single, cc.has_function(func_single))
  cdata.set10('HAVE_' + func_longdouble, cc.has_function(func_longdouble))
endforeach
C99_COMPLEX_TYPES = [
  'complex double', 'complex float', 'complex long double'
]
C99_COMPLEX_FUNCS = [
  'cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan',
  'catanh', 'ccos', 'ccosh', 'cexp', 'cimag', 'clog', 'conj', 'cpow',
  'cproj', 'creal', 'csin', 'csinh', 'csqrt', 'ctan', 'ctanh'
]

optional_locale_funcs = ['strtold_l']
optional_file_funcs = ['ftello', 'fseeko', 'fallocate']
optional_misc_funcs = ['backtrace', 'madvise']

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

configure_file(input: 'config.h.in',
  output: 'config.h',
  configuration: cdata)
