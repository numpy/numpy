# This file should contain what setup.py + setup_common.py do (WIP)
#
# Potential issues to address or keep track of:
#   - sincos detection incorrect on NetBSD: https://github.com/mesonbuild/meson/issues/10641

# Generate config.h
# -----------------

cdata = configuration_data()

if not cc.has_header('Python.h', dependencies: py_dep)
  error('Cannot compile `Python.h`. Perhaps you need to install python-dev|python-devel')
endif

use_svml = (
  host_machine.system() == 'linux' and
  host_machine.cpu_family() == 'x86_64' and
  not get_option('disable-svml')
)
if use_svml
  cdata.set10('NPY_CAN_LINK_SVML', true)
  if not fs.exists('src/umath/svml')
    error('Missing the `SVML` git submodule! Run `git submodule update --init` to fix this.')
  endif
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
]
foreach symbol_type: types_to_check
  cdata.set(symbol_type[0], cc.sizeof(symbol_type[1]))
endforeach
cdata.set('NPY_SIZEOF_OFF_T', cc.sizeof('off_t', prefix: '#include <sys/types.h>'))
cdata.set('NPY_SIZEOF_PY_INTPTR_T',
  cc.sizeof('Py_intptr_t', dependencies: py_dep, prefix: '#include <Python.h>'))
cdata.set('NPY_SIZEOF_PY_LONG_LONG',
  cc.sizeof('PY_LONG_LONG', dependencies: py_dep, prefix: '#include <Python.h>'))

# Check for complex support
if cc.has_header('complex.h')
  cdata.set10('HAVE_COMPLEX_H', true)
  cdata.set10('NPY_USE_C99_COMPLEX', true)
  complex_types_to_check = [
    ['NPY_HAVE_COMPLEX_FLOAT', 'NPY_SIZEOF_COMPLEX_FLOAT', 'complex float', 'float'],
    ['NPY_HAVE_COMPLEX_DOUBLE', 'NPY_SIZEOF_COMPLEX_DOUBLE', 'complex double', 'double'],
    ['NPY_HAVE_COMPLEX_LONG_DOUBLE', 'NPY_SIZEOF_COMPLEX_LONGDOUBLE', 'complex long double', 'long double'],
  ]
  foreach symbol_type: complex_types_to_check
    if cc.has_type(symbol_type[2], prefix: '#include <complex.h>')
      cdata.set10(symbol_type[0], true)
      # Determine size of NumPy's own, struct-based, complex types. Binary
      # compatibility with C99 complex types is checked at build time in `npy_common.h`.
      ftype = symbol_type[3]
      cdata.set(symbol_type[1], cc.sizeof(f'struct {@ftype@ __x; @ftype@ __y;}'))
    endif
  endforeach
endif

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

c99_complex_funcs = [
  'cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan',
  'catanh', 'cexp', 'clog', 'cpow', 'csqrt'
]
foreach func: c99_complex_funcs
  func_single = func + 'f'
  func_longdouble = func + 'l'
  cdata.set10('HAVE_' + func.to_upper(), cc.has_function(func))
  cdata.set10('HAVE_' + func_single.to_upper(), cc.has_function(func_single))
  cdata.set10('HAVE_' + func_longdouble.to_upper(), cc.has_function(func_longdouble))
endforeach

# We require C99 so these should always be found at build time. But for
# libnpymath as a C99 compat layer, these may still be relevant.
c99_macros = ['isfinite', 'isinf', 'isnan', 'signbit']
foreach macro: c99_macros
  cdata.set10('NPY_HAVE_DECL_' + macro.to_upper(), cc.has_function(macro))
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
      cdata.set10(config_value,
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
cdata.set10('HAVE_STRTOLD_L', result)

# Other optional functions
cdata.set10('HAVE_BACKTRACE',
  cc.has_function('backtrace', prefix: '#include <execinfo.h>')
)
cdata.set10('HAVE_MADVISE',
  cc.has_function('madvise', include_directories: inc_curdir,
    prefix: '#include "feature_detection_misc.h"'
  )
)

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
  if cc.has_header(header)
    cdata.set10('HAVE_' + header.to_upper().replace('.', '_').replace('/', '_'), true)
  endif
endforeach

# Optional compiler attributes
cdata.set10('HAVE_ATTRIBUTE_NONNULL', cc.has_function_attribute('nonnull'))
# Is this actually defined in distutils build?
has_optimize_attribute = cc.has_function_attribute('optimize')  # FIXME
#cdata.set10('HAVE_ATTRIBUTE_OPTIMIZE_OPT_3', has_optimize_attribute)
#cdata.set10('HAVE_ATTRIBUTE_OPTIMIZE_UNROLL_LOOPS', has_optimize_attribute)

#TODO: __builtin_* (see setup_common.py)

# FIXME: long double representation detection (see setup_common.py)
#        add the regular x86-64 define (this is a hack to get further in the
#        build without dealing with this thing right now)
cdata.set10('HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE', true)

cdata.set10('NPY_HAVE_ENDIAN_H', cc.has_header('endian.h'))
cdata.set10('NPY_HAVE_SYS_ENDIAN_H', cc.has_header('sys/endian.h'))
if is_windows
  cdata.set10('NPY_NO_SIGNAL', true)
endif
# Command-line switch; distutils build checked for `NPY_NOSMP` env var instead
# TODO: document this (search for NPY_NOSMP in C API docs)
cdata.set10('NPY_NO_SMP', get_option('disable-threading'))

# Use bogus stride debug aid to flush out bugs where users use strides of
# dimensions with length 1 to index a full contiguous array.
cdata.set10('NPY_RELAXED_STRIDES_DEBUG', get_option('relaxed-strides-debug'))

# Check whether we can use inttypes (C99) formats
if cc.has_header_symbol('inttypes.h', 'PRIdPTR')
  cdata.set10('NPY_USE_C99_FORMATS', true)
endif

visibility_hidden = ''
if cc.has_function_attribute('visibility:hidden')
  visibility_hidden = '__attribute__((visibility("hidden")))'
endif
cdata.set('NPY_VISIBILITY_HIDDEN', visibility_hidden)

# TODO: implement version checks and update docs on this - see setup_common.py
cdata.set('NPY_ABI_VERSION', '0x00000009')
cdata.set('NPY_API_VERSION', '0x00000010')


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
  install_dir: np_dir / 'core/include/numpy'
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

npy_math_internal_h = src_file.process('src/npymath/npy_math_internal.h.src')

npymath_sources = [
  src_file.process('src/npymath/ieee754.c.src'),
  src_file.process('src/npymath/npy_math_complex.c.src'),
  npy_math_internal_h,
  'src/npymath/_signbit.c',
  'src/npymath/halffloat.c',
  'src/npymath/npy_math.c',
]
npymath_lib = static_library('npymath',
  npymath_sources,
  c_args: staticlib_cflags,
  include_directories: ['include', 'src/npymath', 'src/common'],
  dependencies: py_dep,
  install: true,
  install_dir: np_dir / 'core/lib',
)
# TODO: deal with npymath.ini.in (a pkg-config file)


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
  input : 'code_generators/generate_umath_doc.py',
  command: [py, '@INPUT@', '-o', '@OUTPUT@'],
)

src_numpy_api = custom_target('__multiarray_api',
  output : ['__multiarray_api.c', '__multiarray_api.h', 'multiarray_api.txt'],
  input : 'code_generators/generate_numpy_api.py',
  command: [py, '@INPUT@', '-o', '@OUTDIR@', '--ignore', src_umath_api_c],
  install: true,  # NOTE: setup.py build installs all, but just need .h?
  install_dir: np_dir / 'core/include/numpy'
)

src_ufunc_api = custom_target('__ufunc_api',
  output : ['__ufunc_api.c', '__ufunc_api.h', 'ufunc_api.txt'],
  input : 'code_generators/generate_ufunc_api.py',
  command: [py, '@INPUT@', '-o', '@OUTDIR@'],
  install: true,  # NOTE: setup.py build installs all, but just need .h?
  install_dir: np_dir / 'core/include/numpy'
)


# Set common build flags for C and C++ code
# -----------------------------------------

# TODO: change to "feature" option in meson_options.txt? See
# https://mesonbuild.com/Build-options.html#build-options
disable_simd_optimizations = []
if get_option('disable-simd-optimizations')
  disable_simd_optimizations = '-DNPY_DISABLE_OPTIMIZATION'
endif

# Common build flags
c_args_common = [
  '-DNPY_INTERNAL_BUILD',
  '-DHAVE_NPY_CONFIG_H',
  disable_simd_optimizations,
  cflags_large_file_support,
]

# Same as NPY_CXX_FLAGS (TODO: extend for what ccompiler_opt adds)
cpp_args_common = c_args_common + [
  '-D__STDC_VERSION__=0',  # for compatibility with C headers
  '-fno-exceptions',  # no exception support
  '-fno-rtti',  # no runtime type information
]

# Build multiarray_tests module
# -----------------------------
templ_common_h = src_file.process('src/common/templ_common.h.src')

py.extension_module('_multiarray_tests',
  [
    src_file.process('src/multiarray/_multiarray_tests.c.src'),
    'src/common/mem_overlap.c',
    'src/common/npy_argparse.c',
    'src/common/npy_hashtable.c',
    src_numpy_api[1],  # __multiarray_api.h
    npy_math_internal_h,
    templ_common_h,
  ],
  c_args: c_args_common,
  include_directories: ['include', 'src/common', 'src/multiarray', 'src/npymath'],
  link_with: npymath_lib,
  install: true,
  subdir: 'numpy/core',
)

# Build _multiarray_umath module
# ------------------------------
src_multiarray_umath_common = [
  'src/common/array_assign.c',
  'src/common/mem_overlap.c',
  'src/common/npy_argparse.c',
  'src/common/npy_hashtable.c',
  'src/common/npy_longdouble.c',
  'src/common/ucsnarrow.c',
  'src/common/ufunc_override.c',
  'src/common/numpyos.c',
  'src/common/npy_cpu_features.c',
  templ_common_h,
]
if have_blas
  src_multiarray_umath_common += [
    'src/common/cblasfuncs.c',
    'src/common/python_xerbla.c',
  ]
endif

src_multiarray = [
  'src/multiarray/abstractdtypes.c',
  'src/multiarray/alloc.c',
  src_file.process('src/multiarray/argfunc.dispatch.c.src'),
  'src/multiarray/arrayobject.c',
  src_file.process('src/multiarray/arraytypes.h.src'),
  'src/multiarray/array_coercion.c',
  'src/multiarray/array_method.c',
  'src/multiarray/array_assign_scalar.c',
  'src/multiarray/array_assign_array.c',
  'src/multiarray/arrayfunction_override.c',
  src_file.process('src/multiarray/arraytypes.c.src'),
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
  src_file.process('src/multiarray/einsum.c.src'),
  src_file.process('src/multiarray/einsum_sumprod.c.src'),
  'src/multiarray/experimental_public_dtype_api.c',
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
  'src/multiarray/number.c',
  'src/multiarray/refcount.c',
  src_file.process('src/multiarray/scalartypes.c.src'),
  'src/multiarray/sequence.c',
  'src/multiarray/shape.c',
  'src/multiarray/scalarapi.c',
  'src/multiarray/strfuncs.c',
  'src/multiarray/temp_elide.c',
  'src/multiarray/typeinfo.c',
  'src/multiarray/usertypes.c',
  'src/multiarray/vdot.c',
  src_file.process('src/common/npy_sort.h.src'),
  'src/npysort/x86-qsort.dispatch.cpp',
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
]

src_umath = [
  src_file.process('src/umath/funcs.inc.src'),
  src_file.process('src/umath/loops.h.src'),
  src_file.process('src/umath/loops_utils.h.src'),
  src_file.process('src/umath/loops.c.src'),
  src_file.process('src/umath/loops_arithm_fp.dispatch.c.src'),
  src_file.process('src/umath/loops_arithmetic.dispatch.c.src'),
  src_file.process('src/umath/loops_comparison.dispatch.c.src'),
  src_file.process('src/umath/loops_exponent_log.dispatch.c.src'),
  src_file.process('src/umath/loops_hyperbolic.dispatch.c.src'),
  src_file.process('src/umath/loops_minmax.dispatch.c.src'),
  src_file.process('src/umath/loops_modulo.dispatch.c.src'),
  src_file.process('src/umath/loops_trigonometric.dispatch.c.src'),
  src_file.process('src/umath/loops_umath_fp.dispatch.c.src'),
  src_file.process('src/umath/loops_unary_fp.dispatch.c.src'),
  src_file.process('src/umath/matmul.c.src'),
  src_file.process('src/umath/matmul.h.src'),
  src_file.process('src/umath/simd.inc.src'),
  'src/umath/ufunc_type_resolution.c',
  'src/umath/clip.cpp',
  'src/umath/clip.h',
  'src/umath/dispatching.c',
  'src/umath/extobj.c',
  'src/umath/legacy_array_method.c',
  'src/umath/override.c',
  'src/umath/reduction.c',
  src_file.process('src/umath/scalarmath.c.src'),
  'src/umath/ufunc_object.c',
  'src/umath/umathmodule.c',
  'src/umath/string_ufuncs.cpp',
  'src/umath/wrapping_array_method.c',
  # For testing. Eventually, should use public API and be separate:
  'src/umath/_scaled_float_dtype.c',
]

# TODO: add SVML sources

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
  c_args: c_args_common,
  cpp_args: cpp_args_common,
  include_directories: [
    'include',
    'src/common',
    'src/multiarray',
    'src/npymath',
    'src/umath',
  ],
  dependencies: [py_dep, blas],
  link_with: npymath_lib,
  install: true,
  subdir: 'numpy/core',
)



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
install_subdir('tests', install_dir: np_dir / 'core')
