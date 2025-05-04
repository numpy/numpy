source_root = meson.project_source_root()
mod_features = import('features')

SSE41 = mod_features.new(
  'SSE41', 5,
  group: ['SSE', 'SSE2', 'SSE3', 'SSSE3'],
  args: ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1'],
  detect: 'SSE41',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse41.c')[0]
)
SSE4_COMMON = mod_features.new(
  'SSE4_COMMON', 7, implies: SSE41, args: ['-mpopcnt', '-msse4.2', '-maes', '-mpclmul'],
  group: ['POPCNT', 'SSE42', 'AES', 'PCLMULQDQ'],
  detect: 'SSE4_COMMON',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse42.c')[0]
)
AVX2_COMMON = mod_features.new(
  'AVX2_COMMON', 25, implies: SSE4_COMMON, args: ['-mavx2', '-mbmi', '-mbmi2', '-mfma', '-mf16c'],
  group: ['AVX2', 'BMI1', 'BMI2', 'FMA3', 'F16C', 'AVX'],
  detect: 'AVX2_COMMON',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx2.c')[0]
)
# 25-40 left as margin for any extra features
AVX512_SKX = mod_features.new(
  'AVX512_SKX', 50, implies: AVX2_COMMON,
  args: ['-mavx512f', '-mavx512cd', '-mavx512vl', '-mavx512bw', '-mavx512dq'],
  group: ['AVX512F', 'AVX512CD', 'AVX512VL', 'AVX512BW', 'AVX512DQ'],
  detect: 'AVX512_SKX',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_skx.c')[0],
  extra_tests: {
    'AVX512F_REDUCE': files(source_root + '/numpy/distutils/checks/extra_avx512f_reduce.c')[0],
    'AVX512BW_MASK': files(source_root + '/numpy/distutils/checks/extra_avx512bw_mask.c')[0],
    'AVX512DQ_MASK': files(source_root + '/numpy/distutils/checks/extra_avx512dq_mask.c')[0]
  }
)
AVX512_ICL = mod_features.new(
  'AVX512_ICL', 53, implies: AVX512_SKX,
  args: ['-mavx512vbmi', '-mavx512vbmi2', '-mavx512vnni', '-mavx512bitalg',
         '-mavx512vpopcntdq', '-mvaes', '-mgfni', '-mvpclmulqdq'],
  group: ['AVX512VBMI', 'AVX512VBMI2', 'AVX512VNNI', 'AVX512BITALG', 'AVX512VPOPCNTDQ',
          'VAES', 'GFNI', 'VPCLMULQDQ'],
  detect: 'AVX512_ICL',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_icl.c')[0]
)

# TODO add support for zen4
AVX512_SPR = mod_features.new(
  'AVX512_SPR', 55, implies: AVX512_ICL, 
  args: ['-mavx512fp16'],
  group: ['AVX512FP16'],
  detect: 'AVX512_SPR',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_spr.c')[0]
)

# Specializations for non unix-like compilers
# -------------------------------------------
cpu_family = host_machine.cpu_family()
compiler_id = meson.get_compiler('c').get_id()
if compiler_id not in ['gcc', 'clang']
  AVX512_SPR.update(disable: compiler_id + ' compiler does not support it')
endif

if compiler_id == 'intel-cl'
  SSE41.update(args: {'val': '/arch:SSE4.1', 'match': '/arch:.*'})
  SSE4_COMMON.update(args: {'val': '/arch:SSE4.2', 'match': '/arch:.*'})
  AVX2_COMMON.update(args: {'val': '/arch:CORE-AVX2', 'match': '/arch:.*'})
  AVX512_SKX.update(args: {'val': '/Qx:SKYLAKE-AVX512', 'match': '/[arch|Qx]:.*'})
  AVX512_ICL.update(args: {'val': '/Qx:ICELAKE-CLIENT', 'match': '/[arch|Qx]:.*'})
endif

if compiler_id == 'intel'
  clear_m = '^(-mcpu=|-march=)'
  clear_any = '^(-mcpu=|-march=|-x[A-Z0-9\-])'
  AVX2_COMMON.update(args: {'val': '-xCORE-AVX2', 'match': clear_m})
  AVX512_SKX.update(args: {'val': '-xSKYLAKE-AVX512', 'match': clear_any})
  AVX512_ICL.update(args: {'val': '-xICELAKE-CLIENT', 'match': clear_any})
endif

if compiler_id == 'msvc'
  # The following features don't own private FLAGS, however the compiler still
  # provides ISA capability for them.
  foreach fet : [
    SSE41, AVX512_ICL
  ]
    fet.update(args: '')
  endforeach
  clear_arch = '/arch:.*'
  SSE4_COMMON.update(args: {'val': ['/arch:SSE4.2', '-DHWY_WANT_SSE4'], 'match': clear_arch})
  AVX2_COMMON.update(args: {'val': '/arch:AVX2', 'match': clear_arch})
  AVX512_SKX.update(args: {'val': '/arch:AVX512', 'match': clear_arch})
endif

# legacy CPU features 
X86_REDIRECT = {
  'SSE': 'SSE41', 'SSE2': 'SSE41', 'SSE3': 'SSE41', 'SSSE3': 'SSE41',
  'SSE42': 'SSE4_COMMON', 'XOP': 'SSE4_COMMON', 'FMA4': 'SSE4_COMMON',
  'FMA3': 'SSE4_COMMON', 'AVX': 'SSE4_COMMON', 'F16C': 'SSE4_COMMON',
  'AVX512F': 'AVX2_COMMON', 'AVX512CD': 'AVX2_COMMON',
  'AVX512_KNL': 'AVX2_COMMON', 'AVX512_KNM': 'AVX2_COMMON',
  'AVX512_CLX': 'AVX512_SKX', 'AVX512_CNL': 'AVX512_SKX',
}

X86_FEATURES = {
  'SSE41': SSE41, 'SSE4_COMMON': SSE4_COMMON, 'AVX2_COMMON': AVX2_COMMON,
  'AVX512_SKX': AVX512_SKX, 'AVX512_ICL': AVX512_ICL, 'AVX512_SPR': AVX512_SPR
}
