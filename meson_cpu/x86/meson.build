source_root = meson.project_source_root()
mod_features = import('features')

SSE = mod_features.new(
  'SSE', 1,  args: '-msse',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse.c')[0]
)
SSE2 = mod_features.new(
  'SSE2', 2, implies: SSE,
  args: '-msse2',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse2.c')[0]
)
# enabling SSE without SSE2 is useless also it's non-optional for x86_64
SSE.update(implies: SSE2)
SSE3 = mod_features.new(
  'SSE3', 3, implies: SSE2,
  args: '-msse3',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse3.c')[0]
)
SSSE3 = mod_features.new(
  'SSSE3', 4, implies: SSE3,
  args: ['-mssse3','-DHWY_WANT_SSSE3=1'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_ssse3.c')[0]
)
SSE41 = mod_features.new(
  'SSE41', 5, implies: SSSE3,
  args: '-msse4.1',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse41.c')[0]
)
POPCNT = mod_features.new(
  'POPCNT', 6, implies: SSE41,
  args: '-mpopcnt',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_popcnt.c')[0]
)
SSE42 = mod_features.new(
  'SSE42', 7, implies: POPCNT, args: ['-msse4.2', '-DHWY_WANT_SSE4=1'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_sse42.c')[0]
)
# 7-20 left as margin for any extra features
AVX = mod_features.new(
  'AVX', 20, implies: SSE42, args: '-mavx',
  detect: {'val': 'AVX', 'match': '.*SSE.*'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx.c')[0]
)
XOP = mod_features.new(
  'XOP', 21, implies: AVX, args: '-mxop',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_xop.c')[0]
)
FMA4 = mod_features.new(
  'FMA4', 22, implies: AVX, args: '-mfma4',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_fma4.c')[0]
)
# x86 half-precision
F16C = mod_features.new(
  'F16C', 23, implies: AVX, args: '-mf16c',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_f16c.c')[0]
)
FMA3 = mod_features.new(
  'FMA3', 24, implies: F16C, args: '-mfma',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_fma3.c')[0]
)
# match this to HWY_AVX2
AVX2 = mod_features.new(
  'AVX2', 25, implies: FMA3, args: ['-mavx2', '-maes', '-mpclmul', '-mbmi', '-mbmi2'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx2.c')[0]
)
# 25-40 left as margin for any extra features
AVX512F = mod_features.new(
  'AVX512F', 40, implies: [AVX2],
  # Disables mmx because of stack corruption that may happen during mask
  # conversions.
  # TODO (seiko2plus): provide more clarification
  args: ['-mno-mmx', '-mavx512f'],
  detect: {'val': 'AVX512F', 'match': '.*'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512f.c')[0],
  extra_tests: {
    'AVX512F_REDUCE': files(source_root + '/numpy/distutils/checks/extra_avx512f_reduce.c')[0]
  }
)
AVX512CD = mod_features.new(
  'AVX512CD', 41, implies: AVX512F, args: '-mavx512cd',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512cd.c')[0]
)
AVX512_KNL = mod_features.new(
  'AVX512_KNL', 42, implies: AVX512CD, args: ['-mavx512er', '-mavx512pf'],
  group: ['AVX512ER', 'AVX512PF'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_knl.c')[0]
)
AVX512_KNM = mod_features.new(
  'AVX512_KNM', 43, implies: AVX512_KNL,
  args: ['-mavx5124fmaps', '-mavx5124vnniw', '-mavx512vpopcntdq'],
  group: ['AVX5124FMAPS', 'AVX5124VNNIW', 'AVX512VPOPCNTDQ'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_knm.c')[0]
)
AVX512_SKX = mod_features.new(
  'AVX512_SKX', 50, implies: AVX512CD,
  args: ['-mavx512vl', '-mavx512bw', '-mavx512dq'],
  group: ['AVX512VL', 'AVX512BW', 'AVX512DQ'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_skx.c')[0],
  extra_tests: {
    'AVX512BW_MASK': files(source_root + '/numpy/distutils/checks/extra_avx512bw_mask.c')[0],
    'AVX512DQ_MASK': files(source_root + '/numpy/distutils/checks/extra_avx512dq_mask.c')[0]
  }
)
AVX512_CLX = mod_features.new(
  'AVX512_CLX', 51, implies: AVX512_SKX, args: '-mavx512vnni',
  group: ['AVX512VNNI'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_clx.c')[0]
)
AVX512_CNL = mod_features.new(
  'AVX512_CNL', 52, implies: AVX512_SKX,
  args: ['-mavx512ifma', '-mavx512vbmi'],
  group: ['AVX512IFMA', 'AVX512VBMI'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_cnl.c')[0]
)
AVX512_ICL = mod_features.new(
  'AVX512_ICL', 53, implies: [AVX512_CLX, AVX512_CNL],
  args: ['-mavx512vbmi2', '-mavx512bitalg', '-mavx512vpopcntdq'],
  group: ['AVX512VBMI2', 'AVX512BITALG', 'AVX512VPOPCNTDQ'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_icl.c')[0]
)
# TODO add support for zen4
AVX512_SPR = mod_features.new(
  'AVX512_SPR', 55, implies: AVX512_ICL,
  args: ['-mavx512fp16'],
  group: ['AVX512FP16'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_spr.c')[0]
)

# Specializations for non unix-like compilers
# -------------------------------------------
cpu_family = host_machine.cpu_family()
compiler_id = meson.get_compiler('c').get_id()
if compiler_id not in ['gcc', 'clang']
  AVX512_SPR.update(disable: compiler_id + ' compiler does not support it')
endif

# Common specializations between both Intel compilers (unix-like and msvc-like)
if compiler_id in ['intel', 'intel-cl']
  # POPCNT, and F16C don't own private FLAGS however the compiler still
  # provides ISA capability for them.
  POPCNT.update(args: '')
  F16C.update(args: '')
  # Intel compilers don't support the following features independently
  FMA3.update(implies: [F16C, AVX2])
  AVX2.update(implies: [F16C, FMA3])
  AVX512F.update(implies: [AVX2, AVX512CD])
  AVX512CD.update(implies: [AVX512F])
  XOP.update(disable: 'Intel Compiler does not support it')
  FMA4.update(disable: 'Intel Compiler does not support it')
endif

if compiler_id == 'intel-cl'
  foreach fet : [SSE, SSE2, SSE3, SSSE3, AVX]
    fet.update(args: {'val': '/arch:' + fet.get('name'), 'match': '/arch:.*'})
  endforeach
  SSE41.update(args: {'val': '/arch:SSE4.1', 'match': '/arch:.*'})
  SSE42.update(args: {'val': '/arch:SSE4.2', 'match': '/arch:.*'})
  FMA3.update(args: {'val': '/arch:CORE-AVX2', 'match': '/arch:.*'})
  AVX2.update(args: {'val': '/arch:CORE-AVX2', 'match': '/arch:.*'})
  AVX512F.update(args: {'val': '/Qx:COMMON-AVX512', 'match': '/arch:.*'})
  AVX512CD.update(args: {'val': '/Qx:COMMON-AVX512', 'match': '/arch:.*'})
  AVX512_KNL.update(args: {'val': '/Qx:KNL', 'match': '/[arch|Qx]:.*'})
  AVX512_KNM.update(args: {'val': '/Qx:KNM', 'match': '/[arch|Qx]:.*'})
  AVX512_SKX.update(args: {'val': '/Qx:SKYLAKE-AVX512', 'match': '/[arch|Qx]:.*'})
  AVX512_CLX.update(args: {'val': '/Qx:CASCADELAKE', 'match': '/[arch|Qx]:.*'})
  AVX512_CNL.update(args: {'val': '/Qx:CANNONLAKE', 'match': '/[arch|Qx]:.*'})
  AVX512_ICL.update(args: {'val': '/Qx:ICELAKE-CLIENT', 'match': '/[arch|Qx]:.*'})
endif

if compiler_id == 'intel'
  clear_m = '^(-mcpu=|-march=)'
  clear_any = '^(-mcpu=|-march=|-x[A-Z0-9\-])'
  FMA3.update(args: {'val': '-xCORE-AVX2', 'match': clear_m})
  AVX2.update(args: {'val': '-xCORE-AVX2', 'match': clear_m})
  AVX512F.update(args: {'val': '-xCOMMON-AVX512', 'match': clear_m})
  AVX512CD.update(args: {'val': '-xCOMMON-AVX512', 'match': clear_m})
  AVX512_KNL.update(args: {'val': '-xKNL', 'match': clear_any})
  AVX512_KNM.update(args: {'val': '-xKNM', 'match': clear_any})
  AVX512_SKX.update(args: {'val': '-xSKYLAKE-AVX512', 'match': clear_any})
  AVX512_CLX.update(args: {'val': '-xCASCADELAKE', 'match': clear_any})
  AVX512_CNL.update(args: {'val': '-xCANNONLAKE', 'match': clear_any})
  AVX512_ICL.update(args: {'val': '-xICELAKE-CLIENT', 'match': clear_any})
endif

if compiler_id == 'msvc'
  # MSVC compiler doesn't support the following features
  foreach fet : [AVX512_KNL, AVX512_KNM]
    fet.update(disable: compiler_id + ' compiler does not support it')
  endforeach
  # The following features don't own private FLAGS, however the compiler still
  # provides ISA capability for them.
  foreach fet : [
    SSE3, SSSE3, SSE41, POPCNT, SSE42, AVX, F16C, XOP, FMA4,
    AVX512F, AVX512CD, AVX512_CLX, AVX512_CNL,
    AVX512_ICL
  ]
    fet.update(args: '')
  SSE42.update(args: '-DHWY_WANT_SSE4=1')
  SSSE3.update(args: '-DHWY_WANT_SSSE3=1')
  endforeach
  # MSVC compiler doesn't support the following features independently
  FMA3.update(implies: [F16C, AVX2])
  AVX2.update(implies: [F16C, FMA3])
  AVX512F.update(implies: [AVX2, AVX512CD, AVX512_SKX])
  AVX512CD.update(implies: [AVX512F, AVX512_SKX])
  clear_arch = '/arch:.*'
  # only available on 32-bit. Its enabled by default on 64-bit mode
  foreach fet : [SSE, SSE2]
    if cpu_family == 'x86'
      fet.update(args: {'val': '/arch:' + fet.get('name'), 'match': clear_arch})
    else
      fet.update(args: '')
    endif
  endforeach
  FMA3.update(args: {'val': '/arch:AVX2', 'match': clear_arch})
  AVX2.update(args: {'val': '/arch:AVX2', 'match': clear_arch})
  AVX512_SKX.update(args: {'val': '/arch:AVX512', 'match': clear_arch})
endif

X86_FEATURES = {
 'SSE': SSE, 'SSE2': SSE2, 'SSE3': SSE3, 'SSSE3': SSSE3,
 'SSE41': SSE41, 'POPCNT': POPCNT, 'SSE42': SSE42, 'AVX': AVX,
 'XOP': XOP, 'FMA4': FMA4, 'F16C': F16C, 'FMA3': FMA3,
 'AVX2': AVX2, 'AVX512F': AVX512F, 'AVX512CD': AVX512CD,
 'AVX512_KNL': AVX512_KNL, 'AVX512_KNM': AVX512_KNM,
 'AVX512_SKX': AVX512_SKX, 'AVX512_CLX': AVX512_CLX,
 'AVX512_CNL': AVX512_CNL, 'AVX512_ICL': AVX512_ICL,
 'AVX512_SPR': AVX512_SPR
}
