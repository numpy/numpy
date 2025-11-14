source_root = meson.project_source_root()
current_dir = meson.current_source_dir()
cpu_family = host_machine.cpu_family()
mod_features = import('features')

HWY_SSE4_FLAGS = ['-DHWY_WANT_SSE4', '-DHWY_DISABLE_PCLMUL_AES']
# Use SSE for floating-point on x86-32 to ensure numeric consistency.
# The x87 FPU's 80-bit internal precision causes unpredictable rounding
# and overflow behavior when converting to smaller types. SSE maintains
# strict 32/64-bit precision throughout all calculations.
X86_64_V2_FLAGS = cpu_family == 'x86'? ['-mfpmath=sse'] : ['-mcx16']
X86_64_V2_NAMES = cpu_family == 'x86'? [] : ['CX16']
X86_V2 = mod_features.new(
  'X86_V2', 1, args: ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2',
                      '-mpopcnt', '-msahf'] + X86_64_V2_FLAGS + HWY_SSE4_FLAGS,
  # Adds compiler definitions `NPY_HAVE_SSE*` 
  group: ['SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE41', 'SSE42', 'POPCNT', 'LAHF'] + X86_64_V2_NAMES,
  detect: 'X86_V2',
  test_code: files(current_dir + '/test_x86_v2.c')[0],
)
X86_V3 = mod_features.new(
  'X86_V3', 10, implies: X86_V2, 
  args: ['-mavx', '-mavx2', '-mfma', '-mbmi', '-mbmi2', '-mlzcnt', '-mf16c', '-mmovbe'],
  group: ['AVX', 'AVX2', 'FMA3', 'BMI', 'BMI2', 'LZCNT', 'F16C', 'MOVBE'],
  detect: 'X86_V3',
  test_code: files(current_dir + '/test_x86_v3.c')[0],
)
X86_V4 = mod_features.new(
  'X86_V4', 20, implies: X86_V3,
  args: ['-mavx512f', '-mavx512cd', '-mavx512vl', '-mavx512bw', '-mavx512dq'],
  group: ['AVX512F', 'AVX512CD', 'AVX512VL', 'AVX512BW', 'AVX512DQ', 'AVX512_SKX',
          'AVX512F_REDUCE', 'AVX512BW_MASK', 'AVX512DQ_MASK'],
  detect: 'X86_V4',
  test_code: files(current_dir + '/test_x86_v4.c')[0],
)
if cpu_family == 'x86'
  X86_V4.update(disable: 'not supported on x86-32')
endif
AVX512_ICL = mod_features.new(
  'AVX512_ICL', 30, implies: X86_V4,
  args: ['-mavx512vbmi', '-mavx512vbmi2', '-mavx512vnni', '-mavx512bitalg',
         '-mavx512vpopcntdq', '-mavx512ifma', '-mvaes', '-mgfni', '-mvpclmulqdq'],
  group: ['AVX512VBMI', 'AVX512VBMI2', 'AVX512VNNI', 'AVX512BITALG', 'AVX512VPOPCNTDQ',
          'AVX512IFMA', 'VAES', 'GFNI', 'VPCLMULQDQ'],
  detect: 'AVX512_ICL',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_icl.c')[0]
)
AVX512_SPR = mod_features.new(
  'AVX512_SPR', 35, implies: AVX512_ICL, 
  args: ['-mavx512fp16', '-mavx512bf16'],
  group: ['AVX512FP16', 'AVX512BF16'],
  detect: 'AVX512_SPR',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_avx512_spr.c')[0]
)

# Specializations for non unix-like compilers
# -------------------------------------------
cc = meson.get_compiler('c')
compiler_id = cc.get_id()
if compiler_id not in ['gcc', 'clang']
  AVX512_SPR.update(disable: compiler_id + ' compiler does not support it')
endif

if compiler_id == 'intel-cl'
  X86_V2.update(args: [{'val': '/arch:SSE4.2', 'match': '/arch:.*'}] + HWY_SSE4_FLAGS)
  X86_V3.update(args: {'val': '/arch:CORE-AVX2', 'match': '/arch:.*'})
  X86_V4.update(args: {'val': '/Qx:SKYLAKE-AVX512', 'match': '/[arch|Qx]:.*'})
  AVX512_ICL.update(args: {'val': '/Qx:ICELAKE-CLIENT', 'match': '/[arch|Qx]:.*'})
endif

if compiler_id == 'intel'
  clear_any = '^(-mcpu=|-march=|-x[A-Z0-9\-])'
  X86_V2.update(args: [{'val': '-xSSE4.2', 'match': clear_any}] + HWY_SSE4_FLAGS)
  X86_V3.update(args: {'val': '-xCORE-AVX2', 'match': clear_any})
  X86_V4.update(args: {'val': '-xSKYLAKE-AVX512', 'match': clear_any})
  AVX512_ICL.update(args: {'val': '-xICELAKE-CLIENT', 'match': clear_any})
endif

if compiler_id == 'msvc'
  cc_ver = cc.version()
  MSVC_SSE4 = cc_ver.version_compare('>=19.40') ? ['/arch:SSE4.2'] : []
  # 32-bit MSVC does not support /arch:SSE4.2
  MSVC_SSE4 = cpu_family == 'x86' ? ['/arch:SSE2'] : MSVC_SSE4
  MSVC_SSE4 = cc_ver.version_compare('>=19.30') ? MSVC_SSE4 + ['/fp:contract'] : MSVC_SSE4 
  X86_V2.update(args: MSVC_SSE4 + HWY_SSE4_FLAGS)
  clear_arch = '/arch:.*'
  X86_V3.update(args: {'val': '/arch:AVX2', 'match': clear_arch})
  # FIXME: After completing transition from universal intrinsics to Highway,
  # investigate which MSVC versions are incompatible with Highway's AVX-512 implementation.
  X86_V4.update(disable: 'Considered broken by Highway on MSVC')
  # To force enable AVX-512, use:
  # X86_V4.update(args: [{'val': '/arch:AVX512', 'match': clear_arch}, '-DHWY_BROKEN_MSVC=0'])
  AVX512_ICL.update(disable: 'unsupported by Highway on MSVC')
endif

# legacy CPU features 
X86_REDIRECT = {
  'SSE': 'X86_V2', 'SSE2': 'X86_V2', 'SSE3': 'X86_V2', 'SSSE3': 'X86_V2',
  'SSE41': 'X86_V2', 'SSE42': 'X86_V2', 'XOP': 'X86_V2', 'FMA4': 'X86_V2',
  'FMA3': 'X86_V3', 'AVX': 'X86_V3', 'F16C': 'X86_V3',
  'AVX512F': 'X86_V3', 'AVX512CD': 'X86_V3',
  'AVX512_KNL': 'X86_V3', 'AVX512_KNM': 'X86_V3',
  'AVX512_SKX': 'X86_V4', 'AVX512_CLX': 'X86_V4', 'AVX512_CNL': 'X86_V4',
}

X86_FEATURES = {
  'X86_V2': X86_V2, 'X86_V3': X86_V3, 'X86_V4': X86_V4,
  'AVX512_ICL': AVX512_ICL, 'AVX512_SPR': AVX512_SPR
}
