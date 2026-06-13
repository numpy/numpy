source_root = meson.project_source_root()
current_dir = meson.current_source_dir()
mod_features = import('features')

HWY_SSE4_FLAGS = ['-DHWY_WANT_SSE4', '-DHWY_DISABLE_PCLMUL_AES']
X86_64_V2_FLAGS = ['-mfpmath=sse']

X86_V2 = mod_features.new(
  'X86_V2', 1, args: ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2',
                      '-mpopcnt'] + X86_64_V2_FLAGS + HWY_SSE4_FLAGS,
  # Adds compiler definitions `NPY_HAVE_SSE*` 
  group: ['SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE41', 'SSE42', 'POPCNT'],
  detect: 'X86_V2',
  test_code: files(current_dir + '/../x86/test_x86_v2.c')[0],
)
X86_V3 = mod_features.new(
  'X86_V3', 10, implies: X86_V2,
  args: ['-mavx', '-mavx2', '-mfma', '-mbmi', '-mbmi2', '-mlzcnt', '-mf16c'],
  group: ['AVX', 'AVX2', 'FMA3', 'BMI', 'BMI2', 'LZCNT', 'F16C'],
  detect: 'X86_V3',
  test_code: files(current_dir + '/../x86/test_x86_v3.c')[0],
)

E2K_FEATURES = {
  'X86_V2': X86_V2, 'X86_V3': X86_V3
}
