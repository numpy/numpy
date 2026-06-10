source_root = meson.project_source_root()
current_dir = meson.current_source_dir()
mod_features = import('features')

E2K_SSE = mod_features.new(
  'E2K_SSE', 1, args: ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1',
                       '-msse4.2', '-mpopcnt', '-mfpmath=sse',
                       '-DHWY_WANT_SSE4', '-DHWY_DISABLE_PCLMUL_AES'],
  # Adds compiler definitions `NPY_HAVE_SSE*`
  group: ['SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE41', 'SSE42', 'POPCNT'],
  detect: 'E2K_SSE',
  # reuse the x86 features testing code
  test_code: files(current_dir + '/../x86/test_x86_v2.c')[0],
)
E2K_AVX = mod_features.new(
  'E2K_AVX', 10, args: ['-mavx', '-mavx2', '-mfma', '-mbmi', '-mbmi2',
                        '-mlzcnt', '-mf16c'],
  implies: E2K_SSE,
  # Adds compiler definitions `NPY_HAVE_SSE*`
  group: ['AVX', 'AVX2', 'FMA3', 'BMI', 'BMI2', 'LZCNT', 'F16C'],
  detect: 'E2K_AVX',
  test_code: files(current_dir + '/../x86/test_x86_v3.c')[0],
)

E2K_FEATURES = {
  'E2K_SSE': E2K_SSE, 'E2K_AVX': E2K_AVX
}
