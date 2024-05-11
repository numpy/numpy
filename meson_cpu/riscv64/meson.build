source_root = meson.project_source_root()
mod_features = import('features')

RVV = mod_features.new(
  'RVV', 1, args: ['-march=rv64gcv'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_rvv.c')[0],
)
RV64_FEATURES = {'RVV': RVV}
