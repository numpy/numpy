source_root = meson.project_source_root()
mod_features = import('features')

LSX = mod_features.new(
  'LSX', 1, args: ['-mlsx'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_lsx.c')[0]
)
LOONGARCH64_FEATURES = {'LSX': LSX}
