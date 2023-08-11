source_root = meson.project_source_root()
mod_features = import('features')

VX = mod_features.new(
  'VX', 1, args: ['-mzvector', '-march=arch11'],
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vx.c')[0],
)
VXE = mod_features.new(
  'VXE', 2, implies: VX, args: {'val': '-march=arch12', 'match': '-march=.*'},
  detect: {'val': 'VXE', 'match': 'VX'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vxe.c')[0],
)
VXE2 = mod_features.new(
  'VXE2', 3, implies: VXE, args: {'val': '-march=arch13', 'match': '-march=.*'},
  detect: {'val': 'VXE2', 'match': 'VX.*'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vxe2.c')[0],
)
S390X_FEATURES = {'VX': VX, 'VXE': VXE, 'VXE2': VXE2}
