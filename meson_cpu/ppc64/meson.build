source_root = meson.project_source_root()
mod_features = import('features')
compiler_id = meson.get_compiler('c').get_id()

VSX = mod_features.new(
  'VSX', 1, args: '-mvsx',
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vsx.c')[0],
  extra_tests: {
    'VSX_ASM': files(source_root + '/numpy/distutils/checks/extra_vsx_asm.c')[0]
  }
)
if compiler_id == 'clang'
  VSX.update(args: ['-mvsx', '-maltivec'])
endif
VSX2 = mod_features.new(
  'VSX2', 2, implies: VSX, args: {'val': '-mcpu=power8', 'match': '.*vsx'},
  detect: {'val': 'VSX2', 'match': 'VSX'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vsx2.c')[0],
)
# VSX2 is hardware baseline feature on ppc64le since the first little-endian
# support was part of Power8
if host_machine.endian() == 'little'
  VSX.update(implies: VSX2)
endif
VSX3 = mod_features.new(
  'VSX3', 3, implies: VSX2, args: {'val': '-mcpu=power9', 'match': '.*[mcpu=|vsx].*'},
  detect: {'val': 'VSX3', 'match': 'VSX.*'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vsx3.c')[0],
)
VSX4 = mod_features.new(
  'VSX4', 4, implies: VSX3, args: {'val': '-mcpu=power10', 'match': '.*[mcpu=|vsx].*'},
  detect: {'val': 'VSX4', 'match': 'VSX.*'},
  test_code: files(source_root + '/numpy/distutils/checks/cpu_vsx3.c')[0],
  extra_tests: {
    'VSX4_MMA': files(source_root + '/numpy/distutils/checks/extra_vsx4_mma.c')[0]
  }
)
PPC64_FEATURES = {'VSX': VSX, 'VSX2': VSX2, 'VSX3': VSX3, 'VSX4': VSX4}
