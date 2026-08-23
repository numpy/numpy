project(
  'checks',
  'c', 'cython',
  meson_version: '>=1.8.3',
)

py = import('python').find_installation(pure: false)

cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')

# Keep synced with pyproject.toml
if not cy.version().version_compare('>=3.0.6')
  error('tests requires Cython >= 3.0.6')
endif

npy_include_path = run_command(py, [
    '-c',
    'import os; os.chdir(".."); import numpy; print(os.path.abspath(numpy.get_include()))'
    ], check: true).stdout().strip()

npy_path = run_command(py, [
    '-c',
    'import os; os.chdir(".."); import numpy; print(os.path.dirname(numpy.__file__).removesuffix("numpy"))'
    ], check: true).stdout().strip()

# TODO: This is a hack due to https://github.com/cython/cython/issues/5820,
# where cython may not find the right __init__.pyd file.
add_project_arguments('-I', npy_path, language : 'cython')

# Build limited API extension modules for all supported Python and NumPy versions
# The _PY_ABI3_VERSIONS and _NPY_TARGET_VERSIONS lists should be kept in sync
# with the lists defined in test_limited_api.py, and the test should be updated if new versions are added here.
# The special 'default' entry builds without defining NPY_TARGET_VERSION at all,
# which exercises the path where numpyconfig.h picks the current API version.
_PY_ABI3_VERSIONS = ['3.9', '3.10', '3.11', '3.12', '3.13', '3.14', '3.15']
_NPY_TARGET_VERSIONS = ['2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 'default']

if py.get_variable('Py_GIL_DISABLED', 0) == 0
  foreach py_ver : _PY_ABI3_VERSIONS
    # skip abi3 version higher than the current interpreter version
    if py.language_version().version_compare('>=' + py_ver)
      foreach npy_ver : _NPY_TARGET_VERSIONS
        # 'default' leaves NPY_TARGET_VERSION undefined
        if npy_ver == 'default'
          npy_target_args = []
        else
          npy_macro = 'NPY_' + npy_ver.replace('.', '_') + '_API_VERSION'
          npy_target_args = ['-DNPY_TARGET_VERSION=' + npy_macro]
        endif
        modname = 'limited_api_' + py_ver.replace('.', '_') + '_npy' + npy_ver.replace('.', '_')
        py.extension_module(
            modname,
            'limited_api.c',
            c_args: [
              '-DNPY_NO_DEPRECATED_API=NPY_1_21_API_VERSION',
              '-DMODULE_NAME=' + modname,
            ] + npy_target_args,
            include_directories: [npy_include_path],
            limited_api: py_ver,
        )
        modname = 'limited_api_cython_' + py_ver.replace('.', '_') + '_npy' + npy_ver.replace('.', '_')
        py.extension_module(
            modname,
            'limited_api_cython.pyx',
            install: false,
            cython_args: ['--module-name', modname],
            c_args: [
              '-DNPY_NO_DEPRECATED_API=NPY_1_21_API_VERSION',
            ] + npy_target_args,
            include_directories: [npy_include_path],
            limited_api: py_ver,
        )
      endforeach
    endif
  endforeach
endif

if (
     py.language_version().version_compare('>=3.15') and
     (host_machine.system() != 'windows' or (py.get_variable('Py_GIL_DISABLED', 0) == 1))
)
  py.extension_module(
      'limited_api_opaque',
      'limited_api.c',
      c_args: [
        '-DMODULE_NAME=limited_api_opaque',
        '-DNPY_TARGET_VERSION=NPY_2_5_API_VERSION',
        '-DPy_TARGET_ABI3T=0x030F0000',
      ],
      include_directories: [npy_include_path],
      limited_api: '3.15',
  )
endif
