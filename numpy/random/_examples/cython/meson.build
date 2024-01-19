project('random-build-examples', 'c', 'cpp', 'cython')

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)

cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')

# Keep synced with pyproject.toml
if not cy.version().version_compare('>=3.0.6')
  error('tests requires Cython >= 3.0.6')
endif

_numpy_abs = run_command(py3, ['-c',
               'import os; os.chdir(".."); import numpy; print(os.path.abspath(numpy.get_include() + "../../.."))'],
                         check: true).stdout().strip()

npy_include_path = _numpy_abs / '_core' / 'include'
libnpyrandom_sources = [
  _numpy_abs / 'random/src/distributions/logfactorial.c',
  _numpy_abs / 'random/src/distributions/distributions.c',
  _numpy_abs / 'random/src/distributions/random_mvhg_count.c',
  _numpy_abs / 'random/src/distributions/random_mvhg_marginals.c',
  _numpy_abs / 'random/src/distributions/random_hypergeometric.c',
]
libnpyrandom_dep = declare_dependency(sources: libnpyrandom_sources, include_directories: npy_include_path)

py3.extension_module(
    'extending_distributions',
    'extending_distributions.pyx',
    dependencies: [libnpyrandom_dep],
)
py3.extension_module(
    'extending',
    'extending.pyx',
    dependencies: [libnpyrandom_dep],
)
py3.extension_module(
    'extending_cpp',
    'extending_distributions.pyx',
    override_options : ['cython_language=cpp'],
    cython_args: ['--module-name', 'extending_cpp'],
    dependencies: [libnpyrandom_dep],
)
