project('f2py_examples', 'c',
  version : '0.1',
  default_options : ['warning_level=2'])

add_languages('fortran')

py_mod = import('python')
py3 = py_mod.find_installation('python3')
py3_dep = py3.dependency()
message(py3.path())
message(py3.get_install_dir())

incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

py3.extension_module('fib2',
           'fib1.f',
           'fib2module.c',
           incdir_f2py+'/fortranobject.c',
           include_directories: inc_np,
           dependencies : py3_dep,
           install : true)