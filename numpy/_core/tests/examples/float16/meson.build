project('float16_checks', 'c')

py = import('python').find_installation(pure: false)
cc = meson.get_compiler('c')

# find NumPy's core include dir
npy_include_path = run_command(
    py,
    [
        '-c',
        'import os; os.chdir(".."); import numpy; print(os.path.abspath(numpy.get_include()))',
    ],
    check: true,
).stdout().strip()

py.extension_module(
    '_float16_tests',
    '_float16_tests.c',
    install: false,
    include_directories: [npy_include_path],
)
