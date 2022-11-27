"""
Check if all the test and .pyi files are installed after building.

Examples::

    $ python check_installed_files.py install_dirname

        install_dirname:
            the relative path to the directory where NumPy is installed after
            building and running `meson install`.

Notes
=====

The script will stop on encountering the first missing file in the install dir,
it will not give a full listing. This should be okay, because the script is
meant for use in CI so it's not like many files will be missing at once.

"""

import os
import glob
import sys


CUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(CUR_DIR)
NUMPY_DIR = os.path.join(ROOT_DIR, 'numpy')


# Files whose installation path will be different from original one
changed_installed_path = {
    #'numpy/_build_utils/some_file.py': 'numpy/lib/some_file.py'
}


def main(install_dir):
    INSTALLED_DIR = os.path.join(ROOT_DIR, install_dir)
    if not os.path.exists(INSTALLED_DIR):
        raise ValueError(
            f"Provided install dir {INSTALLED_DIR} does not exist"
        )

    numpy_test_files = get_files(NUMPY_DIR, kind='test')
    installed_test_files = get_files(INSTALLED_DIR, kind='test')

    # Check test files detected in repo are installed
    for test_file in numpy_test_files.keys():
        if test_file not in installed_test_files.keys():
            raise Exception(
                "%s is not installed" % numpy_test_files[test_file]
            )

    print("----------- All the test files were installed --------------")

    numpy_pyi_files = get_files(NUMPY_DIR, kind='stub')
    installed_pyi_files = get_files(INSTALLED_DIR, kind='stub')

    # Check *.pyi files detected in repo are installed
    for pyi_file in numpy_pyi_files.keys():
        if pyi_file not in installed_pyi_files.keys():
            raise Exception("%s is not installed" % numpy_pyi_files[pyi_file])

    print("----------- All the .pyi files were installed --------------")


def get_files(dir_to_check, kind='test'):
    files = dict()
    patterns = {
        'test': f'{dir_to_check}/**/test_*.py',
        'stub': f'{dir_to_check}/**/*.pyi',
    }
    for path in glob.glob(patterns[kind], recursive=True):
        relpath = os.path.relpath(path, dir_to_check)
        files[relpath] = path

    return files


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        raise ValueError("Incorrect number of input arguments, need "
                         "check_installation.py relpath/to/installed/numpy")

    install_dir = sys.argv[1]
    main(install_dir)
