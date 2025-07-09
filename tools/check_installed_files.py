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

import glob
import json
import os
import sys

CUR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(CUR_DIR)
NUMPY_DIR = os.path.join(ROOT_DIR, 'numpy')


# Files whose installation path will be different from original one
changed_installed_path = {
    #'numpy/_build_utils/some_file.py': 'numpy/lib/some_file.py'
}


def main(install_dir, tests_check):
    INSTALLED_DIR = os.path.join(ROOT_DIR, install_dir)
    if not os.path.exists(INSTALLED_DIR):
        raise ValueError(
            f"Provided install dir {INSTALLED_DIR} does not exist"
        )

    numpy_test_files = get_files(NUMPY_DIR, kind='test')
    installed_test_files = get_files(INSTALLED_DIR, kind='test')

    if tests_check == "--no-tests":
        if len(installed_test_files) > 0:
            raise Exception("Test files aren't expected to be installed in %s"
                        ", found %s" % (INSTALLED_DIR, installed_test_files))
        print("----------- No test files were installed --------------")
    else:
        # Check test files detected in repo are installed
        for test_file in numpy_test_files.keys():
            if test_file not in installed_test_files.keys():
                raise Exception(
                    f"{numpy_test_files[test_file]} is not installed"
                )

        print("----------- All the test files were installed --------------")

    numpy_pyi_files = get_files(NUMPY_DIR, kind='stub')
    installed_pyi_files = get_files(INSTALLED_DIR, kind='stub')

    # Check *.pyi files detected in repo are installed
    for pyi_file in numpy_pyi_files.keys():
        if pyi_file not in installed_pyi_files.keys():
            if (tests_check == "--no-tests" and
                    "tests" in numpy_pyi_files[pyi_file]):
                continue
            raise Exception(f"{numpy_pyi_files[pyi_file]} is not installed")

    print("----------- All the necessary .pyi files "
          "were installed --------------")


def get_files(dir_to_check, kind='test'):
    files = {}
    patterns = {
        'test': f'{dir_to_check}/**/test_*.py',
        'stub': f'{dir_to_check}/**/*.pyi',
    }
    for path in glob.glob(patterns[kind], recursive=True):
        relpath = os.path.relpath(path, dir_to_check)
        files[relpath] = path

    if sys.version_info >= (3, 12):
        files = {
            k: v for k, v in files.items() if not k.startswith('distutils')
        }

    # ignore python files in vendored pythoncapi-compat submodule
    files = {
        k: v for k, v in files.items() if 'pythoncapi-compat' not in k
    }

    return files


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Incorrect number of input arguments, need "
                         "check_installation.py relpath/to/installed/numpy")

    install_dir = sys.argv[1]
    tests_check = ""
    if len(sys.argv) >= 3:
        tests_check = sys.argv[2]
    main(install_dir, tests_check)

    all_tags = set()

    with open(os.path.join('build', 'meson-info',
                           'intro-install_plan.json'), 'r') as f:
        targets = json.load(f)

    for key in targets.keys():
        for values in list(targets[key].values()):
            if values['tag'] not in all_tags:
                all_tags.add(values['tag'])

    if all_tags != {'runtime', 'python-runtime', 'devel', 'tests'}:
        raise AssertionError(f"Found unexpected install tag: {all_tags}")
