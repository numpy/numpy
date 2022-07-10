import os
import glob
import sys
import importlib
import doctest

from scpdt import testmod, testfile, DTConfig
from scpdt._util import get_all_list


BASE_MODULE = "numpy"

PUBLIC_SUBMODULES = [
    'core',
    'f2py',
    'linalg',
    'lib',
    'lib.recfunctions',
    'fft',
    'ma',
    'polynomial',
    'matrixlib',
    'random',
    'testing',
]

# these tutorials fail doctesting
RST_SKIPLIST = ['c-info.ufunc-tutorial.rst',
               'basics.subclassing.rst',
               'basics.interoperability.rst',
               'absolute_beginners.rst',
               'misc.rst']

################### Numpy-specific user configuration ###################
config = DTConfig()
config.rndm_markers.add('#uninitialized')
config.rndm_markers.add('# uninitialized')
config.optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

config.skiplist = [
    # cases where NumPy docstrings import things from SciPy:
    'numpy.lib.vectorize',
    'numpy.random.standard_gamma',
    'numpy.random.gamma',
    'numpy.random.vonmises',
    'numpy.random.power',
    'numpy.random.zipf',
    # cases where NumPy docstrings import things from other 3'rd party libs:
    'numpy.core.from_dlpack',
    # remote / local file IO with DataSource is problematic in doctest:
    'numpy.lib.DataSource',
    'numpy.lib.Repository',
]

if sys.version_info < (3, 9):
    config.skiplist += [
        "numpy.core.ndarray.__class_getitem__",
        "numpy.core.dtype.__class_getitem__",
        "numpy.core.number.__class_getitem__",
    ]

config.skiplist += RST_SKIPLIST

############################################################################

LOGFILE = open('doctest.log', 'a')


def doctest_submodules(module_names, verbose, fail_fast):
    all_success = True
    for submodule_name in module_names:
        prefix = BASE_MODULE + '.'
        if not submodule_name.startswith(prefix):
            module_name = prefix + submodule_name
        else:
            module_name = submodule_name

        module = importlib.import_module(module_name)

        full_name = module.__name__
        line = '='*len(full_name)
        sys.stderr.write(f"\n\n{line}\n")
        sys.stderr.write(full_name)
        sys.stderr.write(f"\n{line}\n")

        result, history = testmod(module, strategy='api',
                                  verbose=verbose,
                                  raise_on_error=fail_fast, config=config)

        LOGFILE.write(module_name + '\n')
        LOGFILE.write("="*len(module_name)  + '\n')
        for entry in history:
            LOGFILE.write(str(entry) + '\n')

        sys.stderr.write(str(result))
        all_success = all_success and (result.failed == 0)
    return all_success


def doctest_single_file(fname, verbose, fail_fast):
    result, history = testfile(fname, config=config, module_relative=False,
                               verbose=verbose, raise_on_error=fail_fast)
    return result.failed == 0


def doctest_tutorial(verbose, fail_fast):
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '..', 'doc', 'source')
    user_path = os.path.relpath(os.path.join(base_dir, 'user'))
    user_rst = glob.glob(os.path.join(user_path, '*rst'))

    dev_path = os.path.relpath(os.path.join(base_dir, 'dev'))
    dev_rst = glob.glob(os.path.join(dev_path, '*rst'))

    all_success = True
    for filename in dev_rst + user_rst:
        sys.stderr.write('\n' + filename + '\n')
        sys.stderr.write("="*len(filename) + '\n')
        if os.path.split(filename)[1] in config.skiplist:
            sys.stderr.write(f'skipping {filename}\n')
            continue

        result, history = testfile(filename, module_relative=False,
                                    verbose=verbose, raise_on_error=fail_fast,
                                    report=True, config=config)
        all_success = all_success and (result.failed == 0)
    return all_success


def main(args):
    if args.submodule and args.filename:
        raise ValueError("Specify either a submodule or a single file,"
                         " not both.")

    tut_success = True
    if args.rst:
        tut_success = doctest_tutorial(verbose=args.verbose,
                                       fail_fast=args.fail_fast)

    all_success = True
    if args.filename:
        all_success = doctest_single_file(args.filename,
                                          verbose=args.verbose,
                                          fail_fast=args.fail_fast)
    else:
        name = args.submodule
        submodule_names = [name]  if name else list(PUBLIC_SUBMODULES)
        all_success = doctest_submodules(submodule_names,
                                         verbose=args.verbose,
                                         fail_fast=args.fail_fast)
    
    all_success = all_success and tut_success

    # final report
    if all_success:
        sys.stderr.write('\n\n>>>> OK: doctests PASSED\n')
        sys.exit(0)
    else:
        sys.stderr.write('\n\n>>>> ERROR: doctests FAILED\n')
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="doctest runner")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='print verbose (`-v`) or very verbose (`-vv`) '
                              'output for all tests')
    parser.add_argument('-x', '--fail-fast', action='store_true',
                        help=('stop running tests after first failure'))
    parser.add_argument( "-s", "--submodule", default=None,
                        help="Submodule whose tests to run (cluster,"
                             " constants, ...)")
    parser.add_argument( "-t", "--filename", default=None,
                        help="Specify a .py file to check")
    parser.add_argument( "-r", "--rst", action='store_true',
                        help="Check *rst tutorials.")
    args = parser.parse_args()

    main(args)
