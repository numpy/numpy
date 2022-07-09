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
################### A user ctx mgr to turn warnings to errors ###################
'''
from contextlib import contextmanager
import warnings

@contextmanager
def warnings_errors(test):
    """Temporarily turn (almost) all warnings to errors.

    Some functions are allowed to emit specific warnings though;
    these are listed explicitly.
    """
    from scipy import integrate
    known_warnings = {
        'scipy.linalg.norm':
            dict(category=RuntimeWarning, message='divide by zero'),
        'scipy.ndimage.center_of_mass':
            dict(category=RuntimeWarning, message='divide by zero'),
        'scipy.signal.bohman':
            dict(category=RuntimeWarning, message='divide by zero'),
        'scipy.signal.windows.bohman':
            dict(category=RuntimeWarning, message='divide by zero'),
        'scipy.signal.cosine':
            dict(category=RuntimeWarning, message='divide by zero'),
        'scipy.signal.windows.cosine':
            dict(category=RuntimeWarning, message='divide by zero'),
        'scipy.stats.anderson_ksamp':
            dict(category=UserWarning, message='p-value capped:'),
        'scipy.special.ellip_normal':
            dict(category=integrate.IntegrationWarning,
                 message='The occurrence of roundoff'),
        'scipy.special.ellip_harm_2':
            dict(category=integrate.IntegrationWarning,
                 message='The occurrence of roundoff'),
        # tutorials
        'linalg.rst':
            dict(message='the matrix subclass is not',
                 category=PendingDeprecationWarning),
        'stats.rst':
            dict(message='The maximum number of subdivisions',
                 category=integrate.IntegrationWarning),
    }

    with warnings.catch_warnings(record=True) as w:
        if test.name in known_warnings:
            warnings.filterwarnings('ignore', **known_warnings[test.name])      
            yield
        else:
            warnings.simplefilter('error', Warning)
            yield
'''

config = DTConfig()
config.rndm_markers.add('#uninitialized')
config.rndm_markers.add('# uninitialized')
config.optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS


# printing the Romberg extrap table is flaky, adds blanklines on some platforms
##config.stopwords.add('integrate.romb(y, show=True)')


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
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    tut_path = os.path.join(base_dir, 'doc', 'source', 'tutorial', '*.rst')
    sys.stderr.write('\nChecking tutorial files at %s:\n'
                     % os.path.relpath(tut_path, os.getcwd()))

    tutorials = [f for f in sorted(glob.glob(tut_path))]

    # set up scipy-specific config
    config.pseudocode = set(['integrate.nquad(func,'])

    io_matfiles = glob.glob(os.path.join(tut_path.replace('.rst', '.mat')))
    config.local_resources = {'io.rst': io_matfiles}


    all_success = True
    for filename in tutorials:
        sys.stderr.write('\n' + filename + '\n')
        sys.stderr.write("="*len(filename) + '\n')

        result, history = testfile(filename, module_relative=False,
                                    verbose=verbose, raise_on_error=fail_fast,
                                    report=True, config=config)
        all_success = all_success and (result.failed == 0)
    return all_success


def main(args):
    if args.submodule and args.filename:
        raise ValueError("Specify either a submodule or a single file,"
                         " not both.")

    if args.filename:
        all_success = doctest_single_file(args.filename,
                                          verbose=args.verbose,
                                          fail_fast=args.fail_fast)
    else:
        name = args.submodule   # XXX : dance w/ subsubmodules : cluster.vq etc
        submodule_names = [name]  if name else list(PUBLIC_SUBMODULES)
        all_success = doctest_submodules(submodule_names,
                                         verbose=args.verbose,
                                         fail_fast=args.fail_fast)

        # if full run: also check the tutorial
        if not args.submodule:
            tut_success = doctest_tutorial(verbose=args.verbose,
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
    args = parser.parse_args()

    main(args)
