#!/usr/bin/env python
"""
runtests.py [OPTIONS] [-- ARGS]

Run tests, building the project first.

Examples::

    $ python runtests.py
    $ python runtests.py -s {SAMPLE_SUBMODULE}
    $ python runtests.py -t {SAMPLE_TEST}
    $ python runtests.py --ipython
    $ python runtests.py --python somescript.py
    $ python runtests.py --bench
    $ python runtests.py --durations 20

Run a debugger:

    $ gdb --args python runtests.py [...other args...]

Generate C code coverage listing under build/lcov/:
(requires http://ltp.sourceforge.net/coverage/lcov.php)

    $ python runtests.py --gcov [...other args...]
    $ python runtests.py --lcov-html

"""
from __future__ import division, print_function

#
# This is a generic test runner script for projects using NumPy's test
# framework. Change the following values to adapt to your project:
#

PROJECT_MODULE = "numpy"
PROJECT_ROOT_FILES = ['numpy', 'LICENSE.txt', 'setup.py']
SAMPLE_TEST = "numpy/linalg/tests/test_linalg.py::test_byteorder_check"
SAMPLE_SUBMODULE = "linalg"

EXTRA_PATH = ['/usr/lib/ccache', '/usr/lib/f90cache',
              '/usr/local/lib/ccache', '/usr/local/lib/f90cache']

# ---------------------------------------------------------------------


if __doc__ is None:
    __doc__ = "Run without -OO if you want usage info"
else:
    __doc__ = __doc__.format(**globals())


import sys
import os

# In case we are run from the source directory, we don't want to import the
# project from there:
sys.path.pop(0)

import shutil
import subprocess
import time
from argparse import ArgumentParser, REMAINDER

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("--verbose", "-v", action="count", default=1,
                        help="more verbosity")
    parser.add_argument("--no-build", "-n", action="store_true", default=False,
                        help="do not build the project (use system installed version)")
    parser.add_argument("--build-only", "-b", action="store_true", default=False,
                        help="just build, do not run any tests")
    parser.add_argument("--doctests", action="store_true", default=False,
                        help="Run doctests in module")
    #parser.add_argument("--refguide-check", action="store_true", default=False,
                        #help="Run refguide check (do not run regular tests.)")
    parser.add_argument("--coverage", action="store_true", default=False,
                        help=("report coverage of project code. HTML output goes "
                              "under build/coverage"))
    parser.add_argument("--durations", action="store", default=-1, type=int,
                        help=("Time N slowest tests, time all if 0, time none if < 0"))
    parser.add_argument("--gcov", action="store_true", default=False,
                        help=("enable C code coverage via gcov (requires GCC). "
                              "gcov output goes to build/**/*.gc*"))
    parser.add_argument("--lcov-html", action="store_true", default=False,
                        help=("produce HTML for C code coverage information "
                              "from a previous run with --gcov. "
                              "HTML output goes to build/lcov/"))
    parser.add_argument("--mode", "-m", default="fast",
                        help="'fast', 'full', or something that could be "
                             "passed to nosetests -A [default: fast]")
    parser.add_argument("--submodule", "-s", default=None,
                        help="Submodule whose tests to run (cluster, constants, ...)")
    parser.add_argument("--pythonpath", "-p", default=None,
                        help="Paths to prepend to PYTHONPATH")
    parser.add_argument("--tests", "-t", action='append',
                        help="Specify tests to run")
    parser.add_argument("--python", action="store_true",
                        help="Start a Python shell with PYTHONPATH set")
    parser.add_argument("--ipython", "-i", action="store_true",
                        help="Start IPython shell with PYTHONPATH set")
    parser.add_argument("--shell", action="store_true",
                        help="Start Unix shell with PYTHONPATH set")
    parser.add_argument("--debug", "-g", action="store_true",
                        help="Debug build")
    parser.add_argument("--parallel", "-j", type=int, default=0,
                        help="Number of parallel jobs during build")
    parser.add_argument("--show-build-log", action="store_true",
                        help="Show build output rather than using a log file")
    parser.add_argument("--bench", action="store_true",
                        help="Run benchmark suite instead of test suite")
    parser.add_argument("--bench-compare", action="store", metavar="COMMIT",
                        help=("Compare benchmark results of current HEAD to "
                              "BEFORE. Use an additional "
                              "--bench-compare=COMMIT to override HEAD with "
                              "COMMIT. Note that you need to commit your "
                              "changes first!"))
    parser.add_argument("args", metavar="ARGS", default=[], nargs=REMAINDER,
                        help="Arguments to pass to Nose, Python or shell")
    args = parser.parse_args(argv)

    if args.durations < 0:
        args.durations = -1

    if args.bench_compare:
        args.bench = True
        args.no_build = True # ASV does the building

    if args.lcov_html:
        # generate C code coverage output
        lcov_generate()
        sys.exit(0)

    if args.pythonpath:
        for p in reversed(args.pythonpath.split(os.pathsep)):
            sys.path.insert(0, p)

    if args.gcov:
        gcov_reset_counters()

    if args.debug and args.bench:
        print("*** Benchmarks should not be run against debug "
              "version; remove -g flag ***")

    if not args.no_build:
        # we need the noarch path in case the package is pure python.
        site_dir, site_dir_noarch = build_project(args)
        sys.path.insert(0, site_dir)
        sys.path.insert(0, site_dir_noarch)
        os.environ['PYTHONPATH'] = site_dir + os.pathsep + site_dir_noarch
    else:
        _temp = __import__(PROJECT_MODULE)
        site_dir = os.path.sep.join(_temp.__file__.split(os.path.sep)[:-2])

    extra_argv = args.args[:]
    if extra_argv and extra_argv[0] == '--':
        extra_argv = extra_argv[1:]

    if args.python:
        # Debugging issues with warnings is much easier if you can see them
        print("Enabling display of all warnings")
        import warnings
        import types

        warnings.filterwarnings("always")
        if extra_argv:
            # Don't use subprocess, since we don't want to include the
            # current path in PYTHONPATH.
            sys.argv = extra_argv
            with open(extra_argv[0], 'r') as f:
                script = f.read()
            sys.modules['__main__'] = types.ModuleType('__main__')
            ns = dict(__name__='__main__',
                      __file__=extra_argv[0])
            exec_(script, ns)
            sys.exit(0)
        else:
            import code
            code.interact()
            sys.exit(0)

    if args.ipython:
        # Debugging issues with warnings is much easier if you can see them
        print("Enabling display of all warnings and pre-importing numpy as np")
        import warnings; warnings.filterwarnings("always")
        import IPython
        import numpy as np
        IPython.embed(user_ns={"np": np})
        sys.exit(0)

    if args.shell:
        shell = os.environ.get('SHELL', 'cmd' if os.name == 'nt' else 'sh')
        print("Spawning a shell ({})...".format(shell))
        subprocess.call([shell] + extra_argv)
        sys.exit(0)

    if args.coverage:
        dst_dir = os.path.join(ROOT_DIR, 'build', 'coverage')
        fn = os.path.join(dst_dir, 'coverage_html.js')
        if os.path.isdir(dst_dir) and os.path.isfile(fn):
            shutil.rmtree(dst_dir)
        extra_argv += ['--cov-report=html:' + dst_dir]

    if args.bench:
        # Run ASV
        items = extra_argv
        if args.tests:
            items += args.tests
        if args.submodule:
            items += [args.submodule]

        bench_args = []
        for a in items:
            bench_args.extend(['--bench', a])

        if not args.bench_compare:
            cmd = ['asv', 'run', '-n', '-e', '--python=same'] + bench_args
            ret = subprocess.call(cmd, cwd=os.path.join(ROOT_DIR, 'benchmarks'))
            sys.exit(ret)
        else:
            commits = [x.strip() for x in args.bench_compare.split(',')]
            if len(commits) == 1:
                commit_a = commits[0]
                commit_b = 'HEAD'
            elif len(commits) == 2:
                commit_a, commit_b = commits
            else:
                p.error("Too many commits to compare benchmarks for")

            # Check for uncommitted files
            if commit_b == 'HEAD':
                r1 = subprocess.call(['git', 'diff-index', '--quiet',
                                      '--cached', 'HEAD'])
                r2 = subprocess.call(['git', 'diff-files', '--quiet'])
                if r1 != 0 or r2 != 0:
                    print("*"*80)
                    print("WARNING: you have uncommitted changes --- "
                          "these will NOT be benchmarked!")
                    print("*"*80)

            # Fix commit ids (HEAD is local to current repo)
            out = subprocess.check_output(['git', 'rev-parse', commit_b])
            commit_b = out.strip().decode('ascii')

            out = subprocess.check_output(['git', 'rev-parse', commit_a])
            commit_a = out.strip().decode('ascii')

            cmd = ['asv', 'continuous', '-e', '-f', '1.05',
                   commit_a, commit_b] + bench_args
            ret = subprocess.call(cmd, cwd=os.path.join(ROOT_DIR, 'benchmarks'))
            sys.exit(ret)

    test_dir = os.path.join(ROOT_DIR, 'build', 'test')

    if args.build_only:
        sys.exit(0)
    else:
        __import__(PROJECT_MODULE)
        test = sys.modules[PROJECT_MODULE].test

    if args.submodule:
        tests = [PROJECT_MODULE + "." + args.submodule]
    elif args.tests:
        tests = args.tests
    else:
        tests = None


    # Run the tests under build/test

    if not args.no_build:
        test_dir = site_dir
    else:
        test_dir = os.path.join(ROOT_DIR, 'build', 'test')
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    shutil.copyfile(os.path.join(ROOT_DIR, '.coveragerc'),
                    os.path.join(test_dir, '.coveragerc'))

    cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        result = test(args.mode,
                      verbose=args.verbose,
                      extra_argv=extra_argv,
                      doctests=args.doctests,
                      coverage=args.coverage,
                      durations=args.durations,
                      tests=tests)
    finally:
        os.chdir(cwd)

    if isinstance(result, bool):
        sys.exit(0 if result else 1)
    elif result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


def build_project(args):
    """
    Build a dev version of the project.

    Returns
    -------
    site_dir
        site-packages directory where it was installed

    """

    import distutils.sysconfig

    root_ok = [os.path.exists(os.path.join(ROOT_DIR, fn))
               for fn in PROJECT_ROOT_FILES]
    if not all(root_ok):
        print("To build the project, run runtests.py in "
              "git checkout or unpacked source")
        sys.exit(1)

    dst_dir = os.path.join(ROOT_DIR, 'build', 'testenv')

    env = dict(os.environ)
    cmd = [sys.executable, 'setup.py']

    # Always use ccache, if installed
    env['PATH'] = os.pathsep.join(EXTRA_PATH + env.get('PATH', '').split(os.pathsep))
    cvars = distutils.sysconfig.get_config_vars()
    compiler = env.get('CC') or cvars.get('CC', '')
    if 'gcc' in compiler:
        # Check that this isn't clang masquerading as gcc.
        if sys.platform != 'darwin' or 'gnu-gcc' in compiler:
            # add flags used as werrors
            warnings_as_errors = ' '.join([
                # from tools/travis-test.sh
                '-Werror=declaration-after-statement',
                '-Werror=vla',
                '-Werror=nonnull',
                '-Werror=pointer-arith',
                '-Wlogical-op',
                # from sysconfig
                '-Werror=unused-function',
            ])
            env['CFLAGS'] = warnings_as_errors + ' ' + env.get('CFLAGS', '')
    if args.debug or args.gcov:
        # assume everyone uses gcc/gfortran
        env['OPT'] = '-O0 -ggdb'
        env['FOPT'] = '-O0 -ggdb'
        if args.gcov:
            env['OPT'] = '-O0 -ggdb'
            env['FOPT'] = '-O0 -ggdb'
            env['CC'] = cvars['CC'] + ' --coverage'
            env['CXX'] = cvars['CXX'] + ' --coverage'
            env['F77'] = 'gfortran --coverage '
            env['F90'] = 'gfortran --coverage '
            env['LDSHARED'] = cvars['LDSHARED'] + ' --coverage'
            env['LDFLAGS'] = " ".join(cvars['LDSHARED'].split()[1:]) + ' --coverage'

    cmd += ["build"]
    if args.parallel > 1:
        cmd += ["-j", str(args.parallel)]
    # Install; avoid producing eggs so numpy can be imported from dst_dir.
    cmd += ['install', '--prefix=' + dst_dir,
            '--single-version-externally-managed',
            '--record=' + dst_dir + 'tmp_install_log.txt']

    from distutils.sysconfig import get_python_lib
    site_dir = get_python_lib(prefix=dst_dir, plat_specific=True)
    site_dir_noarch = get_python_lib(prefix=dst_dir, plat_specific=False)
    # easy_install won't install to a path that Python by default cannot see
    # and isn't on the PYTHONPATH.  Plus, it has to exist.
    if not os.path.exists(site_dir):
        os.makedirs(site_dir)
    if not os.path.exists(site_dir_noarch):
        os.makedirs(site_dir_noarch)
    env['PYTHONPATH'] = site_dir + ':' + site_dir_noarch

    log_filename = os.path.join(ROOT_DIR, 'build.log')

    if args.show_build_log:
        ret = subprocess.call(cmd, env=env, cwd=ROOT_DIR)
    else:
        log_filename = os.path.join(ROOT_DIR, 'build.log')
        print("Building, see build.log...")
        with open(log_filename, 'w') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                 cwd=ROOT_DIR)
        try:
            # Wait for it to finish, and print something to indicate the
            # process is alive, but only if the log file has grown (to
            # allow continuous integration environments kill a hanging
            # process accurately if it produces no output)
            last_blip = time.time()
            last_log_size = os.stat(log_filename).st_size
            while p.poll() is None:
                time.sleep(0.5)
                if time.time() - last_blip > 60:
                    log_size = os.stat(log_filename).st_size
                    if log_size > last_log_size:
                        print("    ... build in progress")
                        last_blip = time.time()
                        last_log_size = log_size

            ret = p.wait()
        except:
            p.kill()
            p.wait()
            raise

    if ret == 0:
        print("Build OK")
    else:
        if not args.show_build_log:
            with open(log_filename, 'r') as f:
                print(f.read())
            print("Build failed!")
        sys.exit(1)

    return site_dir, site_dir_noarch


#
# GCOV support
#
def gcov_reset_counters():
    print("Removing previous GCOV .gcda files...")
    build_dir = os.path.join(ROOT_DIR, 'build')
    for dirpath, dirnames, filenames in os.walk(build_dir):
        for fn in filenames:
            if fn.endswith('.gcda') or fn.endswith('.da'):
                pth = os.path.join(dirpath, fn)
                os.unlink(pth)

#
# LCOV support
#

LCOV_OUTPUT_FILE = os.path.join(ROOT_DIR, 'build', 'lcov.out')
LCOV_HTML_DIR = os.path.join(ROOT_DIR, 'build', 'lcov')

def lcov_generate():
    try: os.unlink(LCOV_OUTPUT_FILE)
    except OSError: pass
    try: shutil.rmtree(LCOV_HTML_DIR)
    except OSError: pass

    print("Capturing lcov info...")
    subprocess.call(['lcov', '-q', '-c',
                     '-d', os.path.join(ROOT_DIR, 'build'),
                     '-b', ROOT_DIR,
                     '--output-file', LCOV_OUTPUT_FILE])

    print("Generating lcov HTML output...")
    ret = subprocess.call(['genhtml', '-q', LCOV_OUTPUT_FILE,
                           '--output-directory', LCOV_HTML_DIR,
                           '--legend', '--highlight'])
    if ret != 0:
        print("genhtml failed!")
    else:
        print("HTML output generated under build/lcov/")


#
# Python 3 support
#

if sys.version_info[0] >= 3:
    import builtins
    exec_ = getattr(builtins, "exec")
else:
    def exec_(code, globs=None, locs=None):
        """Execute code in a namespace."""
        if globs is None:
            frame = sys._getframe(1)
            globs = frame.f_globals
            if locs is None:
                locs = frame.f_locals
            del frame
        elif locs is None:
            locs = globs
        exec("""exec code in globs, locs""")

if __name__ == "__main__":
    main(argv=sys.argv[1:])
