#!/usr/bin/env python3
"""
runtests.py [OPTIONS] [-- ARGS]

Run tests, building the project first.

Examples::

    $ python runtests.py
    $ python runtests.py -s {SAMPLE_SUBMODULE}
    $ # Run a standalone test function:
    $ python runtests.py -t {SAMPLE_TEST}
    $ # Run a test defined as a method of a TestXXX class:
    $ python runtests.py -t {SAMPLE_TEST2}
    $ python runtests.py --ipython
    $ python runtests.py --python somescript.py
    $ python runtests.py --bench
    $ python runtests.py --durations 20

Run a debugger:

    $ gdb --args python runtests.py [...other args...]

Disable pytest capturing of output by using its '-s' option:

    $ python runtests.py -- -s

Generate C code coverage listing under build/lcov/:
(requires http://ltp.sourceforge.net/coverage/lcov.php)

    $ python runtests.py --gcov [...other args...]
    $ python runtests.py --lcov-html

Run lint checks.
Provide target branch name or `uncommitted` to check before committing:

    $ python runtests.py --lint main
    $ python runtests.py --lint uncommitted

"""
#
# This is a generic test runner script for projects using NumPy's test
# framework. Change the following values to adapt to your project:
#

PROJECT_MODULE = "numpy"
PROJECT_ROOT_FILES = ['numpy', 'LICENSE.txt', 'setup.py']
SAMPLE_TEST = "numpy/linalg/tests/test_linalg.py::test_byteorder_check"
SAMPLE_TEST2 = "numpy/core/tests/test_memmap.py::TestMemmap::test_open_with_filename"
SAMPLE_SUBMODULE = "linalg"

EXTRA_PATH = ['/usr/lib/ccache', '/usr/lib/f90cache',
              '/usr/local/lib/ccache', '/usr/local/lib/f90cache']

# ---------------------------------------------------------------------


if __doc__ is None:
    __doc__ = "Run without -OO if you want usage info"
else:
    __doc__ = __doc__.format(**globals())


import sys
import os, glob

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
                        help="Add one verbosity level to pytest. Default is 0")
    parser.add_argument("--debug-info", action="store_true",
                        help=("Add --verbose-cfg to build_src to show "
                              "compiler configuration output while creating "
                              "_numpyconfig.h and config.h"))
    parser.add_argument("--no-build", "-n", action="store_true", default=False,
                        help="Do not build the project (use system installed "
                             "version)")
    parser.add_argument("--build-only", "-b", action="store_true",
                        default=False, help="Just build, do not run any tests")
    parser.add_argument("--doctests", action="store_true", default=False,
                        help="Run doctests in module")
    parser.add_argument("--refguide-check", action="store_true", default=False,
                        help="Run refguide (doctest) check (do not run "
                             "regular tests.)")
    parser.add_argument("--coverage", action="store_true", default=False,
                        help=("Report coverage of project code. HTML output "
                              "goes under build/coverage"))
    parser.add_argument("--lint", default=None,
                        help="'<Target Branch>' or 'uncommitted', passed to "
                             "tools/linter.py [--branch BRANCH] "
                             "[--uncommitted]")
    parser.add_argument("--durations", action="store", default=-1, type=int,
                        help=("Time N slowest tests, time all if 0, time none "
                              "if < 0"))
    parser.add_argument("--gcov", action="store_true", default=False,
                        help=("Enable C code coverage via gcov (requires "
                              "GCC). gcov output goes to build/**/*.gc*"))
    parser.add_argument("--lcov-html", action="store_true", default=False,
                        help=("Produce HTML for C code coverage information "
                              "from a previous run with --gcov. "
                              "HTML output goes to build/lcov/"))
    parser.add_argument("--mode", "-m", default="fast",
                        help="'fast', 'full', or something that could be "
                             "passed to nosetests -A [default: fast]")
    parser.add_argument("--submodule", "-s", default=None,
                        help="Submodule whose tests to run (cluster, "
                             "constants, ...)")
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
    parser.add_argument("--mypy", action="store_true",
                        help="Run mypy on files with NumPy on the MYPYPATH")
    parser.add_argument("--debug", "-g", action="store_true",
                        help="Debug build")
    parser.add_argument("--parallel", "-j", type=int, default=0,
                        help="Number of parallel jobs during build")
    parser.add_argument("--warn-error", action="store_true",
                        help="Set -Werror to convert all compiler warnings to "
                             "errors")
    parser.add_argument("--cpu-baseline", default=None,
                        help="Specify a list of enabled baseline CPU "
                             "optimizations"),
    parser.add_argument("--cpu-dispatch", default=None,
                        help="Specify a list of dispatched CPU optimizations"),
    parser.add_argument("--disable-optimization", action="store_true",
                        help="Disable CPU optimized code (dispatch, simd, "
                             "fast, ...)"),
    parser.add_argument("--simd-test", default=None,
                        help="Specify a list of CPU optimizations to be "
                             "tested against NumPy SIMD interface"),
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
                        help="Arguments to pass to pytest, asv, mypy, Python "
                             "or shell")
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

    if args.lint:
        check_lint(args.lint)

    if not args.no_build:
        # we need the noarch path in case the package is pure python.
        site_dir, site_dir_noarch = build_project(args)
        sys.path.insert(0, site_dir)
        sys.path.insert(0, site_dir_noarch)
        os.environ['PYTHONPATH'] = \
            os.pathsep.join((
                site_dir, 
                site_dir_noarch, 
                os.environ.get('PYTHONPATH', '')
            ))
    else:
        if not args.bench_compare:
            _temp = __import__(PROJECT_MODULE)
            site_dir = os.path.sep.join(_temp.__file__.split(os.path.sep)[:-2])

    extra_argv = args.args[:]
    if not args.bench:
        # extra_argv may also lists selected benchmarks
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
            with open(extra_argv[0]) as f:
                script = f.read()
            sys.modules['__main__'] = types.ModuleType('__main__')
            ns = dict(__name__='__main__',
                      __file__=extra_argv[0])
            exec(script, ns)
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
        IPython.embed(colors='neutral', user_ns={"np": np})
        sys.exit(0)

    if args.shell:
        shell = os.environ.get('SHELL', 'cmd' if os.name == 'nt' else 'sh')
        print("Spawning a shell ({})...".format(shell))
        subprocess.call([shell] + extra_argv)
        sys.exit(0)

    if args.mypy:
        try:
            import mypy.api
        except ImportError:
            raise RuntimeError(
                "Mypy not found. Please install it by running "
                "pip install -r test_requirements.txt from the repo root"
            )

        os.environ['MYPYPATH'] = site_dir
        # By default mypy won't color the output since it isn't being
        # invoked from a tty.
        os.environ['MYPY_FORCE_COLOR'] = '1'

        config = os.path.join(
            site_dir,
            "numpy",
            "typing",
            "tests",
            "data",
            "mypy.ini",
        )

        report, errors, status = mypy.api.run(
            ['--config-file', config] + args.args
        )
        print(report, end='')
        print(errors, end='', file=sys.stderr)
        sys.exit(status)

    if args.coverage:
        dst_dir = os.path.join(ROOT_DIR, 'build', 'coverage')
        fn = os.path.join(dst_dir, 'coverage_html.js')
        if os.path.isdir(dst_dir) and os.path.isfile(fn):
            shutil.rmtree(dst_dir)
        extra_argv += ['--cov-report=html:' + dst_dir]

    if args.refguide_check:
        cmd = [os.path.join(ROOT_DIR, 'tools', 'refguide_check.py'),
               '--doctests']
        if args.verbose:
            cmd += ['-' + 'v'*args.verbose]
        if args.submodule:
            cmd += [args.submodule]
        os.execv(sys.executable, [sys.executable] + cmd)
        sys.exit(0)

    if args.bench:
        # Run ASV
        for i, v in enumerate(extra_argv):
            if v.startswith("--"):
                items = extra_argv[:i]
                if v == "--":
                    i += 1  # skip '--' indicating further are passed on.
                bench_args = extra_argv[i:]
                break
        else:
            items = extra_argv
            bench_args = []

        if args.tests:
            items += args.tests
        if args.submodule:
            items += [args.submodule]
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

            # generate config file with the required build options
            asv_cfpath = [
                '--config', asv_compare_config(
                    os.path.join(ROOT_DIR, 'benchmarks'), args,
                    # to clear the cache if the user changed build options
                    (commit_a, commit_b)
                )
            ]
            cmd = ['asv', 'continuous', '-e', '-f', '1.05',
                   commit_a, commit_b] + asv_cfpath + bench_args
            ret = subprocess.call(cmd, cwd=os.path.join(ROOT_DIR, 'benchmarks'))
            sys.exit(ret)

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

    import sysconfig

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
    cvars = sysconfig.get_config_vars()
    compiler = env.get('CC') or cvars.get('CC', '')
    if 'gcc' in compiler:
        # Check that this isn't clang masquerading as gcc.
        if sys.platform != 'darwin' or 'gnu-gcc' in compiler:
            # add flags used as werrors
            warnings_as_errors = ' '.join([
                # from tools/travis-test.sh
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
    if args.warn_error:
        cmd += ["--warn-error"]
    if args.cpu_baseline:
        cmd += ["--cpu-baseline", args.cpu_baseline]
    if args.cpu_dispatch:
        cmd += ["--cpu-dispatch", args.cpu_dispatch]
    if args.disable_optimization:
        cmd += ["--disable-optimization"]
    if args.simd_test is not None:
        cmd += ["--simd-test", args.simd_test]
    if args.debug_info:
        cmd += ["build_src", "--verbose-cfg"]
    # Install; avoid producing eggs so numpy can be imported from dst_dir.
    cmd += ['install', '--prefix=' + dst_dir,
            '--single-version-externally-managed',
            '--record=' + dst_dir + 'tmp_install_log.txt']

    config_vars = dict(sysconfig.get_config_vars())
    config_vars["platbase"] = dst_dir
    config_vars["base"] = dst_dir

    site_dir_template = os.path.normpath(sysconfig.get_path(
        'platlib', expand=False
    ))
    site_dir = site_dir_template.format(**config_vars)
    noarch_template = os.path.normpath(sysconfig.get_path(
        'purelib', expand=False
    ))
    site_dir_noarch = noarch_template.format(**config_vars)

    # easy_install won't install to a path that Python by default cannot see
    # and isn't on the PYTHONPATH.  Plus, it has to exist.
    if not os.path.exists(site_dir):
        os.makedirs(site_dir)
    if not os.path.exists(site_dir_noarch):
        os.makedirs(site_dir_noarch)
    env['PYTHONPATH'] = \
        os.pathsep.join((site_dir, site_dir_noarch, env.get('PYTHONPATH', '')))

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
            with open(log_filename) as f:
                print(f.read())
            print("Build failed!")
        sys.exit(1)

    # Rebase
    if sys.platform == "cygwin":
        from pathlib import path
        testenv_root = Path(config_vars["platbase"])
        dll_list = testenv_root.glob("**/*.dll")
        rebase_cmd = ["/usr/bin/rebase", "--database", "--oblivious"]
        rebase_cmd.extend(dll_list)
        if subprocess.run(rebase_cmd):
            print("Rebase failed")
            sys.exit(1)

    return site_dir, site_dir_noarch

def asv_compare_config(bench_path, args, h_commits):
    """
    Fill the required build options through custom variable
    'numpy_build_options' and return the generated config path.
    """
    conf_path = os.path.join(bench_path, "asv_compare.conf.json.tpl")
    nconf_path = os.path.join(bench_path, "_asv_compare.conf.json")

    # add custom build
    build = []
    if args.parallel > 1:
        build += ["-j", str(args.parallel)]
    if args.cpu_baseline:
        build += ["--cpu-baseline", args.cpu_baseline]
    if args.cpu_dispatch:
        build += ["--cpu-dispatch", args.cpu_dispatch]
    if args.disable_optimization:
        build += ["--disable-optimization"]

    is_cached = asv_substitute_config(conf_path, nconf_path,
        numpy_build_options = ' '.join([f'\\"{v}\\"' for v in build]),
        numpy_global_options= ' '.join([f'--global-option=\\"{v}\\"' for v in ["build"] + build])
    )
    if not is_cached:
        asv_clear_cache(bench_path, h_commits)
    return nconf_path

def asv_clear_cache(bench_path, h_commits, env_dir="env"):
    """
    Force ASV to clear the cache according to specified commit hashes.
    """
    # FIXME: only clear the cache from the current environment dir
    asv_build_pattern = os.path.join(bench_path, env_dir, "*", "asv-build-cache")
    for asv_build_cache in glob.glob(asv_build_pattern, recursive=True):
        for c in h_commits:
            try: shutil.rmtree(os.path.join(asv_build_cache, c))
            except OSError: pass

def asv_substitute_config(in_config, out_config, **custom_vars):
    """
    A workaround to allow substituting custom tokens within
    ASV configuration file since there's no official way to add custom
    variables(e.g. env vars).

    Parameters
    ----------
    in_config : str
        The path of ASV configuration file, e.g. '/path/to/asv.conf.json'
    out_config : str
        The path of generated configuration file,
        e.g. '/path/to/asv_substituted.conf.json'.

    The other keyword arguments represent the custom variables.

    Returns
    -------
    True(is cached) if 'out_config' is already generated with
    the same '**custom_vars' and updated with latest 'in_config',
    False otherwise.

    Examples
    --------
    See asv_compare_config().
    """
    assert in_config != out_config
    assert len(custom_vars) > 0

    def sdbm_hash(*factors):
        chash = 0
        for f in factors:
            for char in str(f):
                chash  = ord(char) + (chash << 6) + (chash << 16) - chash
                chash &= 0xFFFFFFFF
        return chash

    vars_hash = sdbm_hash(custom_vars, os.path.getmtime(in_config))
    try:
        with open(out_config) as wfd:
            hash_line = wfd.readline().split('hash:')
            if len(hash_line) > 1 and int(hash_line[1]) == vars_hash:
                return True
    except OSError:
        pass

    custom_vars = {f'{{{k}}}':v for k, v in custom_vars.items()}
    with open(in_config, "r") as rfd, open(out_config, "w") as wfd:
        wfd.write(f"// hash:{vars_hash}\n")
        wfd.write("// This file is automatically generated by runtests.py\n")
        for line in rfd:
            for key, val in custom_vars.items():
                line = line.replace(key, val)
            wfd.write(line)
    return False

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

def check_lint(lint_args):
    """
    Adds ROOT_DIR to path and performs lint checks.
    This functions exits the program with status code of lint check.
    """
    sys.path.append(ROOT_DIR)
    try:
        from tools.linter import DiffLinter
    except ModuleNotFoundError as e:
        print(f"Error: {e.msg}. "
              "Install using linter_requirements.txt.")
        sys.exit(1)

    uncommitted = lint_args == "uncommitted"
    branch = "main" if uncommitted else lint_args

    DiffLinter(branch).run_lint(uncommitted)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
