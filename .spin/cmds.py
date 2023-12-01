import os
import shutil
import sys
import argparse
import tempfile
import pathlib
import shutil
import json
import pathlib
import importlib
import subprocess

import click
from spin import util
from spin.cmds import meson


# Check that the meson git submodule is present
curdir = pathlib.Path(__file__).parent
meson_import_dir = curdir.parent / 'vendored-meson' / 'meson' / 'mesonbuild'
if not meson_import_dir.exists():
    raise RuntimeError(
        'The `vendored-meson/meson` git submodule does not exist! ' +
        'Run `git submodule update --init` to fix this problem.'
    )


def _get_numpy_tools(filename):
    filepath = pathlib.Path('tools', filename)
    spec = importlib.util.spec_from_file_location(filename.stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@click.command()
@click.option(
    "-t", "--token",
    help="GitHub access token",
    required=True
)
@click.option(
    "--revision-range",
    help="<revision>..<revision>",
    required=True
)
@click.pass_context
def changelog(ctx, token, revision_range):
    """👩 Get change log for provided revision range

    \b
    Example:

    \b
    $ spin authors -t $GH_TOKEN --revision-range v1.25.0..v1.26.0
    """
    try:
        from github.GithubException import GithubException
        from git.exc import GitError
        changelog = _get_numpy_tools(pathlib.Path('changelog.py'))
    except ModuleNotFoundError as e:
        raise click.ClickException(
            f"{e.msg}. Install the missing packages to use this command."
        )
    click.secho(
        f"Generating change log for range {revision_range}",
        bold=True, fg="bright_green",
    )
    try:
        changelog.main(token, revision_range)
    except GithubException as e:
        raise click.ClickException(
            f"GithubException raised with status: {e.status} "
            f"and message: {e.data['message']}"
        )
    except GitError as e:
        raise click.ClickException(
            f"Git error in command `{' '.join(e.command)}` "
            f"with error message: {e.stderr}"
        )


@click.command()
@click.option(
    "-j", "--jobs",
    help="Number of parallel tasks to launch",
    type=int
)
@click.option(
    "--clean", is_flag=True,
    help="Clean build directory before build"
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Print all build output, even installation"
)
@click.option(
    "--with-scipy-openblas", type=click.Choice(["32", "64"]),
    default=None,
    help="Build with pre-installed scipy-openblas32 or scipy-openblas64 wheel"
)
@click.argument("meson_args", nargs=-1)
@click.pass_context
def build(ctx, meson_args, with_scipy_openblas, jobs=None, clean=False, verbose=False, quiet=False):
    """🔧 Build package with Meson/ninja and install

    MESON_ARGS are passed through e.g.:

    spin build -- -Dpkg_config_path=/lib64/pkgconfig

    The package is installed to build-install

    By default builds for release, to be able to use a debugger set CFLAGS
    appropriately. For example, for linux use

    CFLAGS="-O0 -g" spin build
    """
    # XXX keep in sync with upstream build
    if with_scipy_openblas:
        _config_openblas(with_scipy_openblas)
    ctx.params.pop("with_scipy_openblas", None)
    ctx.forward(meson.build)


@click.command()
@click.argument("sphinx_target", default="html")
@click.option(
    "--clean", is_flag=True,
    default=False,
    help="Clean previously built docs before building"
)
@click.option(
    "--build/--no-build",
    "first_build",
    default=True,
    help="Build numpy before generating docs",
)
@click.option(
    '--jobs', '-j',
    metavar='N_JOBS',
    default="auto",
    help="Number of parallel build jobs"
)
@click.pass_context
def docs(ctx, sphinx_target, clean, first_build, jobs):
    """📖 Build Sphinx documentation

    By default, SPHINXOPTS="-W", raising errors on warnings.
    To build without raising on warnings:

      SPHINXOPTS="" spin docs

    To list all Sphinx targets:

      spin docs targets

    To build another Sphinx target:

      spin docs TARGET

    E.g., to build a zipfile of the html docs for distribution:

      spin docs dist

    """
    meson.docs.ignore_unknown_options = True
    ctx.forward(meson.docs)


@click.command()
@click.argument("pytest_args", nargs=-1)
@click.option(
    "-m",
    "markexpr",
    metavar='MARKEXPR',
    default="not slow",
    help="Run tests with the given markers"
)
@click.option(
    "-j",
    "n_jobs",
    metavar='N_JOBS',
    default="1",
    help=("Number of parallel jobs for testing. "
          "Can be set to `auto` to use all cores.")
)
@click.option(
    "--tests", "-t",
    metavar='TESTS',
    help=("""
Which tests to run. Can be a module, function, class, or method:

 \b
 numpy.random
 numpy.random.tests.test_generator_mt19937
 numpy.random.tests.test_generator_mt19937::TestMultivariateHypergeometric
 numpy.random.tests.test_generator_mt19937::TestMultivariateHypergeometric::test_edge_cases
 \b
""")
)
@click.option(
    '--verbose', '-v', is_flag=True, default=False
)
@click.pass_context
def test(ctx, pytest_args, markexpr, n_jobs, tests, verbose):
    """🔧 Run tests

    PYTEST_ARGS are passed through directly to pytest, e.g.:

      spin test -- --pdb

    To run tests on a directory or file:

     \b
     spin test numpy/linalg
     spin test numpy/linalg/tests/test_linalg.py

    To report the durations of the N slowest tests:

      spin test -- --durations=N

    To run tests that match a given pattern:

     \b
     spin test -- -k "geometric"
     spin test -- -k "geometric and not rgeometric"

    By default, spin will run `-m 'not slow'`. To run the full test suite, use
    `spin -m full`

    For more, see `pytest --help`.
    """  # noqa: E501
    if (not pytest_args) and (not tests):
        pytest_args = ('numpy',)

    if '-m' not in pytest_args:
        if markexpr != "full":
            pytest_args = ('-m', markexpr) + pytest_args

    if (n_jobs != "1") and ('-n' not in pytest_args):
        pytest_args = ('-n', str(n_jobs)) + pytest_args

    if tests and not ('--pyargs' in pytest_args):
        pytest_args = ('--pyargs', tests) + pytest_args

    if verbose:
        pytest_args = ('-v',) + pytest_args

    ctx.params['pytest_args'] = pytest_args

    for extra_param in ('markexpr', 'n_jobs', 'tests', 'verbose'):
        del ctx.params[extra_param]
    ctx.forward(meson.test)


# From scipy: benchmarks/benchmarks/common.py
def _set_mem_rlimit(max_mem=None):
    """
    Set address space rlimit
    """
    import resource
    import psutil

    mem = psutil.virtual_memory()

    if max_mem is None:
        max_mem = int(mem.total * 0.7)
    cur_limit = resource.getrlimit(resource.RLIMIT_AS)
    if cur_limit[0] > 0:
        max_mem = min(max_mem, cur_limit[0])

    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, cur_limit[1]))
    except ValueError:
        # on macOS may raise: current limit exceeds maximum limit
        pass


def _commit_to_sha(commit):
    p = util.run(['git', 'rev-parse', commit], output=False, echo=False)
    if p.returncode != 0:
        raise(
            click.ClickException(
                f'Could not find SHA matching commit `{commit}`'
            )
        )

    return p.stdout.decode('ascii').strip()


def _dirty_git_working_dir():
    # Changes to the working directory
    p0 = util.run(['git', 'diff-files', '--quiet'])

    # Staged changes
    p1 = util.run(['git', 'diff-index', '--quiet', '--cached', 'HEAD'])

    return (p0.returncode != 0 or p1.returncode != 0)


def _run_asv(cmd):
    # Always use ccache, if installed
    PATH = os.environ['PATH']
    EXTRA_PATH = os.pathsep.join([
        '/usr/lib/ccache', '/usr/lib/f90cache',
        '/usr/local/lib/ccache', '/usr/local/lib/f90cache'
    ])
    env = os.environ
    env['PATH'] = f'EXTRA_PATH:{PATH}'

    # Control BLAS/LAPACK threads
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    # Limit memory usage
    try:
        _set_mem_rlimit()
    except (ImportError, RuntimeError):
        pass

    util.run(cmd, cwd='benchmarks', env=env)

@click.command()
@click.option(
    "-b", "--branch",
    metavar='branch',
    default="main",
)
@click.option(
    '--uncommitted',
    is_flag=True,
    default=False,
    required=False,
)
@click.pass_context
def lint(ctx, branch, uncommitted):
    """🔦 Run lint checks on diffs.
    Provide target branch name or `uncommitted` to check changes before committing:

    \b
    Examples:

    \b
    For lint checks of your development brach with `main` or a custom branch:

    \b
    $ spin lint # defaults to main
    $ spin lint --branch custom_branch

    \b
    To check just the uncommitted changes before committing

    \b
    $ spin lint --uncommitted
    """
    try:
        linter = _get_numpy_tools(pathlib.Path('linter.py'))
    except ModuleNotFoundError as e:
        raise click.ClickException(
            f"{e.msg}. Install using linter_requirements.txt"
        )

    linter.DiffLinter(branch).run_lint(uncommitted)

@click.command()
@click.option(
    '--tests', '-t',
    default=None, metavar='TESTS', multiple=True,
    help="Which tests to run"
)
@click.option(
    '--compare', '-c',
    is_flag=True,
    default=False,
    help="Compare benchmarks between the current branch and main "
         "(unless other branches specified). "
         "The benchmarks are each executed in a new isolated "
         "environment."
)
@click.option(
    '--verbose', '-v', is_flag=True, default=False
)
@click.option(
    '--quick', '-q', is_flag=True, default=False,
    help="Run each benchmark only once (timings won't be accurate)"
)
@click.argument(
    'commits', metavar='',
    required=False,
    nargs=-1
)
@click.pass_context
def bench(ctx, tests, compare, verbose, quick, commits):
    """🏋 Run benchmarks.

    \b
    Examples:

    \b
    $ spin bench -t bench_lib
    $ spin bench -t bench_random.Random
    $ spin bench -t Random -t Shuffle

    Two benchmark runs can be compared.
    By default, `HEAD` is compared to `main`.
    You can also specify the branches/commits to compare:

    \b
    $ spin bench --compare
    $ spin bench --compare main
    $ spin bench --compare main HEAD

    You can also choose which benchmarks to run in comparison mode:

    $ spin bench -t Random --compare
    """
    if not commits:
        commits = ('main', 'HEAD')
    elif len(commits) == 1:
        commits = commits + ('HEAD',)
    elif len(commits) > 2:
        raise click.ClickException(
            'Need a maximum of two revisions to compare'
        )

    bench_args = []
    for t in tests:
        bench_args += ['--bench', t]

    if verbose:
        bench_args = ['-v'] + bench_args

    if quick:
        bench_args = ['--quick'] + bench_args

    if not compare:
        # No comparison requested; we build and benchmark the current version

        click.secho(
            "Invoking `build` prior to running benchmarks:",
            bold=True, fg="bright_green"
        )
        ctx.invoke(build)

        meson._set_pythonpath()

        p = util.run(
            ['python', '-c', 'import numpy as np; print(np.__version__)'],
            cwd='benchmarks',
            echo=False,
            output=False
        )
        os.chdir('..')

        np_ver = p.stdout.strip().decode('ascii')
        click.secho(
            f'Running benchmarks on NumPy {np_ver}',
            bold=True, fg="bright_green"
        )
        cmd = [
            'asv', 'run', '--dry-run', '--show-stderr', '--python=same'
        ] + bench_args
        _run_asv(cmd)
    else:
        # Ensure that we don't have uncommited changes
        commit_a, commit_b = [_commit_to_sha(c) for c in commits]

        if commit_b == 'HEAD' and _dirty_git_working_dir():
            click.secho(
                "WARNING: you have uncommitted changes --- "
                "these will NOT be benchmarked!",
                fg="red"
            )

        cmd_compare = [
            'asv', 'continuous', '--factor', '1.05',
        ] + bench_args + [commit_a, commit_b]
        _run_asv(cmd_compare)


@click.command(context_settings={
    'ignore_unknown_options': True
})
@click.argument("python_args", metavar='', nargs=-1)
@click.pass_context
def python(ctx, python_args):
    """🐍 Launch Python shell with PYTHONPATH set

    OPTIONS are passed through directly to Python, e.g.:

    spin python -c 'import sys; print(sys.path)'
    """
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')
    ctx.forward(meson.python)


@click.command(context_settings={
    'ignore_unknown_options': True
})
@click.argument("ipython_args", metavar='', nargs=-1)
@click.pass_context
def ipython(ctx, ipython_args):
    """💻 Launch IPython shell with PYTHONPATH set

    OPTIONS are passed through directly to IPython, e.g.:

    spin ipython -i myscript.py
    """
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')

    ctx.invoke(build)

    ppath = meson._set_pythonpath()

    print(f'💻 Launching IPython with PYTHONPATH="{ppath}"')
    preimport = (r"import numpy as np; "
                 r"print(f'\nPreimported NumPy {np.__version__} as np')")
    util.run(["ipython", "--ignore-cwd",
              f"--TerminalIPythonApp.exec_lines={preimport}"] +
             list(ipython_args))


@click.command(context_settings={"ignore_unknown_options": True})
@click.pass_context
def mypy(ctx):
    """🦆 Run Mypy tests for NumPy
    """
    env = os.environ
    env['NPY_RUN_MYPY_IN_TESTSUITE'] = '1'
    ctx.params['pytest_args'] = [os.path.join('numpy', 'typing')]
    ctx.params['markexpr'] = 'full'
    ctx.forward(test)

@click.command(context_settings={
    'ignore_unknown_options': True
})
@click.option(
    "--with-scipy-openblas", type=click.Choice(["32", "64"]),
    default=None, required=True,
    help="Build with pre-installed scipy-openblas32 or scipy-openblas64 wheel"
)
def config_openblas(with_scipy_openblas):
    """🔧 Create .openblas/scipy-openblas.pc file

    Also create _distributor_init_local.py

    Requires a pre-installed scipy-openblas64 or scipy-openblas32
    """
    _config_openblas(with_scipy_openblas)


def _config_openblas(blas_variant):
    import importlib
    basedir = os.getcwd()
    openblas_dir = os.path.join(basedir, ".openblas")
    pkg_config_fname = os.path.join(openblas_dir, "scipy-openblas.pc")
    if blas_variant:
        module_name = f"scipy_openblas{blas_variant}"
        try:
            openblas = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise RuntimeError(f"'pip install {module_name} first")
        local = os.path.join(basedir, "numpy", "_distributor_init_local.py")
        with open(local, "wt", encoding="utf8") as fid:
            fid.write(f"import {module_name}\n")
        os.makedirs(openblas_dir, exist_ok=True)
        with open(pkg_config_fname, "wt", encoding="utf8") as fid:
            fid.write(openblas.get_pkg_config().replace("\\", "/"))


@click.command()
@click.option(
    "-v", "--version-override",
    help="NumPy version of release",
    required=False
)
@click.pass_context
def notes(ctx, version_override):
    """🎉 Generate release notes and validate

    \b
    Example:

    \b
    $ spin notes --version-override 2.0

    \b
    To automatically pick the version

    \b
    $ spin notes
    """
    project_config = util.get_config()
    version = version_override or project_config['project.version']

    click.secho(
        f"Generating release notes for NumPy {version}",
        bold=True, fg="bright_green",
    )

    # Check if `towncrier` is installed
    if not shutil.which("towncrier"):
        raise click.ClickException(
            f"please install `towncrier` to use this command"
        )

    click.secho(
        f"Reading upcoming changes from {project_config['tool.towncrier.directory']}",
        bold=True, fg="bright_yellow"
    )
    # towncrier build --version 2.1 --yes
    cmd = ["towncrier", "build", "--version", version, "--yes"]
    try:
        p = util.run(
                cmd=cmd,
                sys_exit=False,
                output=True,
                encoding="utf-8"
            )
    except subprocess.SubprocessError as e:
        raise click.ClickException(
            f"`towncrier` failed returned {e.returncode} with error `{e.stderr}`"
        )

    output_path = project_config['tool.towncrier.filename'].format(version=version)
    click.secho(
        f"Release notes successfully written to {output_path}",
        bold=True, fg="bright_yellow"
    )

    click.secho(
        "Verifying consumption of all news fragments",
        bold=True, fg="bright_green",
    )

    try:
        test_notes = _get_numpy_tools(pathlib.Path('ci', 'test_all_newsfragments_used.py'))
    except ModuleNotFoundError as e:
        raise click.ClickException(
            f"{e.msg}. Install the missing packages to use this command."
        )

    test_notes.main()
