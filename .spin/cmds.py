import os
import pathlib
import importlib
import shutil
import subprocess
import sys

import click
import spin
from spin.cmds import meson

IS_PYPY = (sys.implementation.name == 'pypy')

# Check that the meson git submodule is present
curdir = pathlib.Path(__file__).parent
meson_import_dir = curdir.parent / 'vendored-meson' / 'meson' / 'mesonbuild'
if not meson_import_dir.exists():
    raise RuntimeError(
        'The `vendored-meson/meson` git submodule does not exist! '
        'Run `git submodule update --init` to fix this problem.'
    )


def _get_numpy_tools(filename):
    filepath = pathlib.Path('tools', filename)
    spec = importlib.util.spec_from_file_location(filename.stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@click.command()
@click.argument(
    "token",
    required=True
)
@click.argument(
    "revision-range",
    required=True
)
def changelog(token, revision_range):
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


@click.option(
    "--with-scipy-openblas", type=click.Choice(["32", "64"]),
    default=None,
    help="Build with pre-installed scipy-openblas32 or scipy-openblas64 wheel"
)
@spin.util.extend_command(spin.cmds.meson.build)
def build(*, parent_callback, with_scipy_openblas, **kwargs):
    if with_scipy_openblas:
        _config_openblas(with_scipy_openblas)
    parent_callback(**kwargs)


@spin.util.extend_command(spin.cmds.meson.docs)
def docs(*, parent_callback, **kwargs):
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
    kwargs['clean_dirs'] = [
        './doc/build/',
        './doc/source/reference/generated',
        './doc/source/reference/random/bit_generators/generated',
        './doc/source/reference/random/generated',
    ]

    # Run towncrier without staging anything for commit. This is the way to get
    # release notes snippets included in a local doc build.
    cmd = ['towncrier', 'build', '--version', '2.x.y', '--keep', '--draft']
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    outfile = curdir.parent / 'doc' / 'source' / 'release' / 'notes-towncrier.rst'
    with open(outfile, 'w') as f:
        f.write(p.stdout)

    parent_callback(**kwargs)


# Override default jobs to 1
jobs_param = next(p for p in docs.params if p.name == 'jobs')
jobs_param.default = 1

if IS_PYPY:
    default = "not slow and not slow_pypy"
else:
    default = "not slow"

@click.option(
    "-m",
    "markexpr",
    metavar='MARKEXPR',
    default=default,
    help="Run tests with the given markers"
)
@spin.util.extend_command(spin.cmds.meson.test)
def test(*, parent_callback, pytest_args, tests, markexpr, **kwargs):
    """
    By default, spin will run `-m 'not slow'`. To run the full test suite, use
    `spin test -m full`
    """  # noqa: E501
    if (not pytest_args) and (not tests):
        pytest_args = ('--pyargs', 'numpy')

    if '-m' not in pytest_args:
        if markexpr != "full":
            pytest_args = ('-m', markexpr) + pytest_args

    kwargs['pytest_args'] = pytest_args
    parent_callback(**{'pytest_args': pytest_args, 'tests': tests, **kwargs})


@spin.util.extend_command(test, doc='')
def check_docs(*, parent_callback, pytest_args, **kwargs):
    """🔧 Run doctests of objects in the public API.

    PYTEST_ARGS are passed through directly to pytest, e.g.:

      spin check-docs -- --pdb

    To run tests on a directory:

     \b
     spin check-docs numpy/linalg

    To report the durations of the N slowest doctests:

      spin check-docs -- --durations=N

    To run doctests that match a given pattern:

     \b
     spin check-docs -- -k "slogdet"
     spin check-docs numpy/linalg -- -k "det and not slogdet"

    \b
    Note:
    -----

    \b
     - This command only runs doctests and skips everything under tests/
     - This command only doctests public objects: those which are accessible
       from the top-level `__init__.py` file.

    """  # noqa: E501
    try:
        # prevent obscure error later
        import scipy_doctest
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("scipy-doctest not installed") from e

    if (not pytest_args):
        pytest_args = ('--pyargs', 'numpy')

    # turn doctesting on:
    doctest_args = (
        '--doctest-modules',
        '--doctest-collect=api'
    )

    pytest_args = pytest_args + doctest_args

    parent_callback(**{'pytest_args': pytest_args, **kwargs})


@spin.util.extend_command(test, doc='')
def check_tutorials(*, parent_callback, pytest_args, **kwargs):
    """🔧 Run doctests of user-facing rst tutorials.

    To test all tutorials in the numpy doc/source/user/ directory, use

      spin check-tutorials

    To run tests on a specific RST file:

     \b
     spin check-tutorials doc/source/user/absolute-beginners.rst

    \b
    Note:
    -----

    \b
     - This command only runs doctests and skips everything under tests/
     - This command only doctests public objects: those which are accessible
       from the top-level `__init__.py` file.

    """  # noqa: E501
    # handle all of
    #   - `spin check-tutorials` (pytest_args == ())
    #   - `spin check-tutorials path/to/rst`, and
    #   - `spin check-tutorials path/to/rst -- --durations=3`
    if (not pytest_args) or all(arg.startswith('-') for arg in pytest_args):
        pytest_args = ('doc/source/user',) + pytest_args

    # make all paths relative to the numpy source folder
    pytest_args = tuple(
        str(curdir / '..' / arg) if not arg.startswith('-') else arg
        for arg in pytest_args
   )

    # turn doctesting on:
    doctest_args = (
        '--doctest-glob=*rst',
    )

    pytest_args = pytest_args + doctest_args

    parent_callback(**{'pytest_args': pytest_args, **kwargs})


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
    p = spin.util.run(['git', 'rev-parse', commit], output=False, echo=False)
    if p.returncode != 0:
        raise (
            click.ClickException(
                f'Could not find SHA matching commit `{commit}`'
            )
        )

    return p.stdout.decode('ascii').strip()


def _dirty_git_working_dir():
    # Changes to the working directory
    p0 = spin.util.run(['git', 'diff-files', '--quiet'])

    # Staged changes
    p1 = spin.util.run(['git', 'diff-index', '--quiet', '--cached', 'HEAD'])

    return (p0.returncode != 0 or p1.returncode != 0)


def _run_asv(cmd):
    # Always use ccache, if installed
    PATH = os.environ['PATH']
    EXTRA_PATH = os.pathsep.join([
        '/usr/lib/ccache', '/usr/lib/f90cache',
        '/usr/local/lib/ccache', '/usr/local/lib/f90cache'
    ])
    env = os.environ
    env['PATH'] = f'{EXTRA_PATH}{os.pathsep}{PATH}'

    # Control BLAS/LAPACK threads
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    # Limit memory usage
    try:
        _set_mem_rlimit()
    except (ImportError, RuntimeError):
        pass

    spin.util.run(cmd, cwd='benchmarks', env=env)

@click.command()
@click.option(
    '--fix',
    is_flag=True,
    default=False,
    required=False,
)
@click.pass_context
def lint(ctx, fix):
    """🔦 Run lint checks with Ruff

    \b
    To run automatic fixes use:

    \b
    $ spin lint --fix
    """
    try:
        linter = _get_numpy_tools(pathlib.Path('linter.py'))
    except ModuleNotFoundError as e:
        raise click.ClickException(
            f"{e.msg}. Install using requirements/linter_requirements.txt"
        )

    linter.DiffLinter().run_lint(fix)

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
@meson.build_dir_option
@click.pass_context
def bench(ctx, tests, compare, verbose, quick, commits, build_dir):
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

        meson._set_pythonpath(build_dir)

        p = spin.util.run(
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


@spin.util.extend_command(meson.python)
def python(*, parent_callback, **kwargs):
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')

    parent_callback(**kwargs)


@click.command(context_settings={
    'ignore_unknown_options': True
})
@click.argument("ipython_args", metavar='', nargs=-1)
@meson.build_dir_option
def ipython(*, ipython_args, build_dir):
    """💻 Launch IPython shell with PYTHONPATH set

    OPTIONS are passed through directly to IPython, e.g.:

    spin ipython -i myscript.py
    """
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')

    ctx = click.get_current_context()
    ctx.invoke(build)

    ppath = meson._set_pythonpath(build_dir)

    print(f'💻 Launching IPython with PYTHONPATH="{ppath}"')

    # In spin >= 0.13.1, can replace with extended command, setting `pre_import`
    preimport = (r"import numpy as np; "
                 r"print(f'\nPreimported NumPy {np.__version__} as np')")
    spin.util.run(["ipython", "--ignore-cwd",
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
            fid.write(
                openblas.get_pkg_config(use_preloading=True)
            )


@click.command()
@click.option(
    "-v", "--version-override",
    help="NumPy version of release",
    required=False
)
def notes(version_override):
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
    project_config = spin.util.get_config()
    version = version_override or project_config['project.version']

    click.secho(
        f"Generating release notes for NumPy {version}",
        bold=True, fg="bright_green",
    )

    # Check if `towncrier` is installed
    if not shutil.which("towncrier"):
        raise click.ClickException(
            "please install `towncrier` to use this command"
        )

    click.secho(
        f"Reading upcoming changes from {project_config['tool.towncrier.directory']}",
        bold=True, fg="bright_yellow"
    )
    # towncrier build --version 2.1 --yes
    cmd = ["towncrier", "build", "--version", version, "--yes"]
    p = spin.util.run(cmd=cmd, sys_exit=False, output=True, encoding="utf-8")
    if p.returncode != 0:
        raise click.ClickException(
            f"`towncrier` failed returned {p.returncode} with error `{p.stderr}`"
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
