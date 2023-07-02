import os
import shutil
import sys
import argparse

import click
from spin.cmds import meson
from spin import util


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
@click.option(
    "--install-deps/--no-install-deps",
    default=True,
    help="Install dependencies before building"
)
@click.pass_context
def docs(ctx, sphinx_target, clean, first_build, jobs, install_deps):
    """📖 Build Sphinx documentation

    By default, SPHINXOPTS="-W", raising errors on warnings.
    To build without raising on warnings:

      SPHINXOPTS="" spin docs

    To list all Sphinx targets:

      spin docs targets

    To build another Sphinx target:

      spin docs TARGET

    """
    if sphinx_target not in ('targets', 'help'):
        if install_deps:
            util.run(['pip', 'install', '-q', '-r', 'doc_requirements.txt'])

    meson.docs.ignore_unknown_options = True
    del ctx.params['install_deps']
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
@click.pass_context
def test(ctx, pytest_args, markexpr, n_jobs):
    """🔧 Run tests

    PYTEST_ARGS are passed through directly to pytest, e.g.:

      spin test -- -v

    To run tests on a directory or file:

     \b
     spin test numpy/linalg
     spin test numpy/linalg/tests/test_linalg.py

    To run specific tests, by module, function, class, or method:

     \b
     spin test -- --pyargs numpy.random
     spin test -- --pyargs numpy.random.tests.test_generator_mt19937
     spin test -- --pyargs numpy.random.tests.test_generator_mt19937::TestMultivariateHypergeometric
     spin test -- --pyargs numpy.random.tests.test_generator_mt19937::TestMultivariateHypergeometric::test_edge_cases

    To report the durations of the N slowest tests:

      spin test -- --durations=N

    To run tests that match a given pattern:

     \b
     spin test -- -k "geometric"
     spin test -- -k "geometric and not rgeometric"

    For more, see `pytest --help`.
    """  # noqa: E501
    if not pytest_args:
        pytest_args = ('numpy',)

    if '-m' not in pytest_args:
        pytest_args = ('-m', markexpr) + pytest_args

    if (n_jobs != "1") and ('-n' not in pytest_args):
        pytest_args = ('-n', str(n_jobs)) + pytest_args

    ctx.params['pytest_args'] = pytest_args

    for extra_param in ('markexpr', 'n_jobs'):
        del ctx.params[extra_param]
    ctx.forward(meson.test)


@click.command()
@click.argument('python_expr')
def gdb(python_expr):
    """👾 Execute a Python snippet with GDB

    """
    util.run(
        ['gdb', '--args', 'python', '-m', 'spin', 'run',
         'python', '-P', '-c', python_expr],
        replace=True
    )
