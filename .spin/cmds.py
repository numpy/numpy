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
    """
    if sphinx_target not in ('targets', 'help'):
        if install_deps:
            util.run(['pip', 'install', '-q', '-r', 'doc_requirements.txt'])

    meson.docs.ignore_unknown_options = True
    del ctx.params['install_deps']
    ctx.forward(meson.docs)

docs.__doc__ = meson.docs.__doc__


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
    default="auto",
    help="Number of parallel jobs for testing; use all CPUs by default"
)

@click.pass_context
def test(ctx, pytest_args, markexpr, n_jobs):
    """🔧 Run tests
    """
    if not pytest_args:
        pytest_args = ('numpy',)

    if '-m' not in pytest_args:
        pytest_args = ('-m', markexpr) + pytest_args

    if '-n' not in pytest_args:
        pytest_args = ('-n', str(n_jobs)) + pytest_args

    ctx.params['pytest_args'] = pytest_args

    for extra_param in ('markexpr', 'n_jobs'):
        del ctx.params[extra_param]
    ctx.forward(meson.test)

testdoc = meson.test.__doc__.split('\n')
testdoc = [l for l in testdoc if not
           (('To parallelize' in l) or ('NUM_JOBS' in l))]
testdoc = '\n'.join(testdoc).replace('\n\n', '\n')
test.__doc__ = testdoc
