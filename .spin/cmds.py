import os
import shutil
import sys

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
