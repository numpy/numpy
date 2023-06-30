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
    "--install-deps/--no-install-deps",
    default=True,
    help="Install dependencies before building"
)
@click.option(
    '--build/--no-build',
    default=True,
    help="Build numpy before generating docs"
)
@click.option(
    '--jobs', '-j',
    default="auto",
    help="Number of parallel build jobs"
)
@click.pass_context
def docs(ctx, sphinx_target, clean, install_deps, build, jobs):
    """📖 Build documentation

    By default, SPHINXOPTS="-W", raising errors on warnings.
    To build without raising on warnings:

      SPHINXOPTS="" spin docs

    To list all Sphinx targets:

      spin docs targets

    To build another Sphinx target:

      spin docs TARGET

    """
    if sphinx_target == "targets":
        clean = False
        install_deps = False
        build = False
        sphinx_target = "help"

    if clean:
        doc_dirs = [
            "./doc/build/",
            "./doc/source/api/",
            "./doc/source/auto_examples/",
            "./doc/source/jupyterlite_contents/",
        ]
        for doc_dir in doc_dirs:
            if os.path.isdir(doc_dir):
                print(f"Removing {doc_dir!r}")
                shutil.rmtree(doc_dir)

    if build:
        click.secho(
            "Invoking `build` prior to running tests:",
            bold=True, fg="bright_green"
        )
        ctx.invoke(meson.build)

    try:
        site_path = meson._get_site_packages()
    except FileNotFoundError:
        print("No built numpy found; run `spin build` first.")
        sys.exit(1)

    if install_deps:
        util.run(['pip', 'install', '-q', '-r', 'doc_requirements.txt'])

    opts = os.environ.get('SPHINXOPTS', "-W")
    os.environ['SPHINXOPTS'] = f'{opts} -j {jobs}'
    click.secho(
        f"$ export SPHINXOPTS={os.environ['SPHINXOPTS']}", bold=True,
        fg="bright_blue"
    )

    os.environ['PYTHONPATH'] = \
        f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    click.secho(
        f"$ export PYTHONPATH={os.environ['PYTHONPATH']}",
        bold=True, fg="bright_blue"
    )
    util.run(['make', '-C', 'doc', 'html'], replace=True)
