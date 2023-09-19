import os
import sys
import pathlib

from mesonpy import (
    build_sdist,
    build_wheel,
    build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    get_requires_for_build_editable,
)


# The numpy-vendored version of Meson. Put the directory that the executable
# `meson` is in at the front of the PATH.
curdir = pathlib.Path(__file__).parent.resolve()
meson_executable_dir = str(curdir.parent.parent / 'entrypoint')
os.environ['PATH'] = meson_executable_dir + os.pathsep + os.environ['PATH']

# Check that the meson git submodule is present
meson_import_dir = curdir.parent.parent / 'meson' / 'mesonbuild'
if not meson_import_dir.exists():
    raise RuntimeError(
        'The `vendored-meson/meson` git submodule does not exist! ' +
        'Run `git submodule update --init` to fix this problem.'
    )
