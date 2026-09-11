import importlib.metadata
import os
import pathlib

import pytest

import numpy as np
import numpy._core.include
import numpy._core.lib.pkgconfig
from numpy.testing import HAS_SUBPROCESSES, IS_EDITABLE, IS_INSTALLED, NUMPY_ROOT
from numpy.testing._private.utils import run_subprocess

INCLUDE_DIR = NUMPY_ROOT / '_core' / 'include'
PKG_CONFIG_DIR = NUMPY_ROOT / '_core' / 'lib' / 'pkgconfig'


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="`numpy-config` not expected to be installed")
@pytest.mark.skipif(not HAS_SUBPROCESSES,
                    reason="platform cannot start subprocesses")
class TestNumpyConfig:
    def check_numpyconfig(self, arg):
        res = run_subprocess(['numpy-config', arg])
        return res.stdout.strip()

    def test_configtool_version(self):
        stdout = self.check_numpyconfig('--version')
        assert stdout == np.__version__

    def test_configtool_cflags(self):
        stdout = self.check_numpyconfig('--cflags')
        assert f'-I{os.fspath(INCLUDE_DIR)}' in stdout

    def test_configtool_pkgconfigdir(self):
        stdout = self.check_numpyconfig('--pkgconfigdir')
        assert pathlib.Path(stdout) == PKG_CONFIG_DIR.resolve()


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="numpy must be installed to check its entrypoints")
def test_pkg_config_entrypoint():
    (entrypoint,) = importlib.metadata.entry_points(group='pkg_config', name='numpy')
    assert entrypoint.value == numpy._core.lib.pkgconfig.__name__


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="numpy.pc is only available when numpy is installed")
@pytest.mark.skipif(IS_EDITABLE, reason="editable installs don't have a numpy.pc")
def test_pkg_config_config_exists():
    assert PKG_CONFIG_DIR.joinpath('numpy.pc').is_file()
