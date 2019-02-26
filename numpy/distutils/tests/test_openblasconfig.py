from __future__ import division, absolute_import, print_function

import pytest
import platform
from distutils.version import LooseVersion

import numpy as np
from numpy.distutils import system_info

@pytest.mark.skipif('versions' not in system_info.get_info('openblas'),
                     reason="Requires openblas")
def test_openblas_versions_type():
    # the 'versions' key in the openblas get_info
    # dict should correspond to a list
    actual = system_info.get_info('openblas')['versions']
    assert isinstance(actual, list)

@pytest.mark.skipif('versions' not in system_info.get_info('openblas'),
                     reason="Requires openblas")
def test_openblas_version_constraints():
    # verify that the OpenBLAS version is >= 0.3.4
    # as this was the point of implementation of version
    # access in C API
    actual_versions = system_info.get_info('openblas')['versions']
    for version in actual_versions:
        assert ((LooseVersion(version) >= LooseVersion("0.3.4")) or
                version is None)
