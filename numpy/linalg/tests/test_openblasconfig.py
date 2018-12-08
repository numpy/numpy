from __future__ import division, absolute_import, print_function

import pytest
import platform
from distutils.version import LooseVersion

import numpy as np
from numpy.distutils import system_info

@pytest.mark.skipif(len(system_info.get_info('openblas')) < 2,
                     reason="Requires openblas")
def test_openblas_info_return_type():
    # _openblas_info() should return a bytes
    # object containing information about
    # the openblas that was linked to NumPy;
    from numpy.linalg import openblas_config
    actual = openblas_config._openblas_info()
    assert isinstance(actual, bytes)

@pytest.mark.skipif(len(system_info.get_info('openblas')) < 2,
                    reason="Requires openblas")
def test_openblas_private_info():
    # directly probe the private function
    # that accesses the openblas version used
    # to compile NumPy
    from numpy.linalg import openblas_config
    info_string = openblas_config._openblas_info()
    info_list = info_string.split()
    if info_list[0] == b'OpenBLAS':
        # true for OpenBLAS >= 0.3.4
        version = info_list[1].decode()
        assert LooseVersion(version) >= LooseVersion("0.3.4")

@pytest.mark.skipif(len(system_info.get_info('openblas')) < 2,
                    reason="Requires openblas")
def test_openblas_get_info():
    # test for the presence of the 'version' key
    # in the openblas info dict, and its appropriate
    # value
    openblas_dict = system_info.get_info('openblas')
    version = openblas_dict['version']

    # for openblas < 0.3.4 there is no version
    # information provided, so NumPy will default
    # to None
    assert (version == None or
            LooseVersion(version.decode()) >= LooseVersion("0.3.4"))
