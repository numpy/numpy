import pytest
import sys


def _f2py_limited_api(request: pytest.FixtureRequest):
    limited_api = request.param
    if limited_api:
        if not list(request.node.iter_markers(name='f2py_stable_abi_good')):
            pytest.skip("Stable ABI not yet supported")
            return
        if list(request.node.iter_markers(name='f2py_stable_abi_bad')):
            pytest.skip("Test not compatible with Stable ABI")
            return
    return request.param

_limited_api_versions = [None]+[
    f"3.{x}" for x in range(12, sys.version_info[1]+1)
]

@pytest.fixture(scope='function', params=_limited_api_versions)
def f2py_limited_api(request):
    return _f2py_limited_api(request)

@pytest.fixture(scope='module', params=_limited_api_versions)
def f2py_limited_api_module(request):
    return _f2py_limited_api(request)
