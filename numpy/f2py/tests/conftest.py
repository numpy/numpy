# """
# Pytest configuration and fixtures for the Numpy test suite.
# """
# import os
# import tempfile

# import pytest
# import numpy

# _collect_results = {}

# def pytest_configure(config):
#     config.addinivalue_line("markers",
#         "valgrind_error: Tests that are known to error under valgrind.")
#     config.addinivalue_line("markers",
#         "leaks_references: Tests that are known to leak references.")
#     config.addinivalue_line("markers",
#         "slow: Tests that are very slow.")
#     config.addinivalue_line("markers",
#         "slow_pypy: Tests that are very slow on pypy.")

# def pytest_addoption(parser):
#     parser.addoption("--available-memory", action="store", default=None,
#                      help=("Set amount of memory available for running the "
#                            "test suite. This can result to tests requiring "
#                            "especially large amounts of memory to be skipped. "
#                            "Equivalent to setting environment variable "
#                            "NPY_AVAILABLE_MEM. Default: determined"
#                            "automatically."))

# def pytest_sessionstart(session):
#     available_mem = session.config.getoption('available_memory')
#     if available_mem is not None:
#         os.environ['NPY_AVAILABLE_MEM'] = available_mem

# @pytest.fixture(autouse=True)
# def add_np(doctest_namespace):
#     doctest_namespace['np'] = numpy

# @pytest.fixture(autouse=True)
# def env_setup(monkeypatch):
#     monkeypatch.setenv('PYTHONHASHSEED', '0')
