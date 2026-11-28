import subprocess
import sys

import pytest

import numpy as np
from numpy import _smoke
from numpy.testing import HAS_SUBPROCESSES


def test_smoke_output(capsys):
    assert np.test(smoke=True, verbose=0) is True
    assert capsys.readouterr().out == ''
    assert np.test(smoke=True) is True
    assert 'checks passed' in capsys.readouterr().out
    assert np.test(smoke=True, verbose=2) is True
    assert 'PASS linear algebra' in capsys.readouterr().out


@pytest.mark.parametrize('options', [
    {'label': 'full'}, {'extra_argv': ['-x']}, {'doctests': True},
    {'coverage': True}, {'durations': 0}, {'tests': []},
])
def test_smoke_rejects_pytest_options(options):
    with pytest.raises(ValueError, match='pytest-specific'):
        np.test(smoke=True, **options)


def test_subpackage_rejected():
    with pytest.raises(ValueError, match='numpy.test'):
        np.linalg.test(smoke=True)


def test_failure_continues(monkeypatch, capsys):
    def fail():
        raise RuntimeError('injected failure')
    monkeypatch.setattr(_smoke, '_linalg', fail)
    assert np.test(smoke=True, verbose=0) is False
    output = capsys.readouterr().out
    assert 'FAIL linear algebra: RuntimeError: injected failure' in output


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
def test_no_pytest_or_test_imports():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pytest" or name.startswith("pytest.") or ".tests" in name:
        raise RuntimeError("forbidden import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import numpy
if not numpy.test(smoke=True):
    raise SystemExit(1)
import sys
if "pytest" in sys.modules or any(".tests" in n for n in sys.modules):
    raise SystemExit("test module loaded")
'''
    subprocess.run([sys.executable, '-c', code], check=True)
