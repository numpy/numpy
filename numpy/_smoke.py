"""Small installation health checks, independent of NumPy's test suite."""

import importlib
import io
import warnings

import numpy as np
from numpy._public_api import PUBLIC_ALIASED_MODULES, PUBLIC_MODULES
from numpy.testing import assert_allclose, assert_array_equal

_NATIVE_MODULES = (
    'numpy._core._multiarray_umath', 'numpy.fft._pocketfft_umath',
    'numpy.linalg._umath_linalg',
    'numpy.random._generator', 'numpy.random.bit_generator',
    'numpy.random._pcg64', 'numpy.random._mt19937', 'numpy.random._philox',
    'numpy.random._sfc64', 'numpy.random.mtrand',
)
_LEGACY_MODULES = ('numpy.core.multiarray', 'numpy.core.numeric',
                   'numpy.core.umath', 'numpy.linalg.lapack_lite')


def _import_legacy(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        importlib.import_module(name)


def _arrays():
    assert_array_equal(np.arange(6, dtype=np.int64).reshape(2, 3).sum(axis=0),
                       [3, 5, 7])
    a = np.array([(1, 2.0)], dtype=[('x', 'i4'), ('y', 'f8')])
    assert_array_equal(a['x'], [1])
    assert_array_equal(a['y'], [2.0])


def _ufuncs():
    assert_array_equal(np.add([1, 2, 3], [[10], [20]]), [[11, 12, 13], [21, 22, 23]])
    assert_array_equal(np.array([[1, 2], [3, 4]]) @ [[2, 0], [1, 2]], [[4, 4], [10, 8]])


def _linalg():
    assert_allclose(np.linalg.solve([[3., 1.], [1., 2.]], [9., 8.]),
                    [2., 3.], rtol=1e-12, atol=1e-12)


def _fft():
    assert_allclose(np.fft.irfft(np.fft.rfft([1., 2., 3., 4.]), n=4),
                    [1., 2., 3., 4.], rtol=1e-12, atol=1e-12)


def _random():
    a = np.random.default_rng(42).integers(0, 100, size=8)
    b = np.random.default_rng(42).integers(0, 100, size=8)
    assert_array_equal(a, b)
    if a.shape != (8,) or not np.all((0 <= a) & (a < 100)):
        raise AssertionError(f'invalid generator output: {a!r}')
    legacy = np.random.RandomState(42).randint(0, 100, size=8)
    if legacy.shape != (8,) or not np.all((0 <= legacy) & (legacy < 100)):
        raise AssertionError(f'invalid RandomState output: {legacy!r}')


def _strings():
    assert_array_equal(np.strings.upper(np.array(['ab', 'cd'])), ['AB', 'CD'])


def _masked():
    assert_array_equal(np.ma.array([1, 2, 3], mask=[0, 1, 0]).compressed(), [1, 3])


def _polynomial():
    assert_array_equal(np.polynomial.Polynomial([1, 2])(3), 7)


def _io():
    buffer = io.BytesIO()
    data = np.arange(6).reshape(2, 3)
    np.save(buffer, data)
    buffer.seek(0)
    assert_array_equal(np.load(buffer, allow_pickle=False), data)


def run_smoke_checks(verbose=1):
    """Check basic installation health; return a bool without requiring pytest."""
    checks = [(f'import {name}', lambda name=name: importlib.import_module(name))
              for name in (*PUBLIC_MODULES, *_NATIVE_MODULES)]
    checks += [(f'alias {name}', lambda name=name: getattr(np, name.split('.')[-1]))
               for name in PUBLIC_ALIASED_MODULES]
    checks += [(f'import {name}', lambda name=name: _import_legacy(name))
               for name in _LEGACY_MODULES]
    checks += [
        ('array and dtype', _arrays),
        ('ufuncs and matrix multiply', _ufuncs),
        ('linear algebra', _linalg),
        ('FFT', _fft),
        ('random', _random),
        ('strings', _strings),
        ('masked arrays', _masked),
        ('polynomial', _polynomial),
        ('NPY serialization', _io),
    ]
    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            failures += 1
            print(f'FAIL {name}: {type(exc).__name__}: {exc}')
        else:
            if verbose >= 2:
                print(f'PASS {name}')
    if verbose:
        print(f'NumPy smoke: {len(checks) - failures}/{len(checks)} checks passed')
    return failures == 0
