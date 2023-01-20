from numpy.testing import assert_

import numbers

import numpy as np
from numpy.core.numerictypes import sctypes

class TestABC:
    def test_abstract(self):
        assertTrue(issubclass(np.number, numbers.Number))

        assertTrue(issubclass(np.inexact, numbers.Complex))
        assertTrue(issubclass(np.complexfloating, numbers.Complex))
        assertTrue(issubclass(np.floating, numbers.Real))

        assertTrue(issubclass(np.integer, numbers.Integral))
        assertTrue(issubclass(np.signedinteger, numbers.Integral))
        assertTrue(issubclass(np.unsignedinteger, numbers.Integral))

    def test_floats(self):
        for t in sctypes['float']:
            assertTrue(isinstance(t(), numbers.Real),
                    f"{t.__name__} is not instance of Real")
            assertTrue(issubclass(t, numbers.Real),
                    f"{t.__name__} is not subclass of Real")
            assertTrue(not isinstance(t(), numbers.Rational),
                    f"{t.__name__} is instance of Rational")
            assertTrue(not issubclass(t, numbers.Rational),
                    f"{t.__name__} is subclass of Rational")

    def test_complex(self):
        for t in sctypes['complex']:
            assertTrue(isinstance(t(), numbers.Complex),
                    f"{t.__name__} is not instance of Complex")
            assertTrue(issubclass(t, numbers.Complex),
                    f"{t.__name__} is not subclass of Complex")
            assertTrue(not isinstance(t(), numbers.Real),
                    f"{t.__name__} is instance of Real")
            assertTrue(not issubclass(t, numbers.Real),
                    f"{t.__name__} is subclass of Real")

    def test_int(self):
        for t in sctypes['int']:
            assertTrue(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            assertTrue(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")

    def test_uint(self):
        for t in sctypes['uint']:
            assertTrue(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            assertTrue(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")
