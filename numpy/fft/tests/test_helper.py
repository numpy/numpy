#!/usr/bin/env python
# Copied from fftpack.helper by Pearu Peterson, October 2005
""" Test functions for fftpack.helper module
"""

from numpy.testing import *
from numpy.fft import fftshift,ifftshift,fftfreq

from numpy import pi

def random(size):
    return rand(*size)

class TestFFTShift(TestCase):
    def test_definition(self):
        x = [0,1,2,3,4,-4,-3,-2,-1]
        y = [-4,-3,-2,-1,0,1,2,3,4]
        assert_array_almost_equal(fftshift(x),y)
        assert_array_almost_equal(ifftshift(y),x)
        x = [0,1,2,3,4,-5,-4,-3,-2,-1]
        y = [-5,-4,-3,-2,-1,0,1,2,3,4]
        assert_array_almost_equal(fftshift(x),y)
        assert_array_almost_equal(ifftshift(y),x)

    def test_inverse(self):
        for n in [1,4,9,100,211]:
            x = random((n,))
            assert_array_almost_equal(ifftshift(fftshift(x)),x)
            
    def test_axes_keyword(self):
        freqs = [[ 0,  1,  2], [ 3,  4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [ 2,  0,  1], [-4,  3,  4]]
        assert_array_almost_equal(fftshift(freqs, axes=(0, 1)), shifted)
        assert_array_almost_equal(fftshift(freqs, axes=0), fftshift(freqs, axes=(0,)))
        assert_array_almost_equal(ifftshift(shifted, axes=(0, 1)), freqs)
        assert_array_almost_equal(ifftshift(shifted, axes=0), ifftshift(shifted, axes=(0,)))


class TestFFTFreq(TestCase):
    def test_definition(self):
        x = [0,1,2,3,4,-4,-3,-2,-1]
        assert_array_almost_equal(9*fftfreq(9),x)
        assert_array_almost_equal(9*pi*fftfreq(9,pi),x)
        x = [0,1,2,3,4,-5,-4,-3,-2,-1]
        assert_array_almost_equal(10*fftfreq(10),x)
        assert_array_almost_equal(10*pi*fftfreq(10,pi),x)


if __name__ == "__main__":
    run_module_suite()
