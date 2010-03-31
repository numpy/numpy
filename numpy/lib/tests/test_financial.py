from numpy.testing import *
import numpy as np

class TestFinancial(TestCase):
    def test_rate(self):
        assert_almost_equal(np.rate(10,0,-3500,10000),
                            0.1107, 4)

    def test_irr(self):
        v = [-150000, 15000, 25000, 35000, 45000, 60000]
        assert_almost_equal(np.irr(v),
                            0.0524, 2)

    def test_pv(self):
        assert_almost_equal(np.pv(0.07,20,12000,0),
                            -127128.17, 2)

    def test_fv(self):
        assert_almost_equal(np.fv(0.075, 20, -2000,0,0),
                            86609.36, 2)

    def test_pmt(self):
        assert_almost_equal(np.pmt(0.08/12,5*12,15000),
                            -304.146, 3)

    def test_nper(self):
        assert_almost_equal(np.nper(0.075,-2000,0,100000.),
                            21.54, 2)

    def test_nper2(self):
        assert_almost_equal(np.nper(0.0,-2000,0,100000.),
                            50.0, 1)

    def test_npv(self):
        assert_almost_equal(np.npv(0.05,[-15000,1500,2500,3500,4500,6000]),
                            117.04, 2)

    def test_mirr(self):
        val = [-4500,-800,800,800,600,600,800,800,700,3000]
        assert_almost_equal(np.mirr(val, 0.08, 0.055), 0.0666, 4)

        val = [-120000,39000,30000,21000,37000,46000]
        assert_almost_equal(np.mirr(val, 0.10, 0.12), 0.126094, 6)

        val = [100,200,-50,300,-200]
        assert_almost_equal(np.mirr(val, 0.05, 0.06), 0.3428, 4)

        val = [39000,30000,21000,37000,46000]
        assert_(np.isnan(np.mirr(val, 0.10, 0.12)))



def test_unimplemented():
    # np.round(np.ppmt(0.1/12,1,60,55000),2) == 710.25
    assert_raises(NotImplementedError, np.ppmt, 0.1/12, 1, 60, 55000)

    # np.round(np.ipmt(0.1/12,1,24,2000),2) == 16.67
    assert_raises(NotImplementedError, np.ipmt, 0.1/12, 1, 24, 2000)


if __name__ == "__main__":
    run_module_suite()
