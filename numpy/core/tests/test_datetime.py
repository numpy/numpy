from os import path
import numpy as np
from numpy.testing import *

class TestDateTime(TestCase):
    def test_creation(self):
        for unit in ['Y', 'M', 'W', 'B', 'D',
                     'h', 'm', 's', 'ms', 'us',
                     'ns', 'ps', 'fs', 'as']:
            dt1 = np.dtype('M8[750%s]'%unit)
            assert dt1 == np.dtype('datetime64[750%s]' % unit)
            dt2 = np.dtype('m8[%s]' % unit)
            assert dt2 == np.dtype('timedelta64[%s]' % unit)


    def test_hours(self):
        t = np.ones(3, dtype='M8[s]')
        t[0] = 60*60*24 + 60*60*10
        assert t[0].item().hour == 10 

    def test_divisor_conversion_year(self):
        assert np.dtype('M8[Y/4]') == np.dtype('M8[3M]')
        assert np.dtype('M8[Y/13]') == np.dtype('M8[4W]')
        assert np.dtype('M8[3Y/73]') == np.dtype('M8[15D]')

    def test_divisor_conversion_month(self):
        assert np.dtype('M8[M/2]') == np.dtype('M8[2W]')
        assert np.dtype('M8[M/15]') == np.dtype('M8[2D]')
        assert np.dtype('M8[3M/40]') == np.dtype('M8[54h]')

    def test_divisor_conversion_week(self):
        assert np.dtype('m8[W/5]') == np.dtype('m8[B]')
        assert np.dtype('m8[W/7]') == np.dtype('m8[D]')
        assert np.dtype('m8[3W/14]') == np.dtype('m8[36h]')
        assert np.dtype('m8[5W/140]') == np.dtype('m8[360m]')

    def test_divisor_conversion_bday(self):
        assert np.dtype('M8[B/12]') == np.dtype('M8[2h]')
        assert np.dtype('M8[B/120]') == np.dtype('M8[12m]')
        assert np.dtype('M8[3B/960]') == np.dtype('M8[270s]')

    def test_divisor_conversion_day(self):
        assert np.dtype('M8[D/12]') == np.dtype('M8[2h]')
        assert np.dtype('M8[D/120]') == np.dtype('M8[12m]')
        assert np.dtype('M8[3D/960]') == np.dtype('M8[270s]')

    def test_divisor_conversion_hour(self):
        assert np.dtype('m8[h/30]') == np.dtype('m8[2m]')
        assert np.dtype('m8[3h/300]') == np.dtype('m8[36s]')

    def test_divisor_conversion_minute(self):
        assert np.dtype('m8[m/30]') == np.dtype('m8[2s]')
        assert np.dtype('m8[3m/300]') == np.dtype('m8[600ms]')

    def test_divisor_conversion_second(self):
        assert np.dtype('m8[s/100]') == np.dtype('m8[10ms]')
        assert np.dtype('m8[3s/10000]') == np.dtype('m8[300us]')

    def test_divisor_conversion_fs(self):
        assert np.dtype('M8[fs/100]') == np.dtype('M8[10as]')
        self.assertRaises(ValueError, lambda : np.dtype('M8[3fs/10000]'))

    def test_divisor_conversion_as(self):
        self.assertRaises(ValueError, lambda : np.dtype('M8[as/10]'))

    def test_creation_overflow(self):
        date = '1980-03-23 20:00:00'
        timesteps = np.array([date], dtype='datetime64[s]')[0].astype(np.int64)
        for unit in ['ms', 'us', 'ns']:
            timesteps *= 1000
            x = np.array([date], dtype='datetime64[%s]' % unit)

            assert_equal(timesteps, x[0].astype(np.int64),
                         err_msg='Datetime conversion error for unit %s' % unit)

        assert_equal(x[0].astype(np.int64), 322689600000000000)

if __name__ == "__main__":
    run_module_suite()
