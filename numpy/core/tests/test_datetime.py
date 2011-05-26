import os, pickle
import numpy as np
from numpy.testing import *
from numpy.compat import asbytes
import datetime

class TestDateTime(TestCase):
    def test_creation(self):
        for unit in ['Y', 'M', 'W', 'B', 'D',
                     'h', 'm', 's', 'ms', 'us',
                     'ns', 'ps', 'fs', 'as']:
            dt1 = np.dtype('M8[750%s]'%unit)
            assert_(dt1 == np.dtype('datetime64[750%s]' % unit))
            dt2 = np.dtype('m8[%s]' % unit)
            assert_(dt2 == np.dtype('timedelta64[%s]' % unit))
        
        # Check that the parser rejects bad datetime types
        assert_raises(TypeError, np.dtype, 'M8[badunit]')
        assert_raises(TypeError, np.dtype, 'm8[badunit]')
        assert_raises(TypeError, np.dtype, 'M8[YY]')
        assert_raises(TypeError, np.dtype, 'm8[YY]')
        assert_raises(TypeError, np.dtype, 'M4')
        assert_raises(TypeError, np.dtype, 'm4')
        assert_raises(TypeError, np.dtype, 'M7')
        assert_raises(TypeError, np.dtype, 'm7')
        assert_raises(TypeError, np.dtype, 'M16')
        assert_raises(TypeError, np.dtype, 'm16')

    def test_dtype_comparison(self):
        assert_(not (np.dtype('M8[us]') == np.dtype('M8[ms]')))
        assert_(np.dtype('M8[us]') != np.dtype('M8[ms]'))
        assert_(np.dtype('M8[D]') != np.dtype('M8[B]'))
        assert_(np.dtype('M8[2D]') != np.dtype('M8[D]'))
        assert_(np.dtype('M8[D]') != np.dtype('M8[2D]'))
        assert_(np.dtype('M8[Y]//3') != np.dtype('M8[Y]'))

    def test_pydatetime_creation(self):
        a = np.array(['1960-03-12', datetime.date(1960, 3, 12)], dtype='M8[D]')
        assert_equal(a[0], a[1])
        a = np.array(['1960-03-12', datetime.date(1960, 3, 12)], dtype='M8[s]')
        assert_equal(a[0], a[1])
        a = np.array(['1999-12-31', datetime.date(1999, 12, 31)], dtype='M8[D]')
        assert_equal(a[0], a[1])
        a = np.array(['1999-12-31', datetime.date(1999, 12, 31)], dtype='M8[s]')
        assert_equal(a[0], a[1])
        a = np.array(['2000-01-01', datetime.date(2000, 1, 1)], dtype='M8[D]')
        assert_equal(a[0], a[1])
        a = np.array(['2000-01-01', datetime.date(2000, 1, 1)], dtype='M8[s]')
        assert_equal(a[0], a[1])
        # Will fail if the date changes during the exact right moment
        a = np.array(['today', datetime.date.today()], dtype='M8[s]')
        assert_equal(a[0], a[1])
        # datetime.datetime.now() returns local time, not UTC
        #a = np.array(['now', datetime.datetime.now()], dtype='M8[s]')
        #assert_equal(a[0], a[1])

    def test_pickle(self):
        # Check that pickle roundtripping works
        dt = np.dtype('M8[7D]//3')
        assert_equal(dt, pickle.loads(pickle.dumps(dt)))
        dt = np.dtype('M8[B]')
        assert_equal(dt, pickle.loads(pickle.dumps(dt)))

    def test_dtype_promotion(self):
        # datetime <op> datetime computes the metadata gcd
        # timedelta <op> timedelta computes the metadata gcd
        for mM in ['m', 'M']:
            assert_equal(
                np.promote_types(np.dtype(mM+'8[2Y]'), np.dtype(mM+'8[2Y]')),
                np.dtype(mM+'8[2Y]'))
            assert_equal(
                np.promote_types(np.dtype(mM+'8[12Y]'), np.dtype(mM+'8[15Y]')),
                np.dtype(mM+'8[3Y]'))
            assert_equal(
                np.promote_types(np.dtype(mM+'8[62M]'), np.dtype(mM+'8[24M]')),
                np.dtype(mM+'8[2M]'))
            assert_equal(
                np.promote_types(np.dtype(mM+'8[1W]'), np.dtype(mM+'8[2D]')),
                np.dtype(mM+'8[1D]'))
            assert_equal(
                np.promote_types(np.dtype(mM+'8[W]'), np.dtype(mM+'8[13s]')),
                np.dtype(mM+'8[s]'))
            assert_equal(
                np.promote_types(np.dtype(mM+'8[13W]'), np.dtype(mM+'8[49s]')),
                np.dtype(mM+'8[7s]'))
        # timedelta <op> timedelta raises when there is no reasonable gcd
        assert_raises(TypeError, np.promote_types,
                            np.dtype('m8[Y]'), np.dtype('m8[D]'))
        assert_raises(TypeError, np.promote_types,
                            np.dtype('m8[Y]'), np.dtype('m8[B]'))
        assert_raises(TypeError, np.promote_types,
                            np.dtype('m8[D]'), np.dtype('m8[B]'))
        assert_raises(TypeError, np.promote_types,
                            np.dtype('m8[M]'), np.dtype('m8[W]'))
        # timedelta <op> timedelta may overflow with big unit ranges
        assert_raises(OverflowError, np.promote_types,
                            np.dtype('m8[W]'), np.dtype('m8[fs]'))
        assert_raises(OverflowError, np.promote_types,
                            np.dtype('m8[s]'), np.dtype('m8[as]'))


    def test_pyobject_roundtrip(self):
        # All datetime types should be able to roundtrip through object
        a = np.array([0,0,0,0,0,0,0,0,
                      -1020040340, -2942398, -1, 0, 1, 234523453, 1199164176],
                                                        dtype=np.int64)
        for unit in ['M8[as]', 'M8[16fs]', 'M8[ps]', 'M8[us]',
                     'M8[as]//12', 'M8[us]//16', 'M8[D]', 'M8[D]//4',
                     'M8[W]', 'M8[M]', 'M8[Y]']:
            b = a.copy().view(dtype=unit)
            b[0] = '-0001-01-01'
            b[1] = '-0001-12-31'
            b[2] = '0000-01-01'
            b[3] = '0001-01-01'
            b[4] = '1969-12-31T23:59:59.999999Z'
            b[5] = '1970-01-01'
            b[6] = '9999-12-31T23:59:59.999999Z'
            b[7] = '10000-01-01'

            assert_equal(b.astype(object).astype(unit), b,
                            "Error roundtripping unit %s" % unit)

    def test_month_truncation(self):
        # Make sure that months are truncating correctly
        assert_equal(np.array('1945-03-01', dtype='M8[M]'),
                     np.array('1945-03-31', dtype='M8[M]'))
        assert_equal(np.array('1969-11-01', dtype='M8[M]'),
                     np.array('1969-11-30T23:59:59.999999Z', dtype='M8[M]'))
        assert_equal(np.array('1969-12-01', dtype='M8[M]'),
                     np.array('1969-12-31T23:59:59.999999Z', dtype='M8[M]'))
        assert_equal(np.array('1970-01-01', dtype='M8[M]'),
                     np.array('1970-01-31T23:59:59.999999Z', dtype='M8[M]'))
        assert_equal(np.array('1980-02-01', dtype='M8[M]'),
                     np.array('1980-02-29T23:59:59.999999Z', dtype='M8[M]'))

    def test_different_unit_comparison(self):
        # Check some years
        for unit1 in ['Y', 'M', 'D', '6h', 'h', 'm', 's', '10ms',
                                                        'ms', 'us', 'ps']:
            dt1 = np.dtype('M8[%s]' % unit1)
            for unit2 in ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ps']:
                dt2 = np.dtype('M8[%s]' % unit2)
                assert_equal(np.array('1945', dtype=dt1),
                             np.array('1945', dtype=dt2))
                assert_equal(np.array('1970', dtype=dt1),
                             np.array('1970', dtype=dt2))
                assert_equal(np.array('9999', dtype=dt1),
                             np.array('9999', dtype=dt2))
                assert_equal(np.array('10000', dtype=dt1),
                             np.array('10000-01-01', dtype=dt2))
        # Check some days
        for unit1 in ['D', '12h', 'h', 'm', 's', '4s', 'ms', 'us', 'ps']:
            dt1 = np.dtype('M8[%s]' % unit1)
            for unit2 in ['D', 'h', 'm', 's', 'ms', 'us', 'ps']:
                dt2 = np.dtype('M8[%s]' % unit2)
                assert_equal(np.array('1932-02-17', dtype=dt1),
                             np.array('1932-02-17T00:00:00', dtype=dt2))
                assert_equal(np.array('10000-04-27', dtype=dt1),
                             np.array('10000-04-27T00:00:00', dtype=dt2))
                assert_equal(np.array('today', dtype=dt1),
                             np.array('today', dtype=dt2))

    def test_hours(self):
        t = np.ones(3, dtype='M8[s]')
        t[0] = 60*60*24 + 60*60*10
        assert_(t[0].item().hour == 10 )

    def test_divisor_conversion_year(self):
        assert_(np.dtype('M8[Y/4]') == np.dtype('M8[3M]'))
        assert_(np.dtype('M8[Y/13]') == np.dtype('M8[4W]'))
        assert_(np.dtype('M8[3Y/73]') == np.dtype('M8[15D]'))

    def test_divisor_conversion_month(self):
        assert_(np.dtype('M8[M/2]') == np.dtype('M8[2W]'))
        assert_(np.dtype('M8[M/15]') == np.dtype('M8[2D]'))
        assert_(np.dtype('M8[3M/40]') == np.dtype('M8[54h]'))

    def test_divisor_conversion_week(self):
        assert_(np.dtype('m8[W/5]') == np.dtype('m8[B]'))
        assert_(np.dtype('m8[W/7]') == np.dtype('m8[D]'))
        assert_(np.dtype('m8[3W/14]') == np.dtype('m8[36h]'))
        assert_(np.dtype('m8[5W/140]') == np.dtype('m8[360m]'))

    def test_divisor_conversion_bday(self):
        assert_(np.dtype('M8[B/12]') == np.dtype('M8[2h]'))
        assert_(np.dtype('M8[B/120]') == np.dtype('M8[12m]'))
        assert_(np.dtype('M8[3B/960]') == np.dtype('M8[270s]'))

    def test_divisor_conversion_day(self):
        assert_(np.dtype('M8[D/12]') == np.dtype('M8[2h]'))
        assert_(np.dtype('M8[D/120]') == np.dtype('M8[12m]'))
        assert_(np.dtype('M8[3D/960]') == np.dtype('M8[270s]'))

    def test_divisor_conversion_hour(self):
        assert_(np.dtype('m8[h/30]') == np.dtype('m8[2m]'))
        assert_(np.dtype('m8[3h/300]') == np.dtype('m8[36s]'))

    def test_divisor_conversion_minute(self):
        assert_(np.dtype('m8[m/30]') == np.dtype('m8[2s]'))
        assert_(np.dtype('m8[3m/300]') == np.dtype('m8[600ms]'))

    def test_divisor_conversion_second(self):
        assert_(np.dtype('m8[s/100]') == np.dtype('m8[10ms]'))
        assert_(np.dtype('m8[3s/10000]') == np.dtype('m8[300us]'))

    def test_divisor_conversion_fs(self):
        assert_(np.dtype('M8[fs/100]') == np.dtype('M8[10as]'))
        self.assertRaises(ValueError, lambda : np.dtype('M8[3fs/10000]'))

    def test_divisor_conversion_as(self):
        self.assertRaises(ValueError, lambda : np.dtype('M8[as/10]'))

    def test_string_parser_variants(self):
        """
        # Different month formats
        assert_equal(np.array(['1980-02-29'], np.dtype('M8')),
                     np.array(['1980-Feb-29'], np.dtype('M8')))
        assert_equal(np.array(['1980-02-29'], np.dtype('M8')),
                     np.array(['1980-feb-29'], np.dtype('M8')))
        assert_equal(np.array(['1980-02-29'], np.dtype('M8')),
                     np.array(['1980-FEB-29'], np.dtype('M8')))
        """
        # Allow space instead of 'T' between date and time
        assert_equal(np.array(['1980-02-29T01:02:03'], np.dtype('M8')),
                     np.array(['1980-02-29 01:02:03'], np.dtype('M8')))
        # Allow negative years
        assert_equal(np.array(['-1980-02-29T01:02:03'], np.dtype('M8')),
                     np.array(['-1980-02-29 01:02:03'], np.dtype('M8')))
        # UTC specifier
        assert_equal(np.array(['-1980-02-29T01:02:03Z'], np.dtype('M8')),
                     np.array(['-1980-02-29 01:02:03Z'], np.dtype('M8')))
        # Time zone offset
        assert_equal(np.array(['1980-02-29T02:02:03Z'], np.dtype('M8')),
                     np.array(['1980-02-29 00:32:03+0130'], np.dtype('M8')))
        assert_equal(np.array(['1980-02-28T22:32:03Z'], np.dtype('M8')),
                     np.array(['1980-02-29 00:02:03-01:30'], np.dtype('M8')))
        assert_equal(np.array(['1980-02-29T02:32:03.506Z'], np.dtype('M8')),
                     np.array(['1980-02-29 00:32:03.506+02'], np.dtype('M8')))

    def test_string_parser_error_check(self):
        # Arbitrary bad string
        assert_raises(ValueError, np.array, ['badvalue'], np.dtype('M8'))
        # Character after year must be '-'
        assert_raises(ValueError, np.array, ['1980X'], np.dtype('M8'))
        # Cannot have trailing '-'
        assert_raises(ValueError, np.array, ['1980-'], np.dtype('M8'))
        # Month must be in range [1,12]
        assert_raises(ValueError, np.array, ['1980-00'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-13'], np.dtype('M8'))
        # Month must have two digits
        assert_raises(ValueError, np.array, ['1980-1'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-1-02'], np.dtype('M8'))
        # 'Mor' is not a valid month
        assert_raises(ValueError, np.array, ['1980-Mor'], np.dtype('M8'))
        # Cannot have trailing '-'
        assert_raises(ValueError, np.array, ['1980-01-'], np.dtype('M8'))
        # Day must be in range [1,len(month)]
        assert_raises(ValueError, np.array, ['1980-01-0'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-01-00'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-01-32'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1979-02-29'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-30'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-03-32'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-04-31'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-05-32'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-06-31'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-07-32'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-08-32'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-09-31'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-10-32'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-11-31'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-12-32'], np.dtype('M8'))
        # Cannot have trailing characters
        assert_raises(ValueError, np.array, ['1980-02-03%'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 q'], np.dtype('M8'))

        # Hours must be in range [0, 23]
        assert_raises(ValueError, np.array, ['1980-02-03 25'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03T25'], np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 24:01'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03T24:01'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 -1'], np.dtype('M8'))
        # No trailing ':'
        assert_raises(ValueError, np.array, ['1980-02-03 01:'], np.dtype('M8'))
        # Minutes must be in range [0, 59]
        assert_raises(ValueError, np.array, ['1980-02-03 01:-1'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 01:60'],
                                                        np.dtype('M8'))
        # No trailing ':'
        assert_raises(ValueError, np.array, ['1980-02-03 01:60:'],
                                                        np.dtype('M8'))
        # Seconds must be in range [0, 59]
        assert_raises(ValueError, np.array, ['1980-02-03 01:10:-1'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:60'],
                                                        np.dtype('M8'))
        # Timezone offset must within a reasonable range
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00+0661'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00+2500'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00-0070'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00-3000'],
                                                        np.dtype('M8'))
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00-25:00'],
                                                        np.dtype('M8'))


    def test_creation_overflow(self):
        date = '1980-03-23 20:00:00Z'
        timesteps = np.array([date], dtype='datetime64[s]')[0].astype(np.int64)
        for unit in ['ms', 'us', 'ns']:
            timesteps *= 1000
            x = np.array([date], dtype='datetime64[%s]' % unit)

            assert_equal(timesteps, x[0].astype(np.int64),
                         err_msg='Datetime conversion error for unit %s' % unit)

        assert_equal(x[0].astype(np.int64), 322689600000000000)

class TestDateTimeData(TestCase):

    def test_basic(self):
        a = np.array(['1980-03-23'], dtype=np.datetime64)
        assert_equal(np.datetime_data(a.dtype), (asbytes('us'), 1, 1))

if __name__ == "__main__":
    run_module_suite()
