#! Last Change: Mon Nov 12 03:00 PM 2007 J

from numpy.testing import NumpyTestCase, set_package_path, restore_path, set_local_path

set_local_path('..')
from fortran import parse_f77link
restore_path()

set_local_path()
from fortran_output import g77_link_output, gfortran_link_output, \
        sunfort_v12_link_output, ifort_v10_link_output, \
        mingw_g77_link_output, \
        g77_link_expected, gfortran_link_expected, \
        sunfort_v12_link_expected, ifort_v10_link_expected
restore_path()

class test_CheckF77Verbose(NumpyTestCase):
    def setUp(self):
        pass

    def test_g77(self):
        """Parsing g77 link output."""
        assert parse_f77link(g77_link_output.split('\n')) == g77_link_expected

    def test_gfortran(self):
        """Parsing gfortran link output."""
        assert parse_f77link(gfortran_link_output.split('\n')) == \
               gfortran_link_expected

    def test_sunf77(self):
        """Parsing sunfort link output."""
        assert parse_f77link(sunfort_v12_link_output.split('\n')) == \
               sunfort_v12_link_expected

    def test_intel_posix(self):
        """Parsing ifort link output."""
        assert parse_f77link(ifort_v10_link_output.split('\n')) == \
               ifort_v10_link_expected

    def test_intel_win(self):
        """Parsing ifort link output on win32."""
        print "FIXME: testing verbose output of win32 intel fortran"

    def test_mingw_g77(self):
        """Parsing mingw g77 link output on win32 (native, i.e. no cygwin)"""
        print parse_f77link(mingw_g77_link_output.split('\n'))

if __name__ == '__main__':
    from numpy.testing import NumpyTest
    NumpyTest().test()
