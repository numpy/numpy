#! Last Change: Tue Oct 23 08:00 PM 2007 J

import sys
import random

import unittest

from fortran import parse_f77link

from fortran_output import g77_link_output, gfortran_link_output, \
        sunfort_v12_link_output, ifort_v10_link_output

class TestCheckF77Verbose(unittest.TestCase):
    def setUp(self):
        pass

    def test_g77(self):
        print parse_f77link(g77_link_output.split('\n'))

    def test_gfortran(self):
        print parse_f77link(gfortran_link_output.split('\n'))

    def test_sunf77(self):
        print parse_f77link(sunfort_v12_link_output.split('\n'))

    def test_intel_posix(self):
        print parse_f77link(ifort_v10_link_output.split('\n'))

    def test_intel_win(self):
        print "FIXME: testing verbose output of win32 intel fortran"

if __name__ == '__main__':
    unittest.main()
