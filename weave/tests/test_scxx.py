""" Test refcounting and behavior of SCXX.
"""
import unittest
import time
import os,sys
from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import inline_tools
restore_path()

def test_suite(level=1):
    from unittest import makeSuite
    suites = []    
    if level >= 5:
        import test_scxx_object
        suites.append( test_scxx_object.test_suite(level))
        import test_scxx_sequence
        suites.append( test_scxx_sequence.test_suite(level))
        import test_scxx_dict
        suites.append( test_scxx_dict.test_suite(level))
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10,verbose=2):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner(verbosity=verbose)
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
