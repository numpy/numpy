''' Define test function for scipy package

Module tests for presence of useful version of nose.  If present
returns NoseTester, otherwise returns a placeholder test routine
reporting lack of nose and inability to run tests.  Typical use is in
module __init__:

from scipy.testing.pkgtester import Tester
test = Tester().test

See nosetester module for test implementation

'''
fine_nose = True
try:
    import nose
except ImportError:
    fine_nose = False
else:
    nose_version = nose.__versioninfo__
    if nose_version[0] < 1 and nose_version[1] < 10:
        fine_nose = False

if fine_nose:
    from numpy.testing.nosetester import NoseTester as Tester
else:
    from numpy.testing.nulltester import NullTester as Tester
