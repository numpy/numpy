"""Support for parametric tests in unittest.

:Author: Fernando Perez

Purpose
=======

Briefly, the main class in this module allows you to easily and cleanly
(without the gross name-mangling hacks that are normally needed) to write
unittest TestCase classes that have parametrized tests.  That is, tests which
consist of multiple sub-tests that scan for example a parameter range, but
where you want each sub-test to:

* count as a separate test in the statistics.

* be run even if others in the group error out or fail.


The class offers a simple name-based convention to create such tests (see
simple example at the end), in one of two ways:

* Each sub-test in a group can be run fully independently, with the
  setUp/tearDown methods being called each time.

* The whole group can be run with setUp/tearDown being called only once for the
  group.  This lets you conveniently reuse state that may be very expensive to
  compute for multiple tests.  Be careful not to corrupt it!!!


Caveats
=======

This code relies on implementation details of the unittest module (some key
methods are heavily modified versions of those, after copying them in).  So it
may well break either if you make sophisticated use of the unittest APIs, or if
unittest itself changes in the future.  I have only tested this with Python
2.5.

"""
__docformat__ = "restructuredtext en"

import unittest

class ParametricTestCase(unittest.TestCase):
    """TestCase subclass with support for parametric tests.

    Subclasses of this class can implement test methods that return a list of
    tests and arguments to call those with, to do parametric testing (often
    also called 'data driven' testing."""

    #: Prefix for tests with independent state.  These methods will be run with
    #: a separate setUp/tearDown call for each test in the group.
    _indepParTestPrefix = 'testip'

    #: Prefix for tests with shared state.  These methods will be run with
    #: a single setUp/tearDown call for the whole group.  This is useful when
    #: writing a group of tests for which the setup is expensive and one wants
    #: to actually share that state.  Use with care (especially be careful not
    #: to mutate the state you are using, which will alter later tests).
    _shareParTestPrefix = 'testsp'

    def exec_test(self,test,args,result):
        """Execute a single test.  Returns a success boolean"""

        ok = False
        try:
            test(*args)
            ok = True
        except self.failureException:
            result.addFailure(self, self._exc_info())
        except KeyboardInterrupt:
            raise
        except:
            result.addError(self, self._exc_info())

        return ok


    def run_test(self, testInfo,result):
        """Run one test with arguments"""

        test,args = testInfo[0],testInfo[1:]

        # Reset the doc attribute to be the docstring of this particular test,
        # so that in error messages it prints the actual test's docstring and
        # not that of the test factory.
        self._testMethodDoc = test.__doc__
        result.startTest(self)
        try:
            try:
                self.setUp()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
                return

            ok = self.exec_test(test,args,result)

            try:
                self.tearDown()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
                ok = False
            if ok: result.addSuccess(self)
        finally:
            result.stopTest(self)

    def run_tests(self, tests,result):
        """Run many tests with a common setUp/tearDown.

        The entire set of tests is run with a single setUp/tearDown call."""

        try:
            self.setUp()
        except KeyboardInterrupt:
            raise
        except:
            result.testsRun += 1
            result.addError(self, self._exc_info())
            return

        saved_doc = self._testMethodDoc

        try:
            # Run all the tests specified
            for testInfo in tests:
                test,args = testInfo[0],testInfo[1:]

                # Set the doc argument for this test.  Note that even if we do
                # this, the fail/error tracebacks still print the docstring for
                # the parent factory, because they only generate the message at
                # the end of the run, AFTER we've restored it.  There is no way
                # to tell the unittest system (without overriding a lot of
                # stuff) to extract this information right away, the logic is
                # hardcoded to pull it later, since unittest assumes it doesn't
                # change.
                self._testMethodDoc = test.__doc__
                result.startTest(self)
                ok = self.exec_test(test,args,result)
                if ok: result.addSuccess(self)

        finally:
            # Restore docstring info and run tearDown once only.
            self._testMethodDoc = saved_doc
            try:
                self.tearDown()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())

    def run(self, result=None):
        """Test runner."""

        #print
        #print '*** run for method:',self._testMethodName  # dbg
        #print '***            doc:',self._testMethodDoc  # dbg

        if result is None: result = self.defaultTestResult()

        # Independent tests: each gets its own setup/teardown
        if self._testMethodName.startswith(self._indepParTestPrefix):
            for t in getattr(self, self._testMethodName)():
                self.run_test(t,result)
        # Shared-state test: single setup/teardown for all
        elif self._testMethodName.startswith(self._shareParTestPrefix):
            tests = getattr(self, self._testMethodName)()
            self.run_tests(tests,result)
        # Normal unittest Test methods
        else:
            unittest.TestCase.run(self,result)

#############################################################################
# Quick and dirty interactive example/test
if __name__ == '__main__':

    class ExampleTestCase(ParametricTestCase):

        #-------------------------------------------------------------------
        # An instrumented setUp method so we can see when it gets called and
        # how many times per instance
        counter = 0

        def setUp(self):
            self.counter += 1
            print 'setUp count: %2s for: %s' % (self.counter,
                                                self._testMethodDoc)

        #-------------------------------------------------------------------
        # A standard test method, just like in the unittest docs.
        def test_foo(self):
            """Normal test for feature foo."""
            pass

        #-------------------------------------------------------------------
        # Testing methods that need parameters.  These can NOT be named test*,
        # since they would be picked up by unittest and called without
        # arguments.  Instead, call them anything else (I use tst*) and then
        # load them via the factories below.
        def tstX(self,i):
            "Test feature X with parameters."
            print 'tstX, i=',i
            if i==1 or i==3:
                # Test fails
                self.fail('i is bad, bad: %s' % i)

        def tstY(self,i):
            "Test feature Y with parameters."
            print 'tstY, i=',i
            if i==1:
                # Force an error
                1/0

        def tstXX(self,i,j):
            "Test feature XX with parameters."
            print 'tstXX, i=',i,'j=',j
            if i==1:
                # Test fails
                self.fail('i is bad, bad: %s' % i)

        def tstYY(self,i):
            "Test feature YY with parameters."
            print 'tstYY, i=',i
            if i==2:
                # Force an error
                1/0

        def tstZZ(self):
            """Test feature ZZ without parameters, needs multiple runs.

            This could be a random test that you want to run multiple times."""
            pass

        #-------------------------------------------------------------------
        # Parametric test factories that create the test groups to call the
        # above tst* methods with their required arguments.
        def testip(self):
            """Independent parametric test factory.

            A separate setUp() call is made for each test returned by this
            method.

            You must return an iterable (list or generator is fine) containing
            tuples with the actual method to be called as the first argument,
            and the arguments for that call later."""
            return ((self.tstX,i) for i in range(5))

        def testip2(self):
            """Another independent parametric test factory"""
            return ((self.tstY,i) for i in range(5))

        def testip3(self):
            """Test factory combining different subtests.

            This one shows how to assemble calls to different tests."""
            return [(self.tstX,3),(self.tstX,9),(self.tstXX,4,10),
                    (self.tstZZ,),(self.tstZZ,)]

        def testsp(self):
            """Shared parametric test factory

            A single setUp() call is made for all the tests returned by this
            method.
            """
            return ((self.tstXX,i,i+1) for i in range(5))

        def testsp2(self):
            """Another shared parametric test factory"""
            return ((self.tstYY,i) for i in range(5))

        def testsp3(self):
            """Another shared parametric test factory.

            This one simply calls the same test multiple times, without any
            arguments.  Note that you must still return tuples, even if there
            are no arguments."""
            return ((self.tstZZ,) for i in range(10))


    # This test class runs normally under unittest's default runner
    unittest.main()
