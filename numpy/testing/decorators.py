"""Decorators for labeling test objects

Decorators that merely return a modified version of the original
function object are straightforward.  Decorators that return a new
function object need to use
nose.tools.make_decorator(original_function)(decorator) in returning
the decorator, in order to preserve metadata such as function name,
setup and teardown functions and so on - see nose.tools for more
information.

"""

def slow(t):
    """Labels a test as 'slow'.

    The exact definition of a slow test is obviously both subjective and
    hardware-dependent, but in general any individual test that requires more
    than a second or two should be labeled as slow (the whole suite consits of
    thousands of tests, so even a second is significant)."""

    t.slow = True
    return t

def setastest(tf=True):
    ''' Signals to nose that this function is or is not a test

    Parameters
    ----------
    tf : bool
        If True specifies this is a test, not a test otherwise

    e.g
    >>> from numpy.testing.decorators import setastest
    >>> @setastest(False)
    ... def func_with_test_in_name(arg1, arg2): pass
    ...
    >>>

    This decorator cannot use the nose namespace, because it can be
    called from a non-test module. See also istest and nottest in
    nose.tools

    '''
    def set_test(t):
        t.__test__ = tf
        return t
    return set_test

def skipif(skip_condition, msg=None):
    ''' Make function raise SkipTest exception if skip_condition is true

    Parameters
    ---------
    skip_condition : bool
        Flag to determine whether to skip test (True) or not (False)
    msg : string
        Message to give on raising a SkipTest exception

   Returns
   -------
   decorator : function
       Decorator, which, when applied to a function, causes SkipTest
       to be raised when the skip_condition was True, and the function
       to be called normally otherwise.

    Notes
    -----
    You will see from the code that we had to further decorate the
    decorator with the nose.tools.make_decorator function in order to
    transmit function name, and various other metadata.
    '''
    if msg is None:
        msg = 'Test skipped due to test condition'
    def skip_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose
        def skipper(*args, **kwargs):
            if skip_condition:
                raise nose.SkipTest, msg
            else:
                return f(*args, **kwargs)
        return nose.tools.make_decorator(f)(skipper)
    return skip_decorator

def skipknownfailure(f):
    ''' Decorator to raise SkipTest for test known to fail
    '''
    # Local import to avoid a hard nose dependency and only incur the
    # import time overhead at actual test-time.
    import nose
    def skipper(*args, **kwargs):
        raise nose.SkipTest, 'This test is known to fail'
    return nose.tools.make_decorator(f)(skipper)
