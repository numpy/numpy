"""
Decorators for labeling and modifying behavior of test objects.

Decorators that merely return a modified version of the original
function object are straightforward. Decorators that return a new
function object need to use
::

  nose.tools.make_decorator(original_function)(decorator)

in returning the decorator, in order to preserve meta-data such as
function name, setup and teardown functions and so on - see
``nose.tools`` for more information.

"""
import warnings
import sys

def slow(t):
    """
    Label a test as 'slow'.

    The exact definition of a slow test is obviously both subjective and
    hardware-dependent, but in general any individual test that requires more
    than a second or two should be labeled as slow (the whole suite consits of
    thousands of tests, so even a second is significant).

    Parameters
    ----------
    t : callable
        The test to label as slow.

    Returns
    -------
    t : callable
        The decorated test `t`.

    Examples
    --------
    The `numpy.testing` module includes ``import decorators as dec``.
    A test can be decorated as slow like this::

      from numpy.testing import *

      @dec.slow
      def test_big(self):
          print 'Big, slow test'

    """

    t.slow = True
    return t

def setastest(tf=True):
    """
    Signals to nose that this function is or is not a test.

    Parameters
    ----------
    tf : bool
        If True, specifies that the decorated callable is a test.
        If False, specifies that the decorated callable is not a test.
        Default is True.

    Notes
    -----
    This decorator can't use the nose namespace, because it can be
    called from a non-test module. See also ``istest`` and ``nottest`` in
    ``nose.tools``.

    Examples
    --------
    `setastest` can be used in the following way::

      from numpy.testing.decorators import setastest

      @setastest(False)
      def func_with_test_in_name(arg1, arg2):
          pass

    """
    def set_test(t):
        t.__test__ = tf
        return t
    return set_test

def skipif(skip_condition, msg=None):
    """
    Make function raise SkipTest exception if a given condition is true.

    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.

    Parameters
    ----------
    skip_condition : bool or callable
        Flag to determine whether to skip the decorated test.
    msg : str, optional
        Message to give on raising a SkipTest exception. Default is None.

    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, causes SkipTest
        to be raised when `skip_condition` is True, and the function
        to be called normally otherwise.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.

    """

    def skip_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose

        # Allow for both boolean or callable skip conditions.
        if callable(skip_condition):
            skip_val = lambda : skip_condition()
        else:
            skip_val = lambda : skip_condition

        def get_msg(func,msg=None):
            """Skip message with information about function being skipped."""
            if msg is None: 
                out = 'Test skipped due to test condition'
            else: 
                out = '\n'+msg

            return "Skipping test: %s%s" % (func.__name__,out)

        # We need to define *two* skippers because Python doesn't allow both
        # return with value and yield inside the same function.
        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            if skip_val():
                raise nose.SkipTest(get_msg(f,msg))
            else:
                return f(*args, **kwargs)

        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            if skip_val():
                raise nose.SkipTest(get_msg(f,msg))
            else:
                for x in f(*args, **kwargs):
                    yield x

        # Choose the right skipper to use when building the actual decorator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func
            
        return nose.tools.make_decorator(f)(skipper)

    return skip_decorator


def knownfailureif(fail_condition, msg=None):
    """
    Make function raise KnownFailureTest exception if given condition is true.

    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.

    Parameters
    ----------
    fail_condition : bool or callable
        Flag to determine whether to mark the decorated test as a known
        failure (if True) or not (if False).
    msg : str, optional
        Message to give on raising a KnownFailureTest exception.
        Default is None.

    Returns
    -------
    decorator : function
        Decorator, which, when applied to a function, causes SkipTest
        to be raised when `skip_condition` is True, and the function
        to be called normally otherwise.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.

    """
    if msg is None:
        msg = 'Test skipped due to known failure'

    # Allow for both boolean or callable known failure conditions.
    if callable(fail_condition):
        fail_val = lambda : fail_condition()
    else:
        fail_val = lambda : fail_condition

    def knownfail_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose
        from noseclasses import KnownFailureTest
        def knownfailer(*args, **kwargs):
            if fail_val():
                raise KnownFailureTest, msg
            else:
                return f(*args, **kwargs)
        return nose.tools.make_decorator(f)(knownfailer)

    return knownfail_decorator

# The following two classes are copied from python 2.6 warnings module (context
# manager)
class WarningMessage(object):

    """
    Holds the result of a single showwarning() call.

    Notes
    -----
    `WarningMessage` is copied from the Python 2.6 warnings module,
    so it can be used in NumPy with older Python versions.

    """

    _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file",
                        "line")

    def __init__(self, message, category, filename, lineno, file=None,
                    line=None):
        local_values = locals()
        for attr in self._WARNING_DETAILS:
            setattr(self, attr, local_values[attr])
        if category:
            self._category_name = category.__name__
        else:
            self._category_name = None

    def __str__(self):
        return ("{message : %r, category : %r, filename : %r, lineno : %s, "
                    "line : %r}" % (self.message, self._category_name,
                                    self.filename, self.lineno, self.line))

class WarningManager:
    """
    A context manager that copies and restores the warnings filter upon
    exiting the context.

    The 'record' argument specifies whether warnings should be captured by a
    custom implementation of ``warnings.showwarning()`` and be appended to a
    list returned by the context manager. Otherwise None is returned by the
    context manager. The objects appended to the list are arguments whose
    attributes mirror the arguments to ``showwarning()``.

    The 'module' argument is to specify an alternative module to the module
    named 'warnings' and imported under that name. This argument is only useful
    when testing the warnings module itself.

    Notes
    -----
    `WarningManager` is a copy of the ``catch_warnings`` context manager
    from the Python 2.6 warnings module, with slight modifications.
    It is copied so it can be used in NumPy with older Python versions.

    """
    def __init__(self, record=False, module=None):
        self._record = record
        if module is None:
            self._module = sys.modules['warnings']
        else:
            self._module = module
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning
        if self._record:
            log = []
            def showwarning(*args, **kwargs):
                log.append(WarningMessage(*args, **kwargs))
            self._module.showwarning = showwarning
            return log
        else:
            return None

    def __exit__(self):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module.showwarning = self._showwarning

def deprecated(conditional=True):
    """
    Filter deprecation warnings while running the test suite.

    This decorator can be used to filter DeprecationWarning's, to avoid
    printing them during the test suite run, while checking that the test
    actually raises a DeprecationWarning.

    Parameters
    ----------
    conditional : bool or callable, optional
        Flag to determine whether to mark test as deprecated or not. If the
        condition is a callable, it is used at runtime to dynamically make the
        decision. Default is True.

    Returns
    -------
    decorator : function
        The `deprecated` decorator itself.

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    def deprecate_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose
        from noseclasses import KnownFailureTest

        def _deprecated_imp(*args, **kwargs):
            # Poor man's replacement for the with statement
            ctx = WarningManager(record=True)
            l = ctx.__enter__()
            warnings.simplefilter('always')
            try:
                f(*args, **kwargs)
                if not len(l) > 0:
                    raise AssertionError("No warning raised when calling %s"
                            % f.__name__)
                if not l[0].category is DeprecationWarning:
                    raise AssertionError("First warning for %s is not a " \
                            "DeprecationWarning( is %s)" % (f.__name__, l[0]))
            finally:
                ctx.__exit__()

        if callable(conditional):
            cond = conditional()
        else:
            cond = conditional
        if cond:
            return nose.tools.make_decorator(f)(_deprecated_imp)
        else:
            return f
    return deprecate_decorator
