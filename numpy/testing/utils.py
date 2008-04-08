"""
Utility function to facilitate testing.
"""

import os
import sys
import re
import difflib
import operator

__all__ = ['assert_equal', 'assert_almost_equal','assert_approx_equal',
           'assert_array_equal', 'assert_array_less', 'assert_string_equal',
           'assert_array_almost_equal', 'jiffies', 'memusage', 'rand',
           'runstring', 'raises']

def rand(*args):
    """Returns an array of random numbers with the given shape.

    This only uses the standard library, so it is useful for testing purposes.
    """
    import random
    from numpy.core import zeros, float64
    results = zeros(args, float64)
    f = results.flat
    for i in range(len(f)):
        f[i] = random.random()
    return results

if sys.platform[:5]=='linux':
    def jiffies(_proc_pid_stat = '/proc/%s/stat'%(os.getpid()),
                _load_time=[]):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. See man 5 proc. """
        import time
        if not _load_time:
            _load_time.append(time.time())
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[13])
        except:
            return int(100*(time.time()-_load_time[0]))

    def memusage(_proc_pid_stat = '/proc/%s/stat'%(os.getpid())):
        """ Return virtual memory size in bytes of the running python.
        """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[22])
        except:
            return
else:
    # os.getpid is not in all platforms available.
    # Using time is safe but inaccurate, especially when process
    # was suspended or sleeping.
    def jiffies(_load_time=[]):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. [Emulation with time.time]. """
        import time
        if not _load_time:
            _load_time.append(time.time())
        return int(100*(time.time()-_load_time[0]))
    def memusage():
        """ Return memory usage of running python. [Not implemented]"""
        raise NotImplementedError

if os.name=='nt' and sys.version[:3] > '2.3':
    # Code "stolen" from enthought/debug/memusage.py
    def GetPerformanceAttributes(object, counter, instance = None,
                                 inum=-1, format = None, machine=None):
        # NOTE: Many counters require 2 samples to give accurate results,
        # including "% Processor Time" (as by definition, at any instant, a
        # thread's CPU usage is either 0 or 100).  To read counters like this,
        # you should copy this function, but keep the counter open, and call
        # CollectQueryData() each time you need to know.
        # See http://msdn.microsoft.com/library/en-us/dnperfmo/html/perfmonpt2.asp
        # My older explanation for this was that the "AddCounter" process forced
        # the CPU to 100%, but the above makes more sense :)
        import win32pdh
        if format is None: format = win32pdh.PDH_FMT_LONG
        path = win32pdh.MakeCounterPath( (machine,object,instance, None, inum,counter) )
        hq = win32pdh.OpenQuery()
        try:
            hc = win32pdh.AddCounter(hq, path)
            try:
                win32pdh.CollectQueryData(hq)
                type, val = win32pdh.GetFormattedCounterValue(hc, format)
                return val
            finally:
                win32pdh.RemoveCounter(hc)
        finally:
            win32pdh.CloseQuery(hq)

    def memusage(processName="python", instance=0):
        # from win32pdhutil, part of the win32all package
        import win32pdh
        return GetPerformanceAttributes("Process", "Virtual Bytes",
                                        processName, instance,
                                        win32pdh.PDH_FMT_LONG, None)

def build_err_msg(arrays, err_msg, header='Items are not equal:',
                  verbose=True,
                  names=('ACTUAL', 'DESIRED')):
    msg = ['\n' + header]
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79-len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            try:
                r = repr(a)
            except:
                r = '[repr failed]'
            if r.count('\n') > 3:
                r = '\n'.join(r.splitlines()[:3])
                r += '...'
            msg.append(' %s: %s' % (names[i], r))
    return '\n'.join(msg)

def assert_equal(actual,desired,err_msg='',verbose=True):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
    """
    if isinstance(desired, dict):
        assert isinstance(actual, dict), repr(type(actual))
        assert_equal(len(actual),len(desired),err_msg,verbose)
        for k,i in desired.items():
            assert k in actual, repr(k)
            assert_equal(actual[k], desired[k], 'key=%r\n%s' % (k,err_msg), verbose)
        return
    if isinstance(desired, (list,tuple)) and isinstance(actual, (list,tuple)):
        assert_equal(len(actual),len(desired),err_msg,verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], 'item=%r\n%s' % (k,err_msg), verbose)
        return
    from numpy.core import ndarray
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)
    assert desired == actual, msg

def assert_almost_equal(actual,desired,decimal=7,err_msg='',verbose=True):
    """ Raise an assertion if two items are not equal.

    I think this should be part of unittest.py

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal)
    """
    from numpy.core import ndarray
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)
    assert round(abs(desired - actual),decimal) == 0, msg


def assert_approx_equal(actual,desired,significant=7,err_msg='',verbose=True):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
        Approximately equal is defined as the number of significant digits
        correct
    """
    import math
    actual, desired = map(float, (actual, desired))
    if desired==actual:
        return
    # Normalized the numbers to be in range (-10.0,10.0)
    scale = float(pow(10,math.floor(math.log10(0.5*(abs(desired)+abs(actual))))))
    try:
        sc_desired = desired/scale
    except ZeroDivisionError:
        sc_desired = 0.0
    try:
        sc_actual = actual/scale
    except ZeroDivisionError:
        sc_actual = 0.0
    msg = build_err_msg([actual, desired], err_msg,
                header='Items are not equal to %d significant digits:' %
                                 significant,
                verbose=verbose)
    assert math.fabs(sc_desired - sc_actual) < pow(10.,-(significant-1)), msg

def assert_array_compare(comparison, x, y, err_msg='', verbose=True,
                         header=''):
    from numpy.core import asarray, isnan, any
    from numpy import isreal, iscomplex
    x = asarray(x)
    y = asarray(y)

    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPfdgFDG'

    try:
        cond = (x.shape==() or y.shape==()) or x.shape == y.shape
        if not cond:
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(shapes %s, %s mismatch)' % (x.shape,
                                                                  y.shape),
                                verbose=verbose, header=header,
                                names=('x', 'y'))
            assert cond, msg

        if (isnumber(x) and isnumber(y)) and (any(isnan(x)) or any(isnan(y))):
            # Handling nan: we first check that x and y have the nan at the
            # same locations, and then we mask the nan and do the comparison as
            # usual.
            xnanid = isnan(x)
            ynanid = isnan(y)
            try:
                assert_array_equal(xnanid, ynanid)
            except AssertionError:
                msg = build_err_msg([x, y],
                                    err_msg
                                    + '\n(x and y nan location mismatch %s, ' \
                                    '%s mismatch)' % (xnanid, ynanid),
                                    verbose=verbose, header=header,
                                    names=('x', 'y'))
            val = comparison(x[~xnanid], y[~ynanid])
        else:
            val = comparison(x,y)
        if isinstance(val, bool):
            cond = val
            reduced = [0]
        else:
            reduced = val.ravel()
            cond = reduced.all()
            reduced = reduced.tolist()
        if not cond:
            match = 100-100.0*reduced.count(1)/len(reduced)
            msg = build_err_msg([x, y],
                                err_msg
                                + '\n(mismatch %s%%)' % (match,),
                                verbose=verbose, header=header,
                                names=('x', 'y'))
            assert cond, msg
    except ValueError:
        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=('x', 'y'))
        raise ValueError(msg)

def assert_array_equal(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__eq__, x, y, err_msg=err_msg,
                         verbose=verbose, header='Arrays are not equal')

def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    from numpy.core import around
    def compare(x, y):
        return around(abs(x-y),decimal) <= 10.0**(-decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')

def assert_array_less(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__lt__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not less-ordered')

def runstring(astr, dict):
    exec astr in dict

def assert_string_equal(actual, desired):
    assert isinstance(actual, str),`type(actual)`
    assert isinstance(desired, str),`type(desired)`
    if re.match(r'\A'+desired+r'\Z', actual, re.M): return
    diff = list(difflib.Differ().compare(actual.splitlines(1), desired.splitlines(1)))
    diff_list = []
    while diff:
        d1 = diff.pop(0)
        if d1.startswith('  '):
            continue
        if d1.startswith('- '):
            l = [d1]
            d2 = diff.pop(0)
            if d2.startswith('? '):
                l.append(d2)
                d2 = diff.pop(0)
            assert d2.startswith('+ '),`d2`
            l.append(d2)
            d3 = diff.pop(0)
            if d3.startswith('? '):
                l.append(d3)
            else:
                diff.insert(0, d3)
            if re.match(r'\A'+d2[2:]+r'\Z', d1[2:]):
                continue
            diff_list.extend(l)
            continue
        assert False, `d1`
    if not diff_list: return
    msg = 'Differences in strings:\n%s' % (''.join(diff_list)).rstrip()
    assert actual==desired, msg

# Ripped from nose.tools
def raises(*exceptions):
    """Test must raise one of expected exceptions to pass.

    Example use::

      @raises(TypeError, ValueError)
      def test_raises_type_error():
          raise TypeError("This test passes")

      @raises(Exception):
      def test_that_fails_by_passing():
          pass

    If you want to test many assertions about exceptions in a single test,
    you may want to use `assert_raises` instead.
    """
    valid = ' or '.join([e.__name__ for e in exceptions])
    def decorate(func):
        name = func.__name__
        def newfunc(*arg, **kw):
            try:
                func(*arg, **kw)
            except exceptions:
                pass
            except:
                raise
            else:
                message = "%s() did not raise %s" % (name, valid)
                raise AssertionError(message)
        newfunc = make_decorator(func)(newfunc)
        return newfunc
    return decorate
