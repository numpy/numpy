"""
Utility function to facilitate testing.
"""

import os
import sys

__all__ = ['assert_equal', 'assert_almost_equal','assert_approx_equal',
           'assert_array_equal', 'assert_array_less',
           'assert_array_almost_equal', 'jiffies', 'memusage', 'rand',
           'runstring']

def rand(*args):
    """Returns an array of random numbers with the given shape.

    This only uses the standard library, so it is useful for testing purposes.
    """
    import random
    from numpy.core import zeros, Float64
    results = zeros(args,Float64)
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

def build_err_msg(actual, desired, err_msg, header='Items are not equal:',
                  verbose=True):
    msg = ['\n' + header]
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79-len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        try:
            rd = repr(desired)
        except:
            rd = '[repr failed]'
        rd = rd[:100]
        msg.append(' DESIRED: ' + rd)
        try:
            ra = repr(actual)
        except:
            ra = '[repr failed]'
        rd = ra[:100]
        msg.append(' ACTUAL: ' + ra)
    return '\n'.join(msg)

def assert_equal(actual,desired,err_msg='',verbose=True):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
    """
    if isinstance(desired, dict):
        assert isinstance(actual, dict),`type(actual)`
        assert_equal(len(actual),len(desired),err_msg,verbose)
        for k,i in desired.items():
            assert actual.has_key(k),`k`
            assert_equal(actual[k], desired[k], 'key=%r\n%s' % (k,err_msg), verbose)
        return
    if isinstance(desired, list) and isinstance(actual, list):
        assert_equal(len(actual),len(desired),err_msg,verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], 'item=%r\n%s' % (k,err_msg), verbose)
        return
    from numpy.core import ArrayType
    if isinstance(actual, ArrayType) or isinstance(desired, ArrayType):
        return assert_array_equal(actual, desired, err_msg)
    msg = build_err_msg(actual, desired, err_msg, verbose=verbose)
    assert desired == actual, msg

def assert_almost_equal(actual,desired,decimal=7,err_msg='',verbose=True):
    """ Raise an assertion if two items are not equal.

    I think this should be part of unittest.py

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal)
    """
    from numpy.core import ArrayType
    if isinstance(actual, ArrayType) or isinstance(desired, ArrayType):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    msg = build_err_msg(actual, desired, err_msg, verbose=verbose)
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
    msg = build_err_msg(actual, desired, err_msg,
                header='Items are not equal to %d significant digits:' %
                                 significant,
                verbose=verbose)
    assert math.fabs(sc_desired - sc_actual) < pow(10.,-1*significant), msg


def assert_array_equal(x,y,err_msg=''):
    from numpy.core import asarray, alltrue, equal, shape, ravel, array2string
    x,y = asarray(x), asarray(y)
    msg = '\nArrays are not equal'
    try:
        assert 0 in [len(shape(x)),len(shape(y))] \
               or (len(shape(x))==len(shape(y)) and \
                   alltrue(equal(shape(x),shape(y)))),\
                   msg + ' (shapes %s, %s mismatch):\n\t' \
                   % (shape(x),shape(y)) + err_msg
        reduced = ravel(equal(x,y))
        cond = alltrue(reduced)
        if not cond:
            s1 = array2string(x,precision=16)
            s2 = array2string(y,precision=16)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        raise ValueError, msg


def assert_array_almost_equal(x,y,decimal=6,err_msg=''):
    from numpy.core import asarray, alltrue, equal, shape, ravel,\
         array2string, less_equal, around
    x = asarray(x)
    y = asarray(y)
    msg = '\nArrays are not almost equal'
    try:
        cond = alltrue(equal(shape(x),shape(y)))
        if not cond:
            msg = msg + ' (shapes mismatch):\n\t'\
                  'Shape of array 1: %s\n\tShape of array 2: %s' % (shape(x),shape(y))
        assert cond, msg + '\n\t' + err_msg
        reduced = ravel(equal(less_equal(around(abs(x-y),decimal),10.0**(-decimal)),1))
        cond = alltrue(reduced)
        if not cond:
            s1 = array2string(x,precision=decimal+1)
            s2 = array2string(y,precision=decimal+1)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        print sys.exc_value
        print shape(x),shape(y)
        print x, y
        raise ValueError, 'arrays are not almost equal'

def assert_array_less(x,y,err_msg=''):
    from numpy.core import asarray, alltrue, less, equal, shape, ravel, array2string
    x,y = asarray(x), asarray(y)
    msg = '\nArrays are not less-ordered'
    try:
        assert alltrue(equal(shape(x),shape(y))),\
               msg + ' (shapes mismatch):\n\t' + err_msg
        reduced = ravel(less(x,y))
        cond = alltrue(reduced)
        if not cond:
            s1 = array2string(x,precision=16)
            s2 = array2string(y,precision=16)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        print shape(x),shape(y)
        raise ValueError, 'arrays are not less-ordered'

def runstring(astr, dict):
    exec astr in dict
