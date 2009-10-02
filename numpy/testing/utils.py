"""
Utility function to facilitate testing.
"""

import os
import sys
import re
import operator
import types
from nosetester import import_nose

__all__ = ['assert_equal', 'assert_almost_equal','assert_approx_equal',
           'assert_array_equal', 'assert_array_less', 'assert_string_equal',
           'assert_array_almost_equal', 'assert_raises', 'build_err_msg',
           'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal',
           'raises', 'rand', 'rundocs', 'runstring', 'verbose', 'measure',
           'assert_', 'assert_valid_refcount']

verbose = 0

def assert_(val, msg='') :
    """
    Assert that works in release mode.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    if not val :
        raise AssertionError(msg)

def gisnan(x):
    """like isnan, but always raise an error if type not supported instead of
    returning a TypeError object.

    Notes
    -----
    isnan and other ufunc sometimes return a NotImplementedType object instead
    of raising any exception. This function is a wrapper to make sure an
    exception is always raised.

    This should be removed once this problem is solved at the Ufunc level."""
    from numpy.core import isnan
    st = isnan(x)
    if isinstance(st, types.NotImplementedType):
        raise TypeError("isnan not supported for this type")
    return st

def gisfinite(x):
    """like isfinite, but always raise an error if type not supported instead of
    returning a TypeError object.

    Notes
    -----
    isfinite and other ufunc sometimes return a NotImplementedType object instead
    of raising any exception. This function is a wrapper to make sure an
    exception is always raised.

    This should be removed once this problem is solved at the Ufunc level."""
    from numpy.core import isfinite
    st = isfinite(x)
    if isinstance(st, types.NotImplementedType):
        raise TypeError("isfinite not supported for this type")
    return st

def gisinf(x):
    """like isinf, but always raise an error if type not supported instead of
    returning a TypeError object.

    Notes
    -----
    isinf and other ufunc sometimes return a NotImplementedType object instead
    of raising any exception. This function is a wrapper to make sure an
    exception is always raised.

    This should be removed once this problem is solved at the Ufunc level."""
    from numpy.core import isinf
    st = isinf(x)
    if isinstance(st, types.NotImplementedType):
        raise TypeError("isinf not supported for this type")
    return st

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
    """
    Raise an assertion if two objects are not equal.

    Given two objects (lists, tuples, dictionaries or numpy arrays), check
    that all elements of these objects are equal. An exception is raised at
    the first conflicting values.

    Parameters
    ----------
    actual : list, tuple, dict or ndarray
      The object to check.
    desired : list, tuple, dict or ndarray
      The expected object.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
      If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal.

    Examples
    --------
    >>> np.testing.assert_equal([4,5], [4,6])
    ...
    <type 'exceptions.AssertionError'>:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6

    """
    if isinstance(desired, dict):
        if not isinstance(actual, dict) :
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual),len(desired),err_msg,verbose)
        for k,i in desired.items():
            if k not in actual :
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], 'key=%r\n%s' % (k,err_msg), verbose)
        return
    if isinstance(desired, (list,tuple)) and isinstance(actual, (list,tuple)):
        assert_equal(len(actual),len(desired),err_msg,verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], 'item=%r\n%s' % (k,err_msg), verbose)
        return
    from numpy.core import ndarray, isscalar, signbit
    from numpy.lib import iscomplexobj, real, imag
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError("Items are not equal:\n" \
                    "ACTUAL: %s\n" \
                    "DESIRED: %s\n" % (str(actual), str(desired)))

    # Inf/nan/negative zero handling
    try:
        # isscalar test to check cases such as [np.nan] != np.nan
        if isscalar(desired) != isscalar(actual):
            raise AssertionError(msg)

        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            isdesnan = gisnan(desired)
            isactnan = gisnan(actual)
            if isdesnan or isactnan:
                if not (isdesnan and isactnan):
                    raise AssertionError(msg)
            else:
                if not desired == actual:
                    raise AssertionError(msg)
            return
        elif desired == 0 and actual == 0:
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)
    # If TypeError or ValueError raised while using isnan and co, just handle
    # as before
    except TypeError:
        pass
    except ValueError:
        pass
    if desired != actual :
        raise AssertionError(msg)

def print_assert_equal(test_string,actual,desired):
    """
    Test if two objects are equal, and print an error message if test fails.

    The test is performed with ``actual == desired``.

    Parameters
    ----------
    test_string : str
        The message supplied to AssertionError.
    actual : object
        The object to test for equality against `desired`.
    desired : object
        The expected result.

    Examples
    --------
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1])
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 2])
    Traceback (most recent call last):
    ...
    AssertionError: Test XYZ of func xyz failed
    ACTUAL:
    [0, 1]
    DESIRED:
    [0, 2]

    """
    import pprint
    try:
        assert(actual == desired)
    except AssertionError:
        import cStringIO
        msg = cStringIO.StringIO()
        msg.write(test_string)
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual,msg)
        msg.write('DESIRED: \n')
        pprint.pprint(desired,msg)
        raise AssertionError(msg.getvalue())

def assert_almost_equal(actual,desired,decimal=7,err_msg='',verbose=True):
    """
    Raise an assertion if two items are not equal up to desired precision.

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal)

    Given two objects (numbers or ndarrays), check that all elements of these
    objects are almost equal. An exception is raised at conflicting values.
    For ndarrays this delegates to assert_array_almost_equal

    Parameters
    ----------
    actual : number or ndarray
      The object to check.
    desired : number or ndarray
      The expected object.
    decimal : integer (decimal=7)
      desired precision
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
      If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_array_almost_equal: compares array_like objects
    assert_equal: tests objects for equality


    Examples
    --------
    >>> npt.assert_almost_equal(2.3333333333333, 2.33333334)
    >>> npt.assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    ...
    <type 'exceptions.AssertionError'>:
    Items are not equal:
     ACTUAL: 2.3333333333333002
     DESIRED: 2.3333333399999998

    >>> npt.assert_almost_equal(np.array([1.0,2.3333333333333]),
    \t\t\tnp.array([1.0,2.33333334]), decimal=9)
    ...
    <type 'exceptions.AssertionError'>:
    Arrays are not almost equal
    <BLANKLINE>
    (mismatch 50.0%)
     x: array([ 1.        ,  2.33333333])
     y: array([ 1.        ,  2.33333334])

    """
    from numpy.core import ndarray
    from numpy.lib import iscomplexobj, real, imag

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError("Items are not equal:\n" \
                    "ACTUAL: %s\n" \
                    "DESIRED: %s\n" % (str(actual), str(desired)))

    if isinstance(actual, (ndarray, tuple, list)) \
            or isinstance(desired, (ndarray, tuple, list)):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose,
                         header='Arrays are not almost equal')
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(msg)
            else:
                if not desired == actual:
                    raise AssertionError(msg)
            return
    except TypeError:
        pass
    if round(abs(desired - actual),decimal) != 0 :
        raise AssertionError(msg)


def assert_approx_equal(actual,desired,significant=7,err_msg='',verbose=True):
    """
    Raise an assertion if two items are not equal up to significant digits.

    Given two numbers, check that they are approximately equal.
    Approximately equal is defined as the number of significant digits
    that agree.

    Parameters
    ----------
    actual : number
      The object to check.
    desired : number
      The expected object.
    significant : integer (significant=7)
      desired precision
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
      If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_almost_equal: compares objects by decimals
    assert_array_almost_equal: compares array_like objects by decimals
    assert_equal: tests objects for equality


    Examples
    --------
    >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,
                                       significant=8)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,
                                       significant=8)
    ...
    <type 'exceptions.AssertionError'>:
    Items are not equal to 8 significant digits:
     ACTUAL: 1.234567e-021
     DESIRED: 1.2345672000000001e-021

    the evaluated condition that raises the exception is

    >>> abs(0.12345670e-20/1e-21 - 0.12345672e-20/1e-21) >= 10**-(8-1)
    True

    """
    import numpy as np
    actual, desired = map(float, (actual, desired))
    if desired==actual:
        return
    # Normalized the numbers to be in range (-10.0,10.0)
    # scale = float(pow(10,math.floor(math.log10(0.5*(abs(desired)+abs(actual))))))
    scale = 0.5*(np.abs(desired) + np.abs(actual))
    scale = np.power(10,np.floor(np.log10(scale)))
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
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(msg)
            else:
                if not desired == actual:
                    raise AssertionError(msg)
            return
    except TypeError:
        pass
    if np.abs(sc_desired - sc_actual) >= np.power(10.,-(significant-1)) :
        raise AssertionError(msg)

def assert_array_compare(comparison, x, y, err_msg='', verbose=True,
                         header=''):
    from numpy.core import array, isnan, any
    x = array(x, copy=False, subok=True)
    y = array(y, copy=False, subok=True)

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
            if not cond :
                raise AssertionError(msg)

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
                raise AssertionError(msg)
            # If only one item, it was a nan, so just return
            if x.size == y.size == 1:
                return
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
            if not cond :
                raise AssertionError(msg)
    except ValueError:
        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=('x', 'y'))
        raise ValueError(msg)

def assert_array_equal(x, y, err_msg='', verbose=True):
    """
    Raise an assertion if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal. An exception is raised at
    shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if
    both objects have NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : array_like
      The actual object to check.
    y : array_like
      The desired, expected object.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired objects are not equal.

    See Also
    --------
    assert_array_almost_equal: test objects for equality up to precision
    assert_equal: tests objects for equality


    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
    \t\t\t[np.exp(0),2.33333, np.nan])

    assert fails with numerical inprecision with floats

    >>> np.testing.assert_array_equal([1.0,np.pi,np.nan],
    \t\t\t[1, np.sqrt(np.pi)**2, np.nan])
    ...
    <type 'exceptions.ValueError'>:
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (mismatch 50.0%)
     x: array([ 1.        ,  3.14159265,         NaN])
     y: array([ 1.        ,  3.14159265,         NaN])

    use assert_array_almost_equal for these cases instead

    >>> np.testing.assert_array_almost_equal([1.0,np.pi,np.nan],
    \t\t\t[1, np.sqrt(np.pi)**2, np.nan], decimal=15)

    """
    assert_array_compare(operator.__eq__, x, y, err_msg=err_msg,
                         verbose=verbose, header='Arrays are not equal')

def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Raise an assertion if two objects are not equal up to desired precision.

    The test verifies identical shapes and verifies values with
    abs(desired-actual) < 0.5 * 10**(-decimal)

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are almost equal. An exception is raised at
    shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if
    both objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
      The actual object to check.
    y : array_like
      The desired, expected object.
    decimal : integer (decimal=6)
      desired precision
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_almost_equal: simple version for comparing numbers
    assert_array_equal: tests objects for equality


    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
                                             [1.0,2.333,np.nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    \t\t\t[1.0,2.33339,np.nan], decimal=5)
    ...
    <type 'exceptions.AssertionError'>:
    AssertionError:
    Arrays are not almost equal
    <BLANKLINE>
    (mismatch 50.0%)
     x: array([ 1.     ,  2.33333,      NaN])
     y: array([ 1.     ,  2.33339,      NaN])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    \t\t\t[1.0,2.33333, 5], decimal=5)
    <type 'exceptions.ValueError'>:
    ValueError:
    Arrays are not almost equal
     x: array([ 1.     ,  2.33333,      NaN])
     y: array([ 1.     ,  2.33333,  5.     ])

    """
    from numpy.core import around, number, float_
    from numpy.core.numerictypes import issubdtype
    from numpy.core.fromnumeric import any as npany
    def compare(x, y):
        try:
            if npany(gisinf(x)) or npany( gisinf(y)):
                xinfid = gisinf(x)
                yinfid = gisinf(y)
                if not xinfid == yinfid:
                    return False
                # if one item, x and y is +- inf
                if x.size == y.size == 1:
                    return x == y
                x = x[~xinfid]
                y = y[~yinfid]
        except TypeError:
            pass
        z = abs(x-y)
        if not issubdtype(z.dtype, number):
            z = z.astype(float_) # handle object arrays
        return around(z, decimal) <= 10.0**(-decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')

def assert_array_less(x, y, err_msg='', verbose=True):
    """
    Raise an assertion if two array_like objects are not ordered by less than.

    Given two array_like objects, check that the shape is equal and all
    elements of the first object are strictly smaller than those of the
    second object. An exception is raised at shape mismatch or incorrectly
    ordered values. Shape mismatch does not raise if an object has zero
    dimension. In contrast to the standard usage in numpy, NaNs are
    compared, no assertion is raised if both objects have NaNs in the same
    positions.



    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired objects are not equal.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision



    Examples
    --------
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
    ...
    <type 'exceptions.ValueError'>:
    Arrays are not less-ordered
    (mismatch 50.0%)
     x: array([  1.,   1.,  NaN])
     y: array([  1.,   2.,  NaN])

    >>> np.testing.assert_array_less([1.0, 4.0], 3)
    ...
    <type 'exceptions.ValueError'>:
    Arrays are not less-ordered
    (mismatch 50.0%)
     x: array([ 1.,  4.])
     y: array(3)

    >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
    ...
    <type 'exceptions.ValueError'>:
    Arrays are not less-ordered
    (shapes (3,), (1,) mismatch)
     x: array([ 1.,  2.,  3.])
     y: array([4])

    """
    assert_array_compare(operator.__lt__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not less-ordered')

def runstring(astr, dict):
    exec astr in dict

def assert_string_equal(actual, desired):
    """
    Test if two strings are equal.

    If the given strings are equal, `assert_string_equal` does nothing.
    If they are not equal, an AssertionError is raised, and the diff
    between the strings is shown.

    Parameters
    ----------
    actual : str
        The string to test for equality against the expected string.
    desired : str
        The expected string.

    Examples
    --------
    >>> np.testing.assert_string_equal('abc', 'abc')
    >>> np.testing.assert_string_equal('abc', 'abcd')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ...
    AssertionError: Differences in strings:
    - abc+ abcd?    +

    """
    # delay import of difflib to reduce startup time
    import difflib

    if not isinstance(actual, str) :
        raise AssertionError(`type(actual)`)
    if not isinstance(desired, str):
        raise AssertionError(`type(desired)`)
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
            if not d2.startswith('+ ') :
                raise AssertionError(`d2`)
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
        raise AssertionError(`d1`)
    if not diff_list:
        return
    msg = 'Differences in strings:\n%s' % (''.join(diff_list)).rstrip()
    if actual != desired :
        raise AssertionError(msg)


def rundocs(filename=None, raise_on_error=True):
    """
    Run doctests found in the given file.

    By default `rundocs` raises an AssertionError on failure.

    Parameters
    ----------
    filename : str
        The path to the file for which the doctests are run.
    raise_on_error : bool
        Whether to raise an AssertionError when a doctest fails. Default is
        True.

    Notes
    -----
    The doctests can be run by the user/developer by adding the ``doctests``
    argument to the ``test()`` call. For example, to run all tests (including
    doctests) for `numpy.lib`::

      >>> np.lib.test(doctests=True)

    """
    import doctest, imp
    if filename is None:
        f = sys._getframe(1)
        filename = f.f_globals['__file__']
    name = os.path.splitext(os.path.basename(filename))[0]
    path = [os.path.dirname(filename)]
    file, pathname, description = imp.find_module(name, path)
    try:
        m = imp.load_module(name, file, pathname, description)
    finally:
        file.close()

    tests = doctest.DocTestFinder().find(m)
    runner = doctest.DocTestRunner(verbose=False)

    msg = []
    if raise_on_error:
        out = lambda s: msg.append(s)
    else:
        out = None

    for test in tests:
        runner.run(test, out=out)

    if runner.failures > 0 and raise_on_error:
        raise AssertionError("Some doctests failed:\n%s" % "\n".join(msg))


def raises(*args,**kwargs):
    nose = import_nose()
    return nose.tools.raises(*args,**kwargs)

def assert_raises(*args,**kwargs):
    """
    assert_raises(exception_class, callable, *args, **kwargs)

    Fail unless an exception of class exception_class is thrown
    by callable when invoked with arguments args and keyword
    arguments kwargs. If a different type of exception is
    thrown, it will not be caught, and the test case will be
    deemed to have suffered an error, exactly as for an
    unexpected exception.

    """
    nose = import_nose()
    return nose.tools.assert_raises(*args,**kwargs)

def decorate_methods(cls, decorator, testmatch=None):
    """
    Apply a decorator to all methods in a class matching a regular expression.

    The given decorator is applied to all public methods of `cls` that are
    matched by the regular expression `testmatch`
    (``testmatch.search(methodname)``). Methods that are private, i.e. start
    with an underscore, are ignored.

    Parameters
    ----------
    cls : class
        Class whose methods to decorate.
    decorator : function
        Decorator to apply to methods
    testmatch : compiled regexp or str, optional
        The regular expression. Default value is None, in which case the
        nose default (``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``)
        is used.
        If `testmatch` is a string, it is compiled to a regular expression
        first.

    """
    if testmatch is None:
        testmatch = re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)
    else:
        testmatch = re.compile(testmatch)
    cls_attr = cls.__dict__

    # delayed import to reduce startup time
    from inspect import isfunction

    methods = filter(isfunction, cls_attr.values())
    for function in methods:
        try:
            if hasattr(function, 'compat_func_name'):
                funcname = function.compat_func_name
            else:
                funcname = function.__name__
        except AttributeError:
            # not a function
            continue
        if testmatch.search(funcname) and not funcname.startswith('_'):
            setattr(cls, funcname, decorator(function))
    return


def measure(code_str,times=1,label=None):
    """
    Return elapsed time for executing code in the namespace of the caller.

    The supplied code string is compiled with the Python builtin ``compile``.
    The precision of the timing is 10 milli-seconds. If the code will execute
    fast on this timescale, it can be executed many times to get reasonable
    timing accuracy.

    Parameters
    ----------
    code_str : str
        The code to be timed.
    times : int, optional
        The number of times the code is executed. Default is 1. The code is
        only compiled once.
    label : str, optional
        A label to identify `code_str` with. This is passed into ``compile``
        as the second argument (for run-time error messages).

    Returns
    -------
    elapsed : float
        Total elapsed time in seconds for executing `code_str` `times` times.

    Examples
    --------
    >>> etime = np.testing.measure('for i in range(1000): np.sqrt(i**2)',
    ...                            times=times)
    >>> print "Time for a single execution : ", etime / times, "s"
    Time for a single execution :  0.005 s

    """
    frame = sys._getframe(1)
    locs,globs = frame.f_locals,frame.f_globals

    code = compile(code_str,
                   'Test name: %s ' % label,
                   'exec')
    i = 0
    elapsed = jiffies()
    while i < times:
        i += 1
        exec code in globs,locs
    elapsed = jiffies() - elapsed
    return 0.01*elapsed

def assert_valid_refcount(op):
    import numpy as np
    a = np.arange(100 * 100)
    b = np.arange(100*100).reshape(100, 100)
    c = b

    i = 1

    rc = sys.getrefcount(i)
    for j in range(15):
        d = op(b,c)

    assert(sys.getrefcount(i) >= rc)

