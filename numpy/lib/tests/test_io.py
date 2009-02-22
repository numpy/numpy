
import numpy as np
import numpy.ma as ma
from numpy.ma.testutils import *

import StringIO

from tempfile import NamedTemporaryFile
import sys, time
from datetime import datetime


MAJVER, MINVER = sys.version_info[:2]

def strptime(s, fmt=None):
    """This function is available in the datetime module only
    from Python >= 2.5.

    """
    return datetime(*time.strptime(s, fmt)[:3])

class RoundtripTest(object):
    def roundtrip(self, save_func, *args, **kwargs):
        """
        save_func : callable
            Function used to save arrays to file.
        file_on_disk : bool
            If true, store the file on disk, instead of in a
            string buffer.
        save_kwds : dict
            Parameters passed to `save_func`.
        load_kwds : dict
            Parameters passed to `numpy.load`.
        args : tuple of arrays
            Arrays stored to file.

        """
        save_kwds = kwargs.get('save_kwds', {})
        load_kwds = kwargs.get('load_kwds', {})
        file_on_disk = kwargs.get('file_on_disk', False)

        if file_on_disk:
            # Do not delete the file on windows, because we can't
            # reopen an already opened file on that platform, so we
            # need to close the file and reopen it, implying no
            # automatic deletion.
            if sys.platform == 'win32' and MAJVER >= 2 and MINVER >= 6:
                target_file = NamedTemporaryFile(delete=False)
            else:
                target_file = NamedTemporaryFile()
            load_file = target_file.name
        else:
            target_file = StringIO.StringIO()
            load_file = target_file

        arr = args

        save_func(target_file, *arr, **save_kwds)
        target_file.flush()
        target_file.seek(0)

        if sys.platform == 'win32' and not isinstance(target_file, StringIO.StringIO):
            target_file.close()

        arr_reloaded = np.load(load_file, **load_kwds)

        self.arr = arr
        self.arr_reloaded = arr_reloaded

    def test_array(self):
        a = np.array([[1, 2], [3, 4]], float)
        self.roundtrip(a)

        a = np.array([[1, 2], [3, 4]], int)
        self.roundtrip(a)

        a = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        self.roundtrip(a)

        a = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        self.roundtrip(a)

    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        self.roundtrip(a)

    @np.testing.dec.knownfailureif(sys.platform=='win32', "Fail on Win32")
    def test_mmap(self):
        a = np.array([[1, 2.5], [4, 7.3]])
        self.roundtrip(a, file_on_disk=True, load_kwds={'mmap_mode': 'r'})

    def test_record(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        self.roundtrip(a)

class TestSaveLoad(RoundtripTest, TestCase):
    def roundtrip(self, *args, **kwargs):
        RoundtripTest.roundtrip(self, np.save, *args, **kwargs)
        assert_equal(self.arr[0], self.arr_reloaded)

class TestSavezLoad(RoundtripTest, TestCase):
    def roundtrip(self, *args, **kwargs):
        RoundtripTest.roundtrip(self, np.savez, *args, **kwargs)
        for n, arr in enumerate(self.arr):
            assert_equal(arr, self.arr_reloaded['arr_%d' % n])

    def test_multiple_arrays(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        self.roundtrip(a,b)

    def test_named_arrays(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        c = StringIO.StringIO()
        np.savez(c, file_a=a, file_b=b)
        c.seek(0)
        l = np.load(c)
        assert_equal(a, l['file_a'])
        assert_equal(b, l['file_b'])


class TestSaveTxt(TestCase):
    @np.testing.dec.knownfailureif(sys.platform=='win32', "Fail on Win32")
    def test_array(self):
        a =np.array([[1, 2], [3, 4]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        assert(c.readlines() ==
               ['1.000000000000000000e+00 2.000000000000000000e+00\n',
                '3.000000000000000000e+00 4.000000000000000000e+00\n'])

        a =np.array([[1, 2], [3, 4]], int)
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), ['1 2\n', '3 4\n'])

    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, ['1\n', '2\n', '3\n', '4\n'])

    def test_record(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), ['1 2\n', '3 4\n'])

    def test_delimiter(self):
        a = np.array([[1., 2.], [3., 4.]])
        c = StringIO.StringIO()
        np.savetxt(c, a, delimiter=',', fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), ['1,2\n', '3,4\n'])

    def test_format(self):
        a = np.array([(1, 2), (3, 4)])
        c = StringIO.StringIO()
        # Sequence of formats
        np.savetxt(c, a, fmt=['%02d', '%3.1f'])
        c.seek(0)
        assert_equal(c.readlines(), ['01 2.0\n', '03 4.0\n'])

        # A single multiformat string
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%02d : %3.1f')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, ['01 : 2.0\n', '03 : 4.0\n'])

        # Specify delimiter, should be overiden
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%02d : %3.1f', delimiter=',')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, ['01 : 2.0\n', '03 : 4.0\n'])


class TestLoadTxt(TestCase):
    def test_record(self):
        c = StringIO.StringIO()
        c.write('1 2\n3 4')
        c.seek(0)
        x = np.loadtxt(c, dtype=[('x', np.int32), ('y', np.int32)])
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_array_equal(x, a)

        d = StringIO.StringIO()
        d.write('M 64.0 75.0\nF 25.0 60.0')
        d.seek(0)
        mydescriptor = {'names': ('gender','age','weight'),
                        'formats': ('S1',
                                    'i4', 'f4')}
        b = np.array([('M', 64.0, 75.0),
                      ('F', 25.0, 60.0)], dtype=mydescriptor)
        y = np.loadtxt(d, dtype=mydescriptor)
        assert_array_equal(y, b)

    def test_array(self):
        c = StringIO.StringIO()
        c.write('1 2\n3 4')

        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([[1, 2], [3, 4]], int)
        assert_array_equal(x, a)

        c.seek(0)
        x = np.loadtxt(c, dtype=float)
        a = np.array([[1, 2], [3, 4]], float)
        assert_array_equal(x, a)

    def test_1D(self):
        c = StringIO.StringIO()
        c.write('1\n2\n3\n4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

        c = StringIO.StringIO()
        c.write('1,2,3,4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

    def test_missing(self):
        c = StringIO.StringIO()
        c.write('1,2,3,,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            converters={3:lambda s: int(s or -999)})
        a = np.array([1, 2, 3, -999, 5], int)
        assert_array_equal(x, a)

    def test_converters_with_usecols(self):
        c = StringIO.StringIO()
        c.write('1,2,3,,5\n6,7,8,9,10\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            converters={3:lambda s: int(s or -999)}, \
            usecols=(1, 3,))
        a = np.array([[2, -999], [7, 9]], int)
        assert_array_equal(x, a)

    def test_comments(self):
        c = StringIO.StringIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            comments='#')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_skiprows(self):
        c = StringIO.StringIO()
        c.write('comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

        c = StringIO.StringIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_usecols(self):
        a = np.array([[1, 2], [3, 4]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1,))
        assert_array_equal(x, a[:,1])

        a =np.array([[1, 2, 3], [3, 4, 5]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1, 2))
        assert_array_equal(x, a[:, 1:])

        # Testing with arrays instead of tuples.
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=np.array([1, 2]))
        assert_array_equal(x, a[:, 1:])

        # Checking with dtypes defined converters.
        data = '''JOE 70.1 25.3
                BOB 60.5 27.9
                '''
        c = StringIO.StringIO(data)
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        arr = np.loadtxt(c, usecols=(0, 2), dtype=zip(names, dtypes))
        assert_equal(arr['stid'], ["JOE",  "BOB"])
        assert_equal(arr['temp'], [25.3,  27.9])

    def test_fancy_dtype(self):
        c = StringIO.StringIO()
        c.write('1,2,3.0\n4,5,6.0\n')
        c.seek(0)
        dt = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        x = np.loadtxt(c, dtype=dt, delimiter=',')
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dt)
        assert_array_equal(x, a)

    def test_empty_file(self):
        c = StringIO.StringIO()
        assert_raises(IOError, np.loadtxt, c)

    def test_unused_converter(self):
        c = StringIO.StringIO()
        c.writelines(['1 21\n', '3 42\n'])
        c.seek(0)
        data = np.loadtxt(c, usecols=(1,),
                          converters={0: lambda s: int(s, 16)})
        assert_array_equal(data, [21, 42])

        c.seek(0)
        data = np.loadtxt(c, usecols=(1,),
                          converters={1: lambda s: int(s, 16)})
        assert_array_equal(data, [33, 66])

class Testfromregex(TestCase):
    def test_record(self):
        c = StringIO.StringIO()
        c.write('1.312 foo\n1.534 bar\n4.444 qux')
        c.seek(0)

        dt = [('num', np.float64), ('val', 'S3')]
        x = np.fromregex(c, r"([0-9.]+)\s+(...)", dt)
        a = np.array([(1.312, 'foo'), (1.534, 'bar'), (4.444, 'qux')],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_record_2(self):
        c = StringIO.StringIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)

        dt = [('num', np.int32), ('val', 'S3')]
        x = np.fromregex(c, r"(\d+)\s+(...)", dt)
        a = np.array([(1312, 'foo'), (1534, 'bar'), (4444, 'qux')],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_record_3(self):
        c = StringIO.StringIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)

        dt = [('num', np.float64)]
        x = np.fromregex(c, r"(\d+)\s+...", dt)
        a = np.array([(1312,), (1534,), (4444,)], dtype=dt)
        assert_array_equal(x, a)


#####--------------------------------------------------------------------------


class TestFromTxt(TestCase):
    #
    def test_record(self):
        "Test w/ explicit dtype"
        data = StringIO.StringIO('1 2\n3 4')
#        data.seek(0)
        test = np.ndfromtxt(data, dtype=[('x', np.int32), ('y', np.int32)])
        control = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_equal(test, control)
        #
        data = StringIO.StringIO('M 64.0 75.0\nF 25.0 60.0')
#        data.seek(0)
        descriptor = {'names': ('gender','age','weight'),
                      'formats': ('S1', 'i4', 'f4')}
        control = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)],
                           dtype=descriptor)
        test = np.ndfromtxt(data, dtype=descriptor)
        assert_equal(test, control)

    def test_array(self):
        "Test outputing a standard ndarray"
        data = StringIO.StringIO('1 2\n3 4')
        control = np.array([[1,2],[3,4]], dtype=int)
        test = np.ndfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        #
        data.seek(0)
        control = np.array([[1,2],[3,4]], dtype=float)
        test = np.loadtxt(data, dtype=float)
        assert_array_equal(test, control)

    def test_1D(self):
        "Test squeezing to 1D"
        control = np.array([1, 2, 3, 4], int)
        #
        data = StringIO.StringIO('1\n2\n3\n4\n')
        test = np.ndfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        #
        data = StringIO.StringIO('1,2,3,4\n')
        test = np.ndfromtxt(data, dtype=int, delimiter=',')
        assert_array_equal(test, control)

    def test_comments(self):
        "Test the stripping of comments"
        control = np.array([1, 2, 3, 5], int)
        # Comment on its own line
        data = StringIO.StringIO('# comment\n1,2,3,5\n')
        test = np.ndfromtxt(data, dtype=int, delimiter=',', comments='#')
        assert_equal(test, control)
        # Comment at the end of a line
        data = StringIO.StringIO('1,2,3,5# comment\n')
        test = np.ndfromtxt(data, dtype=int, delimiter=',', comments='#')
        assert_equal(test, control)

    def test_skiprows(self):
        "Test row skipping"
        control = np.array([1, 2, 3, 5], int)
        #
        data = StringIO.StringIO('comment\n1,2,3,5\n')
        test = np.ndfromtxt(data, dtype=int, delimiter=',', skiprows=1)
        assert_equal(test, control)
        #
        data = StringIO.StringIO('# comment\n1,2,3,5\n')
        test = np.loadtxt(data, dtype=int, delimiter=',', skiprows=1)
        assert_equal(test, control)

    def test_header(self):
        "Test retrieving a header"
        data = StringIO.StringIO('gender age weight\nM 64.0 75.0\nF 25.0 60.0')
        test = np.ndfromtxt(data, dtype=None, names=True)
        control = {'gender': np.array(['M', 'F']),
                   'age': np.array([64.0, 25.0]),
                   'weight': np.array([75.0, 60.0])}
        assert_equal(test['gender'], control['gender'])
        assert_equal(test['age'], control['age'])
        assert_equal(test['weight'], control['weight'])

    def test_auto_dtype(self):
        "Test the automatic definition of the output dtype"
        data = StringIO.StringIO('A 64 75.0 3+4j True\nBCD 25 60.0 5+6j False')
        test = np.ndfromtxt(data, dtype=None)
        control = [np.array(['A', 'BCD']),
                   np.array([64, 25]),
                   np.array([75.0, 60.0]),
                   np.array([3+4j, 5+6j]),
                   np.array([True, False]),]
        assert_equal(test.dtype.names, ['f0','f1','f2','f3','f4'])
        for (i, ctrl) in enumerate(control):
            assert_equal(test['f%i' % i], ctrl)


    def test_auto_dtype_uniform(self):
        "Tests whether the output dtype can be uniformized"
        data = StringIO.StringIO('1 2 3 4\n5 6 7 8\n')
        test = np.ndfromtxt(data, dtype=None)
        control = np.array([[1,2,3,4],[5,6,7,8]])
        assert_equal(test, control)


    def test_fancy_dtype(self):
        "Check that a nested dtype isn't MIA"
        data = StringIO.StringIO('1,2,3.0\n4,5,6.0\n')
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = np.ndfromtxt(data, dtype=fancydtype, delimiter=',')
        control = np.array([(1,(2,3.0)),(4,(5,6.0))], dtype=fancydtype)
        assert_equal(test, control)


    def test_names_overwrite(self):
        "Test overwriting the names of the dtype"
        descriptor = {'names': ('g','a','w'),
                      'formats': ('S1', 'i4', 'f4')}
        data = StringIO.StringIO('M 64.0 75.0\nF 25.0 60.0')
        names = ('gender','age','weight')
        test = np.ndfromtxt(data, dtype=descriptor, names=names)
        descriptor['names'] = names
        control = np.array([('M', 64.0, 75.0),
                            ('F', 25.0, 60.0)], dtype=descriptor)
        assert_equal(test, control)


    def test_commented_header(self):
        "Check that names can be retrieved even if the line is commented out."
        data = StringIO.StringIO("""
#gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        # The # is part of the first name and should be deleted automatically.
        test = np.genfromtxt(data, names=True, dtype=None)
        ctrl = np.array([('M', 21, 72.1), ('F', 35, 58.33), ('M', 33, 21.99)],
                  dtype=[('gender','|S1'), ('age', int), ('weight', float)])
        assert_equal(test, ctrl)
        # Ditto, but we should get rid of the first element
        data = StringIO.StringIO("""
# gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        test = np.genfromtxt(data, names=True, dtype=None)
        assert_equal(test, ctrl)


    def test_autonames_and_usecols(self):
        "Tests names and usecols"
        data = StringIO.StringIO('A B C D\n aaaa 121 45 9.1')
        test = np.ndfromtxt(data, usecols=('A', 'C', 'D'),
                            names=True, dtype=None)
        control = np.array(('aaaa', 45, 9.1),
                           dtype=[('A', '|S4'), ('C', int), ('D', float)])
        assert_equal(test, control)


    def test_converters_with_usecols(self):
        "Test the combination user-defined converters and usecol"
        data = StringIO.StringIO('1,2,3,,5\n6,7,8,9,10\n')
        test = np.ndfromtxt(data, dtype=int, delimiter=',',
                            converters={3:lambda s: int(s or -999)},
                            usecols=(1, 3, ))
        control = np.array([[2,  -999], [7, 9]], int)
        assert_equal(test, control)

    def test_converters_with_usecols_and_names(self):
        "Tests names and usecols"
        data = StringIO.StringIO('A B C D\n aaaa 121 45 9.1')
        test = np.ndfromtxt(data, usecols=('A', 'C', 'D'), names=True,
                            dtype=None, converters={'C':lambda s: 2 * int(s)})
        control = np.array(('aaaa', 90, 9.1),
            dtype=[('A', '|S4'), ('C', int), ('D', float)])
        assert_equal(test, control)

    def test_converters_cornercases(self):
        "Test the conversion to datetime."
        converter = {'date': lambda s: strptime(s, '%Y-%m-%d %H:%M:%SZ')}
        data = StringIO.StringIO('2009-02-03 12:00:00Z, 72214.0')
        test = np.ndfromtxt(data, delimiter=',', dtype=None,
                            names=['date','stid'], converters=converter)
        control = np.array((datetime(2009,02,03), 72214.),
                           dtype=[('date', np.object_), ('stid', float)])
        assert_equal(test, control)


    def test_unused_converter(self):
        "Test whether unused converters are forgotten"
        data = StringIO.StringIO("1 21\n  3 42\n")
        test = np.ndfromtxt(data, usecols=(1,),
                            converters={0: lambda s: int(s, 16)})
        assert_equal(test, [21, 42])
        #
        data.seek(0)
        test = np.ndfromtxt(data, usecols=(1,),
                            converters={1: lambda s: int(s, 16)})
        assert_equal(test, [33, 66])


    def test_dtype_with_converters(self):
        dstr = "2009; 23; 46"
        test = np.ndfromtxt(StringIO.StringIO(dstr,),
                            delimiter=";", dtype=float, converters={0:str})
        control = np.array([('2009', 23., 46)],
                           dtype=[('f0','|S4'), ('f1', float), ('f2', float)])
        assert_equal(test, control)
        test = np.ndfromtxt(StringIO.StringIO(dstr,),
                            delimiter=";", dtype=float, converters={0:float})
        control = np.array([2009., 23., 46],)
        assert_equal(test, control)


    def test_dtype_with_object(self):
        "Test using an explicit dtype with an object"
        from datetime import date
        import time
        data = """
        1; 2001-01-01
        2; 2002-01-31
        """
        ndtype = [('idx', int), ('code', np.object)]
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        test = np.genfromtxt(StringIO.StringIO(data), delimiter=";", dtype=ndtype,
                             converters=converters)
        control = np.array([(1, datetime(2001,1,1)), (2, datetime(2002,1,31))],
                           dtype=ndtype)
        assert_equal(test, control)
        #
        ndtype = [('nest', [('idx', int), ('code', np.object)])]
        try:
            test = np.genfromtxt(StringIO.StringIO(data), delimiter=";",
                                 dtype=ndtype, converters=converters)
        except NotImplementedError:
            pass
        else:
            errmsg = "Nested dtype involving objects should be supported."
            raise AssertionError(errmsg)


    def test_userconverters_with_explicit_dtype(self):
        "Test user_converters w/ explicit (standard) dtype"
        data = StringIO.StringIO('skip,skip,2001-01-01,1.0,skip')
        test = np.genfromtxt(data, delimiter=",", names=None, dtype=float,
                             usecols=(2, 3), converters={2: str})
        control = np.array([('2001-01-01', 1.)],
                           dtype=[('', '|S10'), ('', float)])
        assert_equal(test, control)


    def test_spacedelimiter(self):
        "Test space delimiter"
        data = StringIO.StringIO("1  2  3  4   5\n6  7  8  9  10")
        test = np.ndfromtxt(data)
        control = np.array([[ 1., 2., 3., 4., 5.],
                            [ 6., 7., 8., 9.,10.]])
        assert_equal(test, control)


    def test_missing(self):
        data = StringIO.StringIO('1,2,3,,5\n')
        test = np.ndfromtxt(data, dtype=int, delimiter=',', \
                            converters={3:lambda s: int(s or -999)})
        control = np.array([1, 2, 3, -999, 5], int)
        assert_equal(test, control)


    def test_usecols(self):
        "Test the selection of columns"
        # Select 1 column
        control = np.array( [[1, 2], [3, 4]], float)
        data = StringIO.StringIO()
        np.savetxt(data, control)
        data.seek(0)
        test = np.ndfromtxt(data, dtype=float, usecols=(1,))
        assert_equal(test, control[:, 1])
        #
        control = np.array( [[1, 2, 3], [3, 4, 5]], float)
        data = StringIO.StringIO()
        np.savetxt(data, control)
        data.seek(0)
        test = np.ndfromtxt(data, dtype=float, usecols=(1, 2))
        assert_equal(test, control[:, 1:])
        # Testing with arrays instead of tuples.
        data.seek(0)
        test = np.ndfromtxt(data, dtype=float, usecols=np.array([1, 2]))
        assert_equal(test, control[:, 1:])
        # Checking with dtypes defined converters.
        data = StringIO.StringIO("""JOE 70.1 25.3\nBOB 60.5 27.9""")
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        test = np.ndfromtxt(data, usecols=(0, 2), dtype=zip(names, dtypes))
        assert_equal(test['stid'],  ["JOE",  "BOB"])
        assert_equal(test['temp'],  [25.3,  27.9])


    def test_empty_file(self):
        "Test that an empty file raises the proper exception"
        data = StringIO.StringIO()
        assert_raises(IOError, np.ndfromtxt, data)


    def test_fancy_dtype_alt(self):
        "Check that a nested dtype isn't MIA"
        data = StringIO.StringIO('1,2,3.0\n4,5,6.0\n')
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = np.mafromtxt(data, dtype=fancydtype, delimiter=',')
        control = ma.array([(1,(2,3.0)),(4,(5,6.0))], dtype=fancydtype)
        assert_equal(test, control)


    def test_withmissing(self):
        data = StringIO.StringIO('A,B\n0,1\n2,N/A')
        test = np.mafromtxt(data, dtype=None, delimiter=',', missing='N/A',
                            names=True)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', np.int), ('B', np.int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        #
        data.seek(0)
        test = np.mafromtxt(data, delimiter=',', missing='N/A', names=True)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[[False, False], [False, True]],)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)


    def test_user_missing_values(self):
        datastr ="A, B, C\n0, 0., 0j\n1, N/A, 1j\n-9, 2.2, N/A\n3, -99, 3j" 
        data = StringIO.StringIO(datastr)
        basekwargs = dict(dtype=None, delimiter=',', names=True, missing='N/A')
        mdtype = [('A', int), ('B', float), ('C', complex)]
        #
        test = np.mafromtxt(data, **basekwargs)
        control = ma.array([(   0, 0.0,    0j), (1, -999, 1j),
                            (  -9, 2.2, -999j), (3,  -99, 3j)],
                            mask=[(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
                            dtype=mdtype)
        assert_equal(test, control)
        #
        data.seek(0)
        test = np.mafromtxt(data, 
                            missing_values={0:-9, 1:-99, 2:-999j}, **basekwargs)
        control = ma.array([(   0, 0.0,    0j), (1, -999, 1j),
                            (  -9, 2.2, -999j), (3,  -99, 3j)],
                            mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)],
                            dtype=mdtype)
        assert_equal(test, control)
        #
        data.seek(0)
        test = np.mafromtxt(data, 
                            missing_values={0:-9, 'B':-99, 'C':-999j},
                            **basekwargs)
        control = ma.array([(   0, 0.0,    0j), (1, -999, 1j),
                            (  -9, 2.2, -999j), (3,  -99, 3j)],
                            mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)],
                            dtype=mdtype)
        assert_equal(test, control)


    def test_withmissing_float(self):
        data = StringIO.StringIO('A,B\n0,1.5\n2,-999.00')
        test = np.mafromtxt(data, dtype=None, delimiter=',', missing='-999.0',
                            names=True,)
        control = ma.array([(0, 1.5), (2, -1.)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', np.int), ('B', np.float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)


    def test_with_masked_column_uniform(self):
        "Test masked column"
        data = StringIO.StringIO('1 2 3\n4 5 6\n')
        test = np.genfromtxt(data, missing='2,5', dtype=None, usemask=True)
        control = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0],[0, 1, 0]])
        assert_equal(test, control)

    def test_with_masked_column_various(self):
        "Test masked column"
        data = StringIO.StringIO('True 2 3\nFalse 5 6\n')
        test = np.genfromtxt(data, missing='2,5', dtype=None, usemask=True)
        control = ma.array([(1, 2, 3), (0, 5, 6)],
                           mask=[(0, 1, 0),(0, 1, 0)],
                           dtype=[('f0', bool), ('f1', bool), ('f2', int)])
        assert_equal(test, control)


    def test_recfromtxt(self):
        #
        data = StringIO.StringIO('A,B\n0,1\n2,3')
        test = np.recfromtxt(data, delimiter=',', missing='N/A', names=True)
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('A', np.int), ('B', np.int)])
        self.failUnless(isinstance(test, np.recarray))
        assert_equal(test, control)
        #
        data = StringIO.StringIO('A,B\n0,1\n2,N/A')
        test = np.recfromtxt(data, dtype=None, delimiter=',', missing='N/A',
                             names=True, usemask=True)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', np.int), ('B', np.int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.A, [0, 2])


    def test_recfromcsv(self):
        #
        data = StringIO.StringIO('A,B\n0,1\n2,3')
        test = np.recfromcsv(data, missing='N/A',
                             names=True, case_sensitive=True)
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('A', np.int), ('B', np.int)])
        self.failUnless(isinstance(test, np.recarray))
        assert_equal(test, control)
        #
        data = StringIO.StringIO('A,B\n0,1\n2,N/A')
        test = np.recfromcsv(data, dtype=None, missing='N/A',
                             names=True, case_sensitive=True, usemask=True)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', np.int), ('B', np.int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.A, [0, 2])
        #
        data = StringIO.StringIO('A,B\n0,1\n2,3')
        test = np.recfromcsv(data, missing='N/A',)
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('a', np.int), ('b', np.int)])
        self.failUnless(isinstance(test, np.recarray))
        assert_equal(test, control)




if __name__ == "__main__":
    run_module_suite()
