from numpy.testing import *
import numpy as np
import StringIO


class RoundtripTest:
    def test_array(self):
        a = np.array( [[1,2],[3,4]], float)
        self.do(a)

        a = np.array( [[1,2],[3,4]], int)
        self.do(a)

        a = np.array( [[1+5j,2+6j],[3+7j,4+8j]], dtype=np.csingle)
        self.do(a)

        a = np.array( [[1+5j,2+6j],[3+7j,4+8j]], dtype=np.cdouble)
        self.do(a)

    def test_1D(self):
        a = np.array([1,2,3,4], int)
        self.do(a)

    def test_record(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        self.do(a)

class TestSaveLoad(RoundtripTest, TestCase):
    def do(self, a):
        c = StringIO.StringIO()
        np.save(c, a)
        c.seek(0)
        a_reloaded = np.load(c)
        assert_equal(a, a_reloaded)


class TestSavezLoad(RoundtripTest, TestCase):
    def do(self, *arrays):
        c = StringIO.StringIO()
        np.savez(c, *arrays)
        c.seek(0)
        l = np.load(c)
        for n, a in enumerate(arrays):
            assert_equal(a, l['arr_%d' % n])

    def test_multiple_arrays(self):
        a = np.array( [[1,2],[3,4]], float)
        b = np.array( [[1+2j,2+7j],[3-6j,4+12j]], complex)
        self.do(a,b)

    def test_named_arrays(self):
        a = np.array( [[1,2],[3,4]], float)
        b = np.array( [[1+2j,2+7j],[3-6j,4+12j]], complex)
        c = StringIO.StringIO()
        np.savez(c, file_a=a, file_b=b)
        c.seek(0)
        l = np.load(c)
        assert_equal(a, l['file_a'])
        assert_equal(b, l['file_b'])


class TestSaveTxt(TestCase):
    def test_array(self):
        a =np.array( [[1,2],[3,4]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        assert(c.readlines(),
               ['1.000000000000000000e+00 2.000000000000000000e+00\n',
                '3.000000000000000000e+00 4.000000000000000000e+00\n'])

        a =np.array( [[1,2],[3,4]], int)
        c = StringIO.StringIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), ['1 2\n', '3 4\n'])

    def test_1D(self):
        a = np.array([1,2,3,4], int)
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
        a = np.array([[1,2],[3,4]], int)
        assert_array_equal(x, a)

        c.seek(0)
        x = np.loadtxt(c, dtype=float)
        a = np.array([[1,2],[3,4]], float)
        assert_array_equal(x, a)

    def test_1D(self):
        c = StringIO.StringIO()
        c.write('1\n2\n3\n4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([1,2,3,4], int)
        assert_array_equal(x, a)

        c = StringIO.StringIO()
        c.write('1,2,3,4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([1,2,3,4], int)
        assert_array_equal(x, a)

    def test_missing(self):
        c = StringIO.StringIO()
        c.write('1,2,3,,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            converters={3:lambda s: int(s or -999)})
        a = np.array([1,2,3,-999,5], int)
        assert_array_equal(x, a)

    def test_converters_with_usecols(self):
        c = StringIO.StringIO()
        c.write('1,2,3,,5\n6,7,8,9,10\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            converters={3:lambda s: int(s or -999)}, \
            usecols=(1, 3, ))
        a = np.array([[2,  -999],[7, 9]], int)
        assert_array_equal(x, a)

    def test_comments(self):
        c = StringIO.StringIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            comments='#')
        a = np.array([1,2,3,5], int)
        assert_array_equal(x, a)

    def test_skiprows(self):
        c = StringIO.StringIO()
        c.write('comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            skiprows=1)
        a = np.array([1,2,3,5], int)
        assert_array_equal(x, a)

        c = StringIO.StringIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', \
            skiprows=1)
        a = np.array([1,2,3,5], int)
        assert_array_equal(x, a)

    def test_usecols(self):
        a =np.array( [[1,2],[3,4]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1,))
        assert_array_equal(x, a[:,1])

        a =np.array( [[1,2,3],[3,4,5]], float)
        c = StringIO.StringIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1,2))
        assert_array_equal(x, a[:,1:])

        # Testing with arrays instead of tuples.
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=np.array([1,2]))
        assert_array_equal(x, a[:,1:])

        # Checking with dtypes defined converters.
        data = '''JOE 70.1 25.3
                BOB 60.5 27.9
                '''
        c = StringIO.StringIO(data)
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        arr = np.loadtxt(c, usecols=(0,2),dtype=zip(names,dtypes))
        assert_equal(arr['stid'],  ["JOE",  "BOB"])
        assert_equal(arr['temp'],  [25.3,  27.9])

    def test_fancy_dtype(self):
        c = StringIO.StringIO()
        c.write('1,2,3.0\n4,5,6.0\n')
        c.seek(0)
        dt = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        x = np.loadtxt(c, dtype=dt, delimiter=',')
        a = np.array([(1,(2,3.0)),(4,(5,6.0))], dt)
        assert_array_equal(x, a)

    def test_empty_file(self):
        c = StringIO.StringIO()
        assert_raises(IOError, np.loadtxt, c)

    def test_unused_converter(self):
        c = StringIO.StringIO()
        c.writelines(['1 21\n', '3 42\n'])
        c.seek(0)
        data = np.loadtxt(c, usecols=(1,), converters={0: lambda s: int(s, 16)})
        assert_array_equal(data, [21, 42])

        c.seek(0)
        data = np.loadtxt(c, usecols=(1,), converters={1: lambda s: int(s, 16)})
        assert_array_equal(data, [33, 66])

class Testfromregex(TestCase):
    def test_record(self):
        c = StringIO.StringIO()
        c.write('1.312 foo\n1.534 bar\n4.444 qux')
        c.seek(0)

        dt = [('num', np.float64), ('val', 'S3')]
        x = np.fromregex(c, r"([0-9.]+)\s+(...)", dt)
        a = np.array([(1.312, 'foo'), (1.534, 'bar'), (4.444, 'qux')], dtype=dt)
        assert_array_equal(x, a)

    def test_record_2(self):
        return # pass this test until #736 is resolved
        c = StringIO.StringIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)

        dt = [('num', np.int32), ('val', 'S3')]
        x = np.fromregex(c, r"(\d+)\s+(...)", dt)
        a = np.array([(1312, 'foo'), (1534, 'bar'), (4444, 'qux')], dtype=dt)
        assert_array_equal(x, a)

    def test_record_3(self):
        c = StringIO.StringIO()
        c.write('1312 foo\n1534 bar\n4444 qux')
        c.seek(0)

        dt = [('num', np.float64)]
        x = np.fromregex(c, r"(\d+)\s+...", dt)
        a = np.array([(1312,), (1534,), (4444,)], dtype=dt)
        assert_array_equal(x, a)


if __name__ == "__main__":
    run_module_suite()
