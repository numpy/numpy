from numpy.testing import *
import numpy as np
import StringIO

class TestSaveTxt(NumpyTestCase):
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


class TestLoadTxt(NumpyTestCase):
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

    def test_empty_file(self):
        c = StringIO.StringIO()
        assert_raises(IOError, np.loadtxt, c)

class Testfromregex(NumpyTestCase):
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
    NumpyTest().run()
