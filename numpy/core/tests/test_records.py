
from numpy.testing import *
set_package_path()
from os import path
import numpy.core;reload(numpy.core)
from numpy.core import *
restore_path()

class TestFromrecords(NumpyTestCase):
    def check_fromrecords(self):
        r = rec.fromrecords([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        assert_equal(r[0].item(),(456, 'dbe', 1.2))

    def check_method_array(self):
        r = rec.array('abcdefg'*100,formats='i2,a3,i4',shape=3,byteorder='big')
        assert_equal(r[1].item(),(25444, 'efg', 1633837924))

    def check_method_array2(self):
        r=rec.array([(1,11,'a'),(2,22,'b'),(3,33,'c'),(4,44,'d'),(5,55,'ex'),(6,66,'f'),(7,77,'g')],formats='u1,f4,a1')
        assert_equal(r[1].item(),(2, 22.0, 'b'))

    def check_recarray_slices(self):
        r=rec.array([(1,11,'a'),(2,22,'b'),(3,33,'c'),(4,44,'d'),(5,55,'ex'),(6,66,'f'),(7,77,'g')],formats='u1,f4,a1')
        assert_equal(r[1::2][1].item(),(4, 44.0, 'd'))

    def check_recarray_fromarrays(self):
        x1 = array([1,2,3,4])
        x2 = array(['a','dd','xyz','12'])
        x3 = array([1.1,2,3,4])
        r = rec.fromarrays([x1,x2,x3],names='a,b,c')
        assert_equal(r[1].item(),(2,'dd',2.0))
        x1[1] = 34
        assert_equal(r.a,array([1,2,3,4]))

    def check_recarray_fromfile(self):
        data_dir = path.join(path.dirname(__file__),'data')
        filename = path.join(data_dir,'recarray_from_file.fits')
        fd = open(filename)
        fd.seek(2880*2)
        r = rec.fromfile(fd, formats='f8,i4,a5', shape=3, byteorder='big')

    def check_recarray_from_obj(self):
        count = 10
        a = zeros(count, dtype='O')
        b = zeros(count, dtype='f8')
        c = zeros(count, dtype='f8')
        for i in range(len(a)):
            a[i] = range(1,10)

        mine = numpy.rec.fromarrays([a,b,c],
                                    names='date,data1,data2')
        for i in range(len(a)):
            assert(mine.date[i]==range(1,10))
            assert(mine.data1[i]==0.0)
            assert(mine.data2[i]==0.0)

    def check_recarray_from_names(self):
        ra = rec.array([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
                       names='c1, c2, c3, c4')
        pa = rec.fromrecords([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
                       names='c1, c2, c3, c4')
        assert ra.dtype == pa.dtype
        assert ra.shape == pa.shape
        for k in xrange(len(ra)):
            assert ra[k].item() == pa[k].item()

    def check_recarray_conflict_fields(self):
        ra = rec.array([(1,'abc',2.3),(2,'xyz',4.2),
                        (3,'wrs',1.3)],
                       names='field, shape, mean')
        ra.mean = [1.1,2.2,3.3]
        assert_array_almost_equal(ra['mean'], [1.1,2.2,3.3])
        assert type(ra.mean) is type(ra.var)
        ra.shape = (1,3)
        assert ra.shape == (1,3)
        ra.shape = ['A','B','C']
        assert_array_equal(ra['shape'], [['A','B','C']])
        ra.field = 5
        assert_array_equal(ra['field'], [[5,5,5]])
        assert callable(ra.field)

class TestRecord(NumpyTestCase):
    def setUp(self):
        self.data = rec.fromrecords([(1,2,3),(4,5,6)],
                            dtype=[("col1", "<i4"),
                                   ("col2", "<i4"),
                                   ("col3", "<i4")])

    def test_assignment1(self):
        a = self.data
        assert_equal(a.col1[0],1)
        a[0].col1 = 0
        assert_equal(a.col1[0],0)

    def test_assignment2(self):
        a = self.data
        assert_equal(a.col1[0],1)
        a.col1[0] = 0
        assert_equal(a.col1[0],0)

    def test_invalid_assignment(self):
        a = self.data
        def assign_invalid_column(x):
            x[0].col5 = 1
        self.failUnlessRaises(AttributeError,assign_invalid_column,a)

if __name__ == "__main__":
    NumpyTest().run()
