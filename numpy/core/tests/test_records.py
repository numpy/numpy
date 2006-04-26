
from numpy.testing import *
set_package_path()
import os as _os
import numpy.core;reload(numpy.core)
from numpy.core import *
restore_path()

class test_fromrecords(NumpyTestCase):
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
        __path__ = _os.path.split(__file__)
        filename = _os.path.join(__path__[0], "testdata.fits")
        fd = open(filename)
        fd.seek(2880*2)
        r = rec.fromfile(fd, formats='f8,i4,a5', shape=3, byteorder='big')

if __name__ == "__main__":
    NumpyTest().run()
