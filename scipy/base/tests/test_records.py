
from scipy.testing import *
set_package_path()
import scipy.base;reload(scipy.base)
from scipy.base import *
from scipy.base import records as rec
restore_path()

class test_fromrecords(ScipyTestCase):
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
        
if __name__ == "__main__":
    ScipyTest().run()
