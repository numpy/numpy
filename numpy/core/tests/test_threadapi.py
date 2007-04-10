from numpy.testing import *
set_package_path()
from numpy.core.umath import _setthreading, _getthreading
from numpy.core import setthreading, getthreading
restore_path()

class test_threading_state(NumpyTestCase):
    
    def check__getthreading(self):
        """ Check "low level" _getthreading() api for getting threading info.
        """
        use_threads, thread_count, element_threshold = _getthreading()
        
        # It is impossible to tell what the defaults should be.
        # They will vary from machine to machine.  Simply check that
        # they are sane.
        self.assertTrue(use_threads in [0,1])
        self.assertTrue(thread_count>0)
        self.assertTrue(element_threshold>0)

    def check__setthreading(self):
        """ Check "low level" _setthreading() api for setting threading info.
        """
        
        _setthreading(1, 4, 100000)
               
        use_threads, thread_count, element_threshold = _getthreading()
        
        self.assertEqual(use_threads, 1)
        self.assertEqual(thread_count, 4)
        self.assertEqual(element_threshold, 100000)

    def check_setthreading(self):
        
        default = getthreading()
        old = setthreading(use_threads=1, thread_count=4, 
                           element_threshold=100000)
        
        # ensure the returned value from setthreading was the previous state.
        self.assertEqual(default, old)
        
        use_threads, thread_count, element_threshold = getthreading()
        
        self.assertEqual(use_threads, 1)
        self.assertEqual(thread_count, 4)
        self.assertEqual(element_threshold, 100000)

    def check_setthreading_use_threads(self):
        
        default = getthreading()
        old = setthreading(use_threads=0)
        
        # ensure the returned value from setthreading was the previous state.
        self.assertEqual(default, old)
        
        use_threads, thread_count, element_threshold = getthreading()
        
        self.assertEqual(use_threads, 0)
        self.assertEqual(thread_count, default[1])
        self.assertEqual(element_threshold, default[2])

    def check_setthreading_thread_count(self):
        
        default = getthreading()
        old = setthreading(thread_count=20)
        
        # ensure the returned value from setthreading was the previous state.
        self.assertEqual(default, old)
        
        use_threads, thread_count, element_threshold = getthreading()
        
        self.assertEqual(use_threads, default[0])
        self.assertEqual(thread_count, 20)
        self.assertEqual(element_threshold, default[2])

    def check_setthreading_element_threshold(self):
        
        default = getthreading()
        old = setthreading(element_threshold=1e6)
        
        # ensure the returned value from setthreading was the previous state.
        self.assertEqual(default, old)
        
        use_threads, thread_count, element_threshold = getthreading()
        
        self.assertEqual(use_threads, default[0])
        self.assertEqual(thread_count, default[1])
        self.assertEqual(element_threshold, 1e6)

    def check_docstrings(self):
        self.assertTrue(len(_setthreading.__doc__) > 0)
        self.assertTrue(len(_getthreading.__doc__) > 0)        
        self.assertTrue(len(setthreading.__doc__) > 0)        
        self.assertTrue(len(getthreading.__doc__) > 0)        
        
        
if __name__ == "__main__":
    NumpyTest().run()
