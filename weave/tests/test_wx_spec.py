"""
check_var_in -- tests whether a variable is passed in correctly
                and also if the passed in variable can be reassigned
check_var_local -- tests wheter a variable is passed in , modified,
                   and returned correctly in the local_dict dictionary
                   argument
check_return -- test whether a variable is passed in, modified, and
                then returned as a function return value correctly                                 
"""
import unittest

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import ext_tools
import wx_spec
restore_path()

import wxPython
import wxPython.wx

class test_wx_specification(unittest.TestCase):    
    def check_type_match_string(self):
        s = wx_spec.wx_specification()
        assert(not s.type_match('string') )
    def check_type_match_int(self):
        s = wx_spec.wx_specification()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = wx_spec.wx_specification()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = wx_spec.wx_specification()        
        assert(not s.type_match(5.+1j))
    def check_type_match_complex(self):
        s = wx_spec.wx_specification()        
        assert(not s.type_match(5.+1j))
    def check_type_match_wxframe(self):
        s = wx_spec.wx_specification()
        f=wxPython.wx.wxFrame(wxPython.wx.NULL,-1,'bob')        
        assert(s.type_match(f))
        
    def check_var_in(self):
        mod = ext_tools.ext_module('wx_var_in',compiler='msvc')
        a = wxPython.wx.wxFrame(wxPython.wx.NULL,-1,'bob')        
        code = """
               a->SetTitle(wxString("jim"));
               """
        test = ext_tools.ext_function('test',code,['a'],locals(),globals())
        mod.add_function(test)
        mod.compile()
        import wx_var_in
        b=a
        wx_var_in.test(b)
        assert(b.GetTitle() == "jim")
        try:
            b = 1.
            wx_var_in.test(b)
        except TypeError:
            pass
        try:
            b = 1
            wx_var_in.test(b)
        except TypeError:
            pass
            
    def no_check_var_local(self):
        mod = ext_tools.ext_module('wx_var_local')
        a = 'string'
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = 'a=Py::String("hello");'
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import wx_var_local
        b='bub'
        q={}
        wx_var_local.test(b,q)
        assert(q['a'] == 'hello')
    def no_check_return(self):
        mod = ext_tools.ext_module('wx_return')
        a = 'string'
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a= Py::wx("hello");
               return_val = Py::new_reference_to(a);
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import wx_return
        b='bub'
        c = wx_return.test(b)
        assert( c == 'hello')

def test_suite():
    suites = []
    
    suites.append( unittest.makeSuite(test_wx_specification,'check_'))
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
