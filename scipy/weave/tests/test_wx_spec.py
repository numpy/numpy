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

from scipy_test.testing import *
set_package_path()
from weave import ext_tools, wx_spec
restore_path()

import wxPython
import wxPython.wx

class test_wx_converter(unittest.TestCase):    
    def check_type_match_string(self,level=5):
        s = wx_spec.wx_converter()
        assert(not s.type_match('string') )
    def check_type_match_int(self,level=5):
        s = wx_spec.wx_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self,level=5):
        s = wx_spec.wx_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self,level=5):
        s = wx_spec.wx_converter()        
        assert(not s.type_match(5.+1j))
    def check_type_match_complex(self,level=5):
        s = wx_spec.wx_converter()        
        assert(not s.type_match(5.+1j))
    def check_type_match_wxframe(self,level=5):
        s = wx_spec.wx_converter()
        f=wxPython.wx.wxFrame(wxPython.wx.NULL,-1,'bob')        
        assert(s.type_match(f))
        
    def check_var_in(self,level=5):
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
            
    def no_check_var_local(self,level=5):
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
    def no_check_return(self,level=5):
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

if __name__ == "__main__":
    ScipyTest('weave.wx_spec').run()
