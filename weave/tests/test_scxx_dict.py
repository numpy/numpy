""" Test refcounting and behavior of SCXX.
"""
import unittest
import time
import os,sys

from scipy_test.testing import *
set_package_path()
from weave import inline_tools
restore_path()


class test_dict_construct(unittest.TestCase):
    #------------------------------------------------------------------------
    # Check that construction from basic types is allowed and have correct
    # reference counts
    #------------------------------------------------------------------------
    def check_empty(self,level=5):
        # strange int value used to try and make sure refcount is 2.
        code = """
               py::dict val;
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == {}
                              
                    
class test_dict_has_key(unittest.TestCase):
    def check_obj(self,level=5):
        class foo:
            pass
        key = foo()
        a = {}
        a[key] = 12345
        code = """
               return_val = a.has_key(key);               
               """
        res = inline_tools.inline(code,['a','key'])
        assert res
    def check_int(self,level=5):
        a = {}
        a[1234] = 12345
        code = """
               return_val = a.has_key(1234);               
               """
        res = inline_tools.inline(code,['a'])
        assert res
    def check_double(self,level=5):
        a = {}
        a[1234.] = 12345
        code = """
               return_val = a.has_key(1234.);               
               """
        res = inline_tools.inline(code,['a'])
        assert res
    def check_complex(self,level=5):
        a = {}
        a[1+1j] = 12345
        key = 1+1j
        code = """
               return_val = a.has_key(key);               
               """
        res = inline_tools.inline(code,['a','key'])
        assert res
        
    def check_string(self,level=5):
        a = {}
        a["b"] = 12345
        code = """
               return_val = a.has_key("b");               
               """
        res = inline_tools.inline(code,['a'])
        assert res
    def check_std_string(self,level=5):
        a = {}
        a["b"] = 12345
        key_name = "b"
        code = """
               return_val = a.has_key(key_name);               
               """
        res = inline_tools.inline(code,['a','key_name'])
        assert res        
    def check_string_fail(self,level=5):
        a = {}
        a["b"] = 12345
        code = """
               return_val = a.has_key("c");               
               """
        res = inline_tools.inline(code,['a'])
        assert not res

class test_dict_get_item_op(unittest.TestCase):

    def generic_get(self,code,args=['a']):
        a = {}
        a['b'] = 12345
                
        res = inline_tools.inline(code,args)
        assert res == a['b']

    def check_char(self,level=5):
        self.generic_get('return_val = a["b"];')

    def DOESNT_WORK_check_char_fail(self,level=5):
        # We can't through a KeyError for dicts on RHS of
        # = but not on LHS.  Not sure how to deal with this.
        try:
            self.generic_get('return_val = a["c"];')
        except KeyError:
            pass
            
    def check_string(self,level=5):
        self.generic_get('return_val = a[std::string("b")];')


    def check_obj(self,level=5):
        code = """
               py::object name = "b";
               return_val = a[name];
               """ 
        self.generic_get(code,['a'])

    def DOESNT_WORK_check_obj_fail(self,level=5):
        # We can't through a KeyError for dicts on RHS of
        # = but not on LHS.  Not sure how to deal with this.
        try:
            code = """
                   py::object name = "c";
                   return_val = a[name];
                   """ 
            self.generic_get(code,['a'])
        except KeyError:
            pass    
            
class test_dict_set_operator(unittest.TestCase):
    def generic_new(self,key,val):
        # test that value is set correctly and that reference counts
        # on dict, key, and val are being handled correctly.
        a = {}
        # call once to handle mysterious addition of one ref count
        # on first call to inline.
        inline_tools.inline("a[key] = val;",['a','key','val'])
        assert a[key] == val
        before = sys.getrefcount(a), sys.getrefcount(key), sys.getrefcount(val)
        inline_tools.inline("a[key] = val;",['a','key','val'])
        assert a[key] == val
        after = sys.getrefcount(a), sys.getrefcount(key), sys.getrefcount(val)
        assert before == after
    def generic_overwrite(self,key,val):
        a = {}
        overwritten = 1
        a[key] = overwritten # put an item in the dict to be overwritten
        # call once to handle mysterious addition of one ref count
        # on first call to inline.
        before_overwritten = sys.getrefcount(overwritten)
        inline_tools.inline("a[key] = val;",['a','key','val'])
        assert a[key] == val
        before = sys.getrefcount(a), sys.getrefcount(key), sys.getrefcount(val)
        inline_tools.inline("a[key] = val;",['a','key','val'])
        assert a[key] == val
        after = sys.getrefcount(a), sys.getrefcount(key), sys.getrefcount(val)
        after_overwritten = sys.getrefcount(overwritten)
        assert before == after
        assert before_overwritten == after_overwritten
        
    def check_new_int_int(self,level=5):
        key,val = 1234,12345
        self.generic_new(key,val)
    def check_new_double_int(self,level=5):
        key,val = 1234.,12345
        self.generic_new(key,val)
    def check_new_std_string_int(self,level=5):
        key,val = "hello",12345
        self.generic_new(key,val)
    def check_new_complex_int(self,level=5):
        key,val = 1+1j,12345
        self.generic_new(key,val)
    def check_new_obj_int(self,level=5):
        class foo:
           pass
        key,val = foo(),12345
        self.generic_new(key,val)

    def check_overwrite_int_int(self,level=5):
        key,val = 1234,12345
        self.generic_overwrite(key,val)
    def check_overwrite_double_int(self,level=5):
        key,val = 1234.,12345
        self.generic_overwrite(key,val)
    def check_overwrite_std_string_int(self,level=5):
        key,val = "hello",12345
        self.generic_overwrite(key,val)
    def check_overwrite_complex_int(self,level=5):
        key,val = 1+1j,12345
        self.generic_overwrite(key,val)
    def check_overwrite_obj_int(self,level=5):
        class foo:
           pass
        key,val = foo(),12345
        self.generic_overwrite(key,val)
                
class test_dict_del(unittest.TestCase):
    def generic(self,key):
        # test that value is set correctly and that reference counts
        # on dict, key, are being handled correctly. after deletion,
        # the keys refcount should be one less than before.
        a = {}
        a[key] = 1
        inline_tools.inline("a.del(key);",['a','key'])
        assert not a.has_key(key)
        a[key] = 1
        before = sys.getrefcount(a), sys.getrefcount(key)
        inline_tools.inline("a.del(key);",['a','key'])
        assert not a.has_key(key)
        after = sys.getrefcount(a), sys.getrefcount(key)
        assert before[0] == after[0]
        assert before[1] == after[1] + 1
    def check_int(self,level=5):
        key = 1234
        self.generic(key)
    def check_double(self,level=5):
        key = 1234.
        self.generic(key)
    def check_std_string(self,level=5):
        key = "hello"
        self.generic(key)
    def check_complex(self,level=5):
        key = 1+1j
        self.generic(key)
    def check_obj(self,level=5):
        class foo:
           pass
        key = foo()
        self.generic(key)

class test_dict_others(unittest.TestCase):
    def check_clear(self,level=5):
        a = {}
        a["hello"] = 1
        inline_tools.inline("a.clear();",['a'])
        assert not a
    def check_items(self,level=5):
        a = {}
        a["hello"] = 1
        items = inline_tools.inline("return_val = a.items();",['a'])
        assert items == a.items()
    def check_values(self,level=5):
        a = {}
        a["hello"] = 1
        values = inline_tools.inline("return_val = a.values();",['a'])
        assert values == a.values()
    def check_keys(self,level=5):
        a = {}
        a["hello"] = 1
        keys = inline_tools.inline("return_val = a.keys();",['a'])
        assert keys == a.keys()
    def check_update(self,level=5):
        a,b = {},{}
        a["hello"] = 1
        b["hello"] = 2
        inline_tools.inline("a.update(b);",['a','b'])
        assert a == b

if __name__ == "__main__":
    ScipyTest('weave.scxx').run()
