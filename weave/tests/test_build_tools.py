# still needed
# tests for MingW32Compiler
# don't know how to test gcc_exists() and msvc_exists()...

import unittest
import os, sys, tempfile

from scipy_test.testing import *
set_package_path()
from weave import build_tools
restore_path()

def is_writable(val):
    return os.access(val,os.W_OK)
    
class test_configure_build_dir(unittest.TestCase):
    def check_default(self):
        " default behavior is to return current directory "
        d = build_tools.configure_build_dir()
        if is_writable('.'):
            assert(d == os.path.abspath('.'))
        assert(is_writable(d))
    def check_curdir(self):
        " make sure it handles relative values. "
        d = build_tools.configure_build_dir('.')
        if is_writable('.'):
            assert(d == os.path.abspath('.'))
        assert(is_writable(d))    
    def check_pardir(self):
        " make sure it handles relative values "        
        d = build_tools.configure_build_dir('..')
        if is_writable('..'):
            assert(d == os.path.abspath('..'))
        assert(is_writable(d))                
    def check_bad_path(self):
        " bad path should return same as default (and warn) "
        d = build_tools.configure_build_dir('_bad_path_')
        d2 = build_tools.configure_build_dir()
        assert(d == d2)
        assert(is_writable(d))

class test_configure_temp_dir(test_configure_build_dir):
    def check_default(self):
        " default behavior returns tempdir"
        # this'll fail if the temp directory isn't writable.
        d = build_tools.configure_temp_dir()
        assert(d == tempfile.gettempdir())
        assert(is_writable(d))

class test_configure_sys_argv(unittest.TestCase):
    def check_simple(self):
        build_dir = 'build_dir'
        temp_dir = 'temp_dir'
        compiler = 'compiler'
        pre_argv = sys.argv[:]
        build_tools.configure_sys_argv(compiler,temp_dir,build_dir)
        argv = sys.argv[:]
        bd = argv[argv.index('--build-lib')+1]
        assert(bd == build_dir)
        td = argv[argv.index('--build-temp')+1]
        assert(td == temp_dir)
        argv.index('--compiler='+compiler)
        build_tools.restore_sys_argv()
        assert(pre_argv == sys.argv[:])

if __name__ == "__main__":
    ScipyTest('weave.build_tools').run()
