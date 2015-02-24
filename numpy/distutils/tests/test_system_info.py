from __future__ import division, absolute_import, print_function

import os
from tempfile import mkstemp, mkdtemp

import distutils
from numpy.testing import *
from numpy.distutils.system_info import *

def get_class(name, notfound_action=1):
    """
    notfound_action:
      0 - do nothing
      1 - display warning message
      2 - raise error
    """
    cl = {'temp1': test_temp1,
          'temp2': test_temp2
          }.get(name.lower(), test_system_info)
    return cl()

def get_standard_file(fname):
    """
    Overrides the get_standard_file from system_info
    """
    tmpdir = mkdtemp()
    filename = tmpdir + '/' + fname 
    with open(filename,'w') as fd:
        fd.write(site_cfg.encode('ascii'))
    filenames = [filename]
    return filenames

simple_site = """
[ALL]
library_dirs = {dir1:s}:{dir2:s}
libraries = {lib1:s},{lib2:s}
extra_compile_args = -I/fake/directory
runtime_library_dirs = {dir1:s}

[temp1]
library_dirs = {dir1:s}
libraries = {lib1:s}
runtime_library_dirs = {dir1:s}

[temp2]
library_dirs = {dir2:s}
libraries = {lib2:s}
extra_link_args = -Wl,-rpath={lib2:s}
"""
site_cfg = simple_site

fakelib_c_text = """
/* This file is generated from numpy/distutils/testing/test_system_info.py */
#include<stdio.h>
void foo(void) {
   printf("Hello foo");
}
void bar(void) {
   printf("Hello bar");
}
"""

class test_system_info(system_info):
    def __init__(self,
                 default_lib_dirs=default_lib_dirs,
                 default_include_dirs=default_include_dirs,
                 verbosity=1,
                 ):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {}
        defaults['library_dirs'] = []
        defaults['include_dirs'] = []
        defaults['runtime_library_dirs'] = []
        defaults['src_dirs'] = []
        defaults['search_static_first'] = []
        defaults['extra_compile_args'] = []
        defaults['extra_link_args'] = []
        self.cp = ConfigParser(defaults)
        self.files = []
        self.files.extend(get_standard_file('site.cfg'))
        self.parse_config_files()
        if self.section is not None:
            try:
                self.search_static_first = self.cp.getboolean(self.section, 'search_static_first')
            except: pass
        assert isinstance(self.search_static_first, int)

    def _check_libs(self, lib_dirs, libs, opt_libs, exts):
        """Override _check_libs to return with all dirs """
        info = {'libraries' : libs , 'library_dirs' : lib_dirs }
        return info

class test_temp1(test_system_info):
    section = 'temp1'
class test_temp2(test_system_info):
    section = 'temp2'

class TestSystemInfoReading(TestCase):

    def setUp(self):
        """ Create the libraries """
        # Create 2 sources and 2 libraries
        self._dir1 = mkdtemp()
        self._src1 = os.path.join(self._dir1,'foo.c')
        self._lib1 = os.path.join(self._dir1,'libfoo.so')
        self._dir2 = mkdtemp()
        self._src2 = os.path.join(self._dir2,'bar.c')
        self._lib2 = os.path.join(self._dir2,'libbar.so')
        # Update local site.cfg
        global simple_site, site_cfg
        site_cfg = simple_site.format(**{
                'dir1' : self._dir1 ,
                'lib1' : self._lib1 ,
                'dir2' : self._dir2 ,
                'lib2' : self._lib2 
                })
        # Write the sources
        with open(self._src1,'w') as fd:
            fd.write(fakelib_c_text)
        with open(self._src2,'w') as fd:
            fd.write(fakelib_c_text)

    def tearDown(self):
        try: 
            shutil.rmtree(self._dir1)
            shutil.rmtree(self._dir2)
        except: 
            pass

    def test_all(self):
        """ Read in all information in the ALL block """
        tsi = get_class('default')
        a = [self._dir1,self._dir2]
        self.assertTrue(tsi.get_lib_dirs() == a,
                        (tsi.get_lib_dirs(),a))
        a = [self._lib1,self._lib2]
        self.assertTrue(tsi.get_libraries() == a,
                        (tsi.get_libraries(),a))
        a = [self._dir1]
        self.assertTrue(tsi.get_runtime_lib_dirs() == a,
                        (tsi.get_runtime_lib_dirs(),a))
        extra = tsi.calc_extra_info()
        a = ['-I/fake/directory']
        self.assertTrue(extra['extra_compile_args'] == a,
                        (extra['extra_compile_args'],a))

    def test_temp1(self):
        """ Read in all information in the temp1 block """
        tsi = get_class('temp1')
        a = [self._dir1]
        self.assertTrue(tsi.get_lib_dirs() == a,
                        (tsi.get_lib_dirs(),a))
        a = [self._lib1]
        self.assertTrue(tsi.get_libraries() == a,
                        (tsi.get_libraries(),a))
        a = [self._dir1]
        self.assertTrue(tsi.get_runtime_lib_dirs() == a,
                        (tsi.get_runtime_lib_dirs(),a))

    def test_temp2(self):
        """ Read in all information in the temp2 block """
        tsi = get_class('temp2')
        a = [self._dir2]
        self.assertTrue(tsi.get_lib_dirs() == a,
                        (tsi.get_lib_dirs(),a))
        a = [self._lib2]
        self.assertTrue(tsi.get_libraries() == a,
                        (tsi.get_libraries(),a))
        extra = tsi.calc_extra_info()
        a = ['-Wl,-rpath='+self._lib2]
        self.assertTrue(extra['extra_link_args'] == a,
                        (extra['extra_link_args'],a))

    def test_compile1(self):
        """ Compile source and link the first source """
        tsi = get_class('temp1')
        c = distutils.ccompiler.new_compiler()
        # Change directory to not screw up directories
        try:
            previousDir = os.getcwd()
        except OSError:
            return
        os.chdir(self._dir1)
        c.compile([os.path.basename(self._src1)], output_dir=self._dir1,
                  include_dirs=tsi.get_include_dirs())
        # Ensure that the object exists
        self.assertTrue(os.path.isfile(self._src1.replace('.c','.o')))
        os.chdir(previousDir)

    def test_compile2(self):
        """ Compile source and link the second source """
        tsi = get_class('temp2')
        c = distutils.ccompiler.new_compiler()
        # Change directory to not screw up directories
        try:
            previousDir = os.getcwd()
        except OSError:
            return
        os.chdir(self._dir2)
        c.compile([os.path.basename(self._src2)], output_dir=self._dir2,
                  include_dirs=tsi.get_include_dirs(),
                  extra_postargs=tsi.calc_extra_info()['extra_link_args'])
        # Ensure that the object exists
        self.assertTrue(os.path.isfile(self._src2.replace('.c','.o')))
        os.chdir(previousDir)

if __name__ == '__main__':
    run_module_suite()
