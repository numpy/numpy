from __future__ import division, absolute_import, print_function
import os, sysconfig, sys
import setuptools # monkeypatches distutils to find MSVC compilers

from numpy.testing import tempdir, assert_raises
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils import ccompiler, system_info
from numpy.distutils.core import numpy_cmdclass

class Extension(object):
    depends = []
    extra_compile_args = []
    extra_objects = []
    extra_link_args = []
    libraries = []
    runtime_library_dirs = []
    define_macros = []
    undef_macros = []
    language = 'C99'
    include_dirs = [sysconfig.get_paths()['platinclude']]
    library_dirs = system_info.default_lib_dirs
    export_symbols = ['adder']

class DummyCompiler(object):
    extra_f77_compile_args = []
    library_dirs = []
    libraries = []

dist = NumpyDistribution()
dist.finalize_options()
dist.cmdclass.update(numpy_cmdclass)

testdefs = '''
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
'''


addercode = testdefs + '''
#include <numpy/ndarraytypes.h>

npy_intp
adder(npy_intp a, npy_intp b)
{
    return a + b;
}

/* Fake the exported initialization functions since build_ext exports them */
npy_intp
initadder(void)
{
    return 0;
}

npy_intp
PyInit_adder(void)
{
    return 0;
}
'''

def test_build_ext():
    # baseline test - can we compile a C extension?
    b = dist.get_command_obj('build_ext')
    b.compiler = ccompiler.new_compiler()
    b._f90_compiler = DummyCompiler()
    b._cxx_compiler = DummyCompiler()
    b.swig_opts = None
    ext = Extension()
    ext.name = 'adder'
    src = 'adder.c'
    with tempdir() as tmpdir:
        tmpfile = os.path.join(tmpdir, src)
        with open(tmpfile, 'w') as fid:
            fid.write(addercode)
        ext.sources = [tmpfile]
        b.build_lib = tmpdir
        if b.library_dirs:
            ext.library_dirs = b.library_dirs
        b.finalize_options()
        b.build_extension(ext)
        # XXX try to import the extension with ctypes and call the function
        # Do not try to import it, the init routine will not work

def test_ext_include_dir_order():
    # Make sure the real numpy includes take precedence over ext.include_dirs
    b = dist.get_command_obj('build_ext')
    b.compiler = ccompiler.new_compiler()
    b._f90_compiler = DummyCompiler()
    b._cxx_compiler = DummyCompiler()
    b.swig_opts = None
    ext = Extension()
    ext.name = 'adder'
    src = 'adder.c'
    with tempdir() as tmpdir:
        fake_include = os.path.join(tmpdir, 'dummy', 'numpy', 'ndarraytypes.h')
        os.makedirs(os.path.dirname(fake_include))
        with open(fake_include, 'w') as fid:
            fid.write('#error got the wrong header file')
        # Put our dummy ndarraytypes.h at the front of the include path
        # If not overridden, this will error out since the dummy include file is invalid
        ext.include_dirs.insert(0, os.path.join(tmpdir, 'dummy'))

        tmpfile = os.path.join(tmpdir, src)
        with open(tmpfile, 'w') as fid:
            fid.write(addercode)
        ext.sources = [tmpfile]
        if b.library_dirs:
            ext.library_dirs = b.library_dirs
        b.finalize_options()
        b.build_lib = tmpdir
        b.build_extension(ext)
        # Do not try to import it, the init routine will not work

def test_build_clib():
    # baseline test - can we compile a C lib?
    b = dist.get_command_obj('build_clib')
    b.compiler = ccompiler.new_compiler()
    b._f_compiler = DummyCompiler()
    # b._cxx_compiler = DummyCompiler()
    src = 'adder.c'
    ext = {}
    with tempdir() as tmpdir:
        tmpfile = os.path.join(tmpdir, src)
        with open(tmpfile, 'w') as fid:
            fid.write(addercode)
        ext['sources'] = [tmpfile]
        ext['include_dirs'] = [sysconfig.get_paths()['platinclude']]
        b.build_clib = tmpdir
        b.build_a_library(ext, 'adder', [])
        # XXX this creates a lib archive. What can we do with it?

def test_clib_build_clib():
    # baseline test - can we compile a C lib?
    b = dist.get_command_obj('build_clib')
    b.compiler = ccompiler.new_compiler()
    b._f_compiler = DummyCompiler()
    # b._cxx_compiler = DummyCompiler()
    src = 'adder.c'
    ext = {}
    with tempdir() as tmpdir:
        tmpfile = os.path.join(tmpdir, src)
        with open(tmpfile, 'w') as fid:
            fid.write(addercode)
        ext['sources'] = [tmpfile]

        fake_include = os.path.join(tmpdir, 'dummy', 'numpy', 'ndarraytypes.h')
        os.makedirs(os.path.dirname(fake_include))
        with open(fake_include, 'w') as fid:
            fid.write('#error got the wrong header file')
        # Put our dummy ndarraytypes.h at the front of the include path
        # If not overridden, this will error out since the dummy include file is invalid
        ext['include_dirs'] = [os.path.join(tmpdir, 'dummy'),
                               sysconfig.get_paths()['platinclude']]
        b.build_clib = tmpdir
        b.build_a_library(ext, 'adder', [])
        # XXX this creates a lib archive. What can we do with it?
