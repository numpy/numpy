from __future__ import division, absolute_import, print_function
import os, sysconfig

from numpy.testing import tempdir, assert_raises
from numpy.distutils import log
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils import ccompiler
from numpy.distutils.command import build_ext, build_clib

class Extension(object):
    depends = []
    extra_compile_args = []
    extra_objects = []
    extra_link_args = []
    libraries = []
    library_dirs = []
    runtime_library_dirs = []
    define_macros = []
    undef_macros = []
    language = 'C99'
    include_dirs = [sysconfig.get_paths()['platinclude']]
    export_symbols = []

class DummyCompiler(object):
    extra_f77_compile_args = []

dist = NumpyDistribution()

testdefs = '''
#ifdef __GNUC__
#  define NPY_EXPORTED extern __attribute__((visibility("default")))
#else
#  define NPY_EXPORTED extern __declspec(dllexport)
#endif
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
'''


addercode = testdefs + '''
#include <numpy/ndarraytypes.h>

NPY_EXPORTED npy_intp
adder(npy_intp a, npy_intp b)
{
    return a + b;
}
'''

def test_build_ext():
    # baseline test - can we compile a C extension?
    b = build_ext.build_ext(dist)
    b.compiler = ccompiler.new_compiler()
    b._f90_compiler = DummyCompiler()
    b._cxx_compiler = DummyCompiler()
    ext = Extension()
    ext.name = 'adder'
    src = 'adder.c'
    with tempdir() as tmpdir:
        tmpfile = os.path.join(tmpdir, src)
        with open(tmpfile, 'w') as fid:
            fid.write(addercode)
        ext.sources = [tmpfile]
        b.build_lib = tmpdir
        b.build_extension(ext)
        # XXX try to import the extension with ctypes and call the function

def test_ext_include_dir_order():
    # Make sure the real numpy includes take precedence over ext.include_dirs
    b = build_ext.build_ext(dist)
    b.compiler = ccompiler.new_compiler()
    b._f90_compiler = DummyCompiler()
    b._cxx_compiler = DummyCompiler()
    ext = Extension()
    ext.name = 'adder'
    src = 'adder.c'
    with tempdir() as tmpdir:
        fake_include = os.path.join(tmpdir, 'dummy', 'numpy', 'ndarraytypes.h')
        os.makedirs(os.path.dirname(fake_include))
        with open(fake_include, 'w') as fid:
            fid.write('#error got the wrong header file')
        # Put our dummy ndarraytypes.h at the front of the include path
        ext.include_dirs.insert(0, os.path.join(tmpdir, 'dummy'))

        tmpfile = os.path.join(tmpdir, src)
        with open(tmpfile, 'w') as fid:
            fid.write(addercode)
        ext.sources = [tmpfile]
        b.build_lib = tmpdir
        b.build_extension(ext)

def test_build_clib():
    # baseline test - can we compile a C lib?
    b = build_clib.build_clib(dist)
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
    b = build_clib.build_clib(dist)
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
        ext['include_dirs'] = [os.path.join(tmpdir, 'dummy'),
                               sysconfig.get_paths()['platinclude']]
        b.build_clib = tmpdir
        b.build_a_library(ext, 'adder', [])
        # XXX this creates a lib archive. What can we do with it?
