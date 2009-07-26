import os
from tempfile import mkstemp

from numpy.testing import *
from numpy.distutils.npy_pkg_config import read_config, parse_flags

simple = """\
[meta]
Name = foo
Description = foo lib
Version = 0.1

[default]
cflags = -I/usr/include
libs = -L/usr/lib
"""
simple_d = {'cflags': '-I/usr/include', 'libflags': '-L/usr/lib',
        'version': '0.1', 'name': 'foo'}

simple_variable = """\
[meta]
Name = foo
Description = foo lib
Version = 0.1

[variables]
prefix = /foo/bar
libdir = ${prefix}/lib
includedir = ${prefix}/include

[default]
cflags = -I${includedir}
libs = -L${libdir}
"""
simple_variable_d = {'cflags': '-I/foo/bar/include', 'libflags': '-L/foo/bar/lib',
        'version': '0.1', 'name': 'foo'}

class TestLibraryInfo(TestCase):
    def test_simple(self):
        fd, filename = mkstemp('foo.ini')
        try:
            try:
                os.write(fd, simple)
            finally:
                os.close(fd)

            out = read_config(filename)
            self.failUnless(out.cflags() == simple_d['cflags'])
            self.failUnless(out.libs() == simple_d['libflags'])
            self.failUnless(out.name == simple_d['name'])
            self.failUnless(out.version == simple_d['version'])
        finally:
            os.remove(filename)

    def test_simple_variable(self):
        fd, filename = mkstemp('foo.ini')
        try:
            try:
                os.write(fd, simple_variable)
            finally:
                os.close(fd)

            out = read_config(filename)
            self.failUnless(out.cflags() == simple_variable_d['cflags'])
            self.failUnless(out.libs() == simple_variable_d['libflags'])
            self.failUnless(out.name == simple_variable_d['name'])
            self.failUnless(out.version == simple_variable_d['version'])

            out.vars['prefix'] = '/Users/david'
            self.failUnless(out.cflags() == '-I/Users/david/include')
        finally:
            os.remove(filename)

class TestParseFlags(TestCase):
    def test_simple_cflags(self):
        d = parse_flags("-I/usr/include")
        self.failUnless(d['include_dirs'] == ['/usr/include'])

        d = parse_flags("-I/usr/include -DFOO")
        self.failUnless(d['include_dirs'] == ['/usr/include'])
        self.failUnless(d['macros'] == ['FOO'])

        d = parse_flags("-I /usr/include -DFOO")
        self.failUnless(d['include_dirs'] == ['/usr/include'])
        self.failUnless(d['macros'] == ['FOO'])

    def test_simple_lflags(self):
        d = parse_flags("-L/usr/lib -lfoo -L/usr/lib -lbar")
        self.failUnless(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
        self.failUnless(d['libs'] == ['foo', 'bar'])

        d = parse_flags("-L /usr/lib -lfoo -L/usr/lib -lbar")
        self.failUnless(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
        self.failUnless(d['libs'] == ['foo', 'bar'])
