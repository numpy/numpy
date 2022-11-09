import os

from numpy.distutils.npy_pkg_config import parse_flags, read_config
from numpy.testing import assert_, temppath

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

class TestLibraryInfo:
    def test_simple(self):
        with temppath('foo.ini') as path:
            with open(path,  'w') as f:
                f.write(simple)
            pkg = os.path.splitext(path)[0]
            out = read_config(pkg)

        assertTrue(out.cflags() == simple_d['cflags'])
        assertTrue(out.libs() == simple_d['libflags'])
        assertTrue(out.name == simple_d['name'])
        assertTrue(out.version == simple_d['version'])

    def test_simple_variable(self):
        with temppath('foo.ini') as path:
            with open(path,  'w') as f:
                f.write(simple_variable)
            pkg = os.path.splitext(path)[0]
            out = read_config(pkg)

        assertTrue(out.cflags() == simple_variable_d['cflags'])
        assertTrue(out.libs() == simple_variable_d['libflags'])
        assertTrue(out.name == simple_variable_d['name'])
        assertTrue(out.version == simple_variable_d['version'])
        out.vars['prefix'] = '/Users/david'
        assertTrue(out.cflags() == '-I/Users/david/include')

class TestParseFlags:
    def test_simple_cflags(self):
        d = parse_flags("-I/usr/include")
        assertTrue(d['include_dirs'] == ['/usr/include'])

        d = parse_flags("-I/usr/include -DFOO")
        assertTrue(d['include_dirs'] == ['/usr/include'])
        assertTrue(d['macros'] == ['FOO'])

        d = parse_flags("-I /usr/include -DFOO")
        assertTrue(d['include_dirs'] == ['/usr/include'])
        assertTrue(d['macros'] == ['FOO'])

    def test_simple_lflags(self):
        d = parse_flags("-L/usr/lib -lfoo -L/usr/lib -lbar")
        assertTrue(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
        assertTrue(d['libraries'] == ['foo', 'bar'])

        d = parse_flags("-L /usr/lib -lfoo -L/usr/lib -lbar")
        assertTrue(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
        assertTrue(d['libraries'] == ['foo', 'bar'])
