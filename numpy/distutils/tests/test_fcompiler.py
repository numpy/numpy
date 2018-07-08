from __future__ import division, absolute_import, print_function

from numpy.testing import assert_
import numpy.distutils.fcompiler
import os

customizable_flags = [
    ('f77', 'F77FLAGS'),
    ('f90', 'F90FLAGS'),
    ('free', 'FREEFLAGS'),
    ('arch', 'FARCH'),
    ('debug', 'FDEBUG'),
    ('flags', 'FFLAGS'),
    ('linker_so', 'LDFLAGS'),
]


def test_fcompiler_flags_override(monkeypatch):
    monkeypatch.setitem(os.environ, 'NPY_DISTUTILS_APPEND_FLAGS', '0')
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='none')
    flag_vars = fc.flag_vars.clone(lambda *args, **kwargs: None)

    for opt, envvar in customizable_flags:
        new_flag = '-dummy-{}-flag'.format(opt)
        prev_flags = getattr(flag_vars, opt)

        monkeypatch.setitem(os.environ, envvar, new_flag)
        new_flags = getattr(flag_vars, opt)
        assert_(new_flags == [new_flag])


def test_fcompiler_flags_append(monkeypatch):
    monkeypatch.setitem(os.environ, 'NPY_DISTUTILS_APPEND_FLAGS', '1')
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='none')
    flag_vars = fc.flag_vars.clone(lambda *args, **kwargs: None)

    for opt, envvar in customizable_flags:
        new_flag = '-dummy-{}-flag'.format(opt)
        prev_flags = getattr(flag_vars, opt)

        monkeypatch.setitem(os.environ, envvar, new_flag)
        new_flags = getattr(flag_vars, opt)
        if prev_flags is None:
            assert_(new_flags == [new_flag])
        else:
            assert_(new_flags == prev_flags + [new_flag])

