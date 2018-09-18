from __future__ import division, absolute_import, print_function

import sys
import os
import uuid
from importlib import import_module
import pytest

import numpy.f2py

from numpy.testing import assert_equal
from . import util

class TestQuotedCharacter(util.F2PyTest):
    code = """
      SUBROUTINE FOO(OUT1, OUT2, OUT3, OUT4, OUT5, OUT6)
      CHARACTER SINGLE, DOUBLE, SEMICOL, EXCLA, OPENPAR, CLOSEPAR
      PARAMETER (SINGLE="'", DOUBLE='"', SEMICOL=';', EXCLA="!",
     1           OPENPAR="(", CLOSEPAR=")")
      CHARACTER OUT1, OUT2, OUT3, OUT4, OUT5, OUT6
Cf2py intent(out) OUT1, OUT2, OUT3, OUT4, OUT5, OUT6
      OUT1 = SINGLE
      OUT2 = DOUBLE
      OUT3 = SEMICOL
      OUT4 = EXCLA
      OUT5 = OPENPAR
      OUT6 = CLOSEPAR
      RETURN
      END
    """

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_quoted_character(self):
        assert_equal(self.module.foo(), (b"'", b'"', b';', b'!', b'(', b')'))

@pytest.mark.xfail(sys.version_info[0] < 3 and os.name == 'nt',
                   reason="our Appveyor CI configuration does not"
                          " have Fortran compilers available for"
                          " Python 2.x")
@pytest.mark.parametrize("extra_args", [
    # extra_args can be a list as of gh-11937
    ['--noopt', '--debug'],
    # test for string as well, using the same
    # fcompiler options
    '--noopt --debug',
    # also test absence of extra_args
    '',
    ])
def test_f2py_init_compile(extra_args):
    # flush through the f2py __init__
    # compile() function code path
    # as a crude test for input handling
    # following migration from exec_command()
    # to subprocess.check_output() in gh-11937

    # the Fortran 77 syntax requires 6 spaces
    # before any commands, but more space may
    # be added; gfortran can also compile
    # with --ffree-form to remove the indentation
    # requirement; here, the Fortran source is
    # formatted to roughly match an example from
    # the F2PY User Guide
    fsource =  '''
             integer function foo()
             foo = 10 + 5
             return
             end
             '''
    # use various helper functions in util.py to
    # enable robust build / compile and
    # reimport cycle in test suite
    d = util.get_module_dir()
    modulename = util.get_temp_module_name()

    cwd = os.getcwd()
    target = os.path.join(d, str(uuid.uuid4()) + '.f')
    # try running compile() with and without a
    # source_fn provided so that the code path 
    # where a temporary file for writing Fortran
    # source is created is also explored
    for source_fn in [target, None]:

        # mimic the path changing behavior used
        # by build_module() in util.py, but don't
        # actually use build_module() because it
        # has its own invocation of subprocess
        # that circumvents the f2py.compile code
        # block under test
        try:
            os.chdir(d)
            ret_val = numpy.f2py.compile(fsource,
                                         modulename=modulename,
                                         extra_args=extra_args,
                                         source_fn=source_fn)
        finally:
            os.chdir(cwd)

        # check for compile success return value
        assert_equal(ret_val, 0)

        # we are not currently able to import the
        # Python-Fortran interface module on Windows /
        # Appveyor, even though we do get successful
        # compilation on that platform with Python 3.x
        if os.name != 'nt':
            # check for sensible result of Fortran function;
            # that means we can import the module name in Python
            # and retrieve the result of the sum operation
            return_check = import_module(modulename)
            calc_result = return_check.foo()
            assert_equal(calc_result, 15)

def test_f2py_init_compile_failure():
    # verify an appropriate integer status
    # value returned by f2py.compile() when
    # invalid Fortran is provided
    ret_val = numpy.f2py.compile(b"invalid")
    assert_equal(ret_val, 1)

def test_f2py_init_compile_bad_cmd():
    # verify that usage of invalid command in
    # f2py.compile() returns status value of 127
    # for historic consistency with exec_command()
    # error handling

    # patch the sys Python exe path temporarily to
    # induce an OSError downstream
    # NOTE: how bad of an idea is this patching?
    try:
        temp = sys.executable
        sys.executable = 'does not exist'

        # the OSError should take precedence over the invalid
        # Fortran
        ret_val = numpy.f2py.compile(b"invalid")

        assert_equal(ret_val, 127)
    finally:
        sys.executable = temp
