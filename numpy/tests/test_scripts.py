""" Test scripts

Test that we can run executable scripts that have been installed with numpy.
"""
from __future__ import division, print_function, absolute_import

import sys
import os
import pytest
from os.path import join as pathjoin, isfile, dirname
from subprocess import Popen, PIPE

import numpy as np
from numpy.compat.py3k import basestring
from numpy.testing import assert_, assert_equal

is_inplace = isfile(pathjoin(dirname(np.__file__),  '..', 'setup.py'))


def run_command(cmd, check_code=True):
    """ Run command sequence `cmd` returning exit code, stdout, stderr

    Parameters
    ----------
    cmd : str or sequence
        string with command name or sequence of strings defining command
    check_code : {True, False}, optional
        If True, raise error for non-zero return code

    Returns
    -------
    returncode : int
        return code from execution of `cmd`
    stdout : bytes (python 3) or str (python 2)
        stdout from `cmd`
    stderr : bytes (python 3) or str (python 2)
        stderr from `cmd`

    Raises
    ------
    RuntimeError
        If `check_code` is True, and return code !=0
    """
    cmd = [cmd] if isinstance(cmd, basestring) else list(cmd)
    if os.name == 'nt':
        # Quote any arguments with spaces. The quotes delimit the arguments
        # on Windows, and the arguments might be file paths with spaces.
        # On Unix the list elements are each separate arguments.
        cmd = ['"{0}"'.format(c) if ' ' in c else c for c in cmd]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    if proc.poll() is None:
        proc.terminate()
    if check_code and proc.returncode != 0:
        raise RuntimeError('\n'.join(
            ['Command "{0}" failed with',
             'stdout', '------', '{1}', '',
             'stderr', '------', '{2}']).format(cmd, stdout, stderr))
    return proc.returncode, stdout, stderr


@pytest.mark.skipif(is_inplace, reason="Cannot test f2py command inplace")
@pytest.mark.xfail(reason="Test is unreliable")
def test_f2py():
    # test that we can run f2py script

    def try_f2py_commands(cmds):
        success = 0
        for f2py_cmd in cmds:
            try:
                code, stdout, stderr = run_command([f2py_cmd, '-v'])
                assert_equal(stdout.strip(), b'2')
                success += 1
            except Exception:
                pass
        return success

    if sys.platform == 'win32':
        # Only the single 'f2py' script is installed in windows.
        exe_dir = dirname(sys.executable)
        if exe_dir.endswith('Scripts'): # virtualenv
            f2py_cmds = [os.path.join(exe_dir, 'f2py')]
        else:
            f2py_cmds = [os.path.join(exe_dir, "Scripts", 'f2py')]
        success = try_f2py_commands(f2py_cmds)
        msg = "Warning: f2py not found in path"
        assert_(success == 1, msg)
    else:
        # Three scripts are installed in Unix-like systems:
        # 'f2py', 'f2py{major}', and 'f2py{major.minor}'. For example,
        # if installed with python3.7 the scripts would be named
        # 'f2py', 'f2py3', and 'f2py3.7'.
        version = sys.version_info
        major = str(version.major)
        minor = str(version.minor)
        f2py_cmds = ('f2py', 'f2py' + major, 'f2py' + major + '.' + minor)
        success = try_f2py_commands(f2py_cmds)
        msg = "Warning: not all of %s, %s, and %s are found in path" % f2py_cmds
        assert_(success == 3, msg)
