"""Python script to build windows binaries to be fed to the "superpack".

The script is pretty dumb: it assumes python executables are installed the
standard way, and the location for blas/lapack/atlas is harcoded.

TODO:
    - integrate the x86analysis script to check built binaries
    - make the config configurable with a file

"""
from __future__ import division, print_function

import sys
import subprocess
import os
import shutil
from os.path import join as pjoin, split as psplit, dirname

PYEXECS = {"2.5" : "C:\python25\python.exe",
        "2.4" : "C:\python24\python24.exe",
        "2.6" : "C:\python26\python26.exe"}

_SSE3_CFG = r"""[atlas]
library_dirs = C:\local\lib\yop\sse3"""
_SSE2_CFG = r"""[atlas]
library_dirs = C:\local\lib\yop\sse2"""
_NOSSE_CFG = r"""[DEFAULT]
library_dirs = C:\local\lib\yop\nosse"""

SITECFG = {"sse2" : _SSE2_CFG, "sse3" : _SSE3_CFG, "nosse" : _NOSSE_CFG}

def get_python_exec(ver):
    """Return the executable of python for the given version."""
    # XXX Check that the file actually exists
    try:
        return PYEXECS[ver]
    except KeyError:
        raise ValueError("Version %s not supported/recognized" % ver)

def get_clean():
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")

def write_site_cfg(arch):
    if os.path.exists("site.cfg"):
        os.remove("site.cfg")
    f = open("site.cfg", 'w')
    f.writelines(SITECFG[arch])
    f.close()

def build(arch, pyver):
    print("Building numpy binary for python %s, arch is %s" % (get_python_exec(pyver), arch))
    get_clean()
    write_site_cfg(arch)

    if BUILD_MSI:
        cmd = "%s setup.py build -c mingw32 bdist_msi" % get_python_exec(pyver)
    else:
        cmd = "%s setup.py build -c mingw32 bdist_wininst" % get_python_exec(pyver)
    build_log = "build-%s-%s.log" % (arch, pyver)
    f = open(build_log, 'w')

    try:
        try:
            subprocess.check_call(cmd, shell = True, stderr = subprocess.STDOUT, stdout = f)
        finally:
            f.close()
    except subprocess.CalledProcessError as e:
        msg = """
There was an error while executing the following command:

    %s

Error was : %s

Look at the build log (%s).""" % (cmd, str(e), build_log)
        raise Exception(msg)

    move_binary(arch, pyver)

def move_binary(arch, pyver):
    if not os.path.exists("binaries"):
        os.makedirs("binaries")

    shutil.move(os.path.join('dist', get_windist_exec(pyver)),
            os.path.join("binaries", get_binary_name(arch)))

def get_numpy_version():
    if sys.version_info[0] >= 3:
        import builtins
    else:
        import __builtin__ as builtins

    builtins.__NUMPY_SETUP__ = True
    from numpy.version import version
    return version

def get_binary_name(arch):
    if BUILD_MSI:
        ext = '.msi'
    else:
        ext = '.exe'
    return "numpy-%s-%s%s" % (get_numpy_version(), arch, ext)

def get_windist_exec(pyver):
    """Return the name of the installer built by wininst command."""
    # Yeah, the name logic is harcoded in distutils. We have to reproduce it
    # here
    if BUILD_MSI:
        ext = '.msi'
    else:
        ext = '.exe'
    name = "numpy-%s.win32-py%s%s" % (get_numpy_version(), pyver, ext)
    return name

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-a", "--arch", dest="arch",
                      help = "Architecture to build (sse2, sse3, nosse, etc...)")
    parser.add_option("-p", "--pyver", dest="pyver",
                      help = "Python version (2.4, 2.5, etc...)")
    parser.add_option("-m", "--build-msi", dest="msi",
                      help = "0 or 1. If 1, build a msi instead of an exe.")

    opts, args = parser.parse_args()
    arch = opts.arch
    pyver = opts.pyver
    msi = opts.msi

    if not pyver:
        pyver = "2.5"
    if not msi:
        BUILD_MSI = False
    else:
        BUILD_MSI = True

    if not arch:
        for arch in SITECFG.keys():
            build(arch, pyver)
    else:
        build(arch, pyver)
