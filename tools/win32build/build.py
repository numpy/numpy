"""Python script to build windows binaries to be fed to the "superpack".

The script is pretty dumb: it assumes python executables are installed the
standard way, and the location for blas/lapack/atlas is harcoded."""

# TODO:
#  - integrate the x86analysis script to check built binaries
#  - make the config configurable with a file
import sys
import subprocess
import os
import shutil

PYEXECS = {"2.5" : "C:\python25\python.exe",
        "2.4" : "C:\python24\python2.4.exe"}

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
    print "Building numpy binary for python %s, arch is %s" % (get_python_exec(pyver), arch)
    get_clean()
    write_site_cfg(arch)

    cmd = "%s setup.py build -c mingw32 bdist_wininst" % get_python_exec(pyver)
    build_log = "build-%s-%s.log" % (arch, pyver)
    f = open(build_log, 'w')

    try:
        try:
            print "Executing command %s" % cmd
            subprocess.check_call(cmd, shell = True, stderr = subprocess.STDOUT, stdout = f)
        finally:
            f.close()
    except subprocess.CalledProcessError, e:
        msg = """
There was an error while executing the following command:

    %s

Error was : %s

Look at the build log (%s).""" % (cmd, str(e), build_log)
        raise Exception(msg)

def get_numpy_version():
    import __builtin__
    __builtin__.__NUMPY_SETUP__ = True
    from numpy.version import version
    return version

def get_windist_exec(pyver):
    """Return the name of the installer built by wininst command."""
    # Yeah, the name logic is harcoded in distutils. We have to reproduce it
    # here
    name = "numpy-%s.win32-%s.exe" % (get_numpy_version(), pyver)
    return name

USAGE = """build.py ARCH PYTHON_VERSION

Example: build.py sse2 2.4."""

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError(Usage)
        sys.exit(-1)

    #arch = sys.argv[1]
    pyver = sys.argv[2]
    for arch in SITECFG.keys():
        build(arch, pyver)
