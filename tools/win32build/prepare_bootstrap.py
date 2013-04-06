from __future__ import division, print_function

import os
import subprocess
import shutil
from os.path import join as pjoin, split as psplit, dirname
from zipfile import ZipFile
import re

def get_sdist_tarball():
    """Return the name of the installer built by wininst command."""
    # Yeah, the name logic is harcoded in distutils. We have to reproduce it
    # here
    name = "numpy-%s.zip" % get_numpy_version()
    return name

def build_sdist():
    cwd = os.getcwd()
    try:
        os.chdir('../..')
        cmd = ["python", "setup.py", "sdist", "--format=zip"]
        subprocess.call(cmd)
    except Exception as e:
        raise RuntimeError("Error while executing cmd (%s)" % e)
    finally:
        os.chdir(cwd)

def prepare_numpy_sources(bootstrap = 'bootstrap'):
    zid = ZipFile(pjoin('..', '..', 'dist', get_sdist_tarball()))
    root = 'numpy-%s' % get_numpy_version()

    # From the sdist-built tarball, extract all files into bootstrap directory,
    # but removing the numpy-VERSION head path
    for name in zid.namelist():
        cnt = zid.read(name)
        if name.startswith(root):
            # XXX: even on windows, the path sep in zip is '/' ?
            name = name.split('/', 1)[1]
        newname = pjoin(bootstrap, name)

        if not os.path.exists(dirname(newname)):
            os.makedirs(dirname(newname))
        fid = open(newname, 'wb')
        fid.write(cnt)

def prepare_nsis_script(bootstrap, pyver, numver):
    tpl = os.path.join('nsis_scripts', 'numpy-superinstaller.nsi.in')
    source = open(tpl, 'r')
    target = open(pjoin(bootstrap, 'numpy-superinstaller.nsi'), 'w')

    installer_name = 'numpy-%s-win32-superpack-python%s.exe' % (numver, pyver)
    cnt = "".join(source.readlines())
    cnt = cnt.replace('@NUMPY_INSTALLER_NAME@', installer_name)
    for arch in ['nosse', 'sse2', 'sse3']:
        cnt = cnt.replace('@%s_BINARY@' % arch.upper(),
                          get_binary_name(arch))

    target.write(cnt)

def prepare_bootstrap(pyver):
    bootstrap = "bootstrap-%s" % pyver
    if os.path.exists(bootstrap):
        shutil.rmtree(bootstrap)
    os.makedirs(bootstrap)

    build_sdist()
    prepare_numpy_sources(bootstrap)

    shutil.copy('build.py', bootstrap)
    prepare_nsis_script(bootstrap, pyver, get_numpy_version())

def get_binary_name(arch):
    return "numpy-%s-%s.exe" % (get_numpy_version(), arch)

def get_numpy_version(chdir = pjoin('..', '..')):
    cwd = os.getcwd()
    try:
        if not chdir:
            chdir = cwd
        os.chdir(chdir)
        version = subprocess.Popen(['python', '-c', 'import __builtin__; __builtin__.__NUMPY_SETUP__ = True; from numpy.version import version;print version'], stdout =  subprocess.PIPE).communicate()[0]
        version = version.strip()
        if 'dev' in version:
            out = subprocess.Popen(['svn', 'info'], stdout = subprocess.PIPE).communicate()[0]
            r = re.compile('Revision: ([0-9]+)')
            svnver = None
            for line in out.split('\n'):
                m = r.match(line)
                if m:
                    svnver = m.group(1)

            if not svnver:
                raise ValueError("Error while parsing svn version ?")
            version += svnver
    finally:
        os.chdir(cwd)
    return version

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p", "--pyver", dest="pyver",
                      help = "Python version (2.4, 2.5, etc...)")

    opts, args = parser.parse_args()
    pyver = opts.pyver

    if not pyver:
        pyver = "2.5"

    prepare_bootstrap(pyver)
