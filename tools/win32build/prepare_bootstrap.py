import os
import subprocess
import shutil
from os.path import join as pjoin, split as psplit, dirname
from zipfile import ZipFile

from build import get_numpy_version, get_binary_name

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
    except Exception, e:
        raise RuntimeError("Error while executing cmd (%s)" % e)
    finally:
        os.chdir(cwd)

def prepare_numpy_sources(bootstrap = 'bootstrap'):
    zid = ZipFile(pjoin('../..', 'dist', get_sdist_tarball()))
    root = 'numpy-%s' % get_numpy_version()

    # From the sdist-built tarball, extract all files into bootstrap directory,
    # but removing the numpy-VERSION head path
    for name in zid.namelist():
        cnt = zid.read(name)
        if name.startswith(root):
            name = name.split(os.sep, 1)[1]
        newname = pjoin(bootstrap, name)

        if not os.path.exists(dirname(newname)):
            os.makedirs(dirname(newname))
        fid = open(newname, 'w')
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
                          get_binary_name(arch)))

    target.write(cnt)

def prepare_bootstrap(pyver = "2.5"):
    bootstrap = "bootstrap-%s" % pyver
    if os.path.exists(bootstrap):
        shutil.rmtree(bootstrap)
    os.makedirs(bootstrap)

    #build_sdist()
    #prepare_numpy_sources(bootstrap)

    #shutil.copy('build.py', bootstrap)
    prepare_nsis_script(bootstrap, pyver, get_numpy_version())

if __name__ == '__main__':
    prepare_bootstrap("2.5")
