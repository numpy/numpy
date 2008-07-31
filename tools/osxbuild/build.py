"""Python script to build the OSX universal binaries.

This is a simple script, most of the heavy lifting is done in bdist_mpkg.

To run this script:  'python build.py'

Requires a svn version of numpy is installed, svn is used to revert
file changes made to the docs for the end-user install.  Installer is
built using sudo so file permissions are correct when installed on
user system.  Script will prompt for sudo pwd.

"""

import os
import shutil
import subprocess
from getpass import getuser

SRC_DIR = '../../'

USER_README = 'docs/README.txt'
DEV_README = SRC_DIR + 'README.txt'

BUILD_DIR = 'build'
DIST_DIR = 'dist'

def remove_dirs():
    print 'Removing old build and distribution directories...'
    print """The distribution is built as root, so the files have the correct
    permissions when installed by the user.  Chown them to user for removal."""
    if os.path.exists(BUILD_DIR):
        cmd = 'sudo chown -R %s %s' % (getuser(), BUILD_DIR)
        shellcmd(cmd)
        shutil.rmtree(BUILD_DIR)
    if os.path.exists(DIST_DIR):
        cmd = 'sudo chown -R %s %s' % (getuser(), DIST_DIR)
        shellcmd(cmd)
        shutil.rmtree(DIST_DIR)
        
def build_dist():
    print 'Building distribution... (using sudo)'
    cmd = 'sudo python setupegg.py bdist_mpkg'
    shellcmd(cmd)

def build_dmg():
    print 'Building disk image...'
    # Since we removed the dist directory at the start of the script,
    # our pkg should be the only file there.
    pkg = os.listdir(DIST_DIR)[0]
    fn, ext = os.path.splitext(pkg)
    dmg = fn + '.dmg'
    srcfolder = os.path.join(DIST_DIR, pkg)
    dstfolder = os.path.join(DIST_DIR, dmg)
    # build disk image
    cmd = 'sudo hdiutil create -srcfolder %s %s' % (srcfolder, dstfolder)
    shellcmd(cmd)

def copy_readme():
    """Copy a user README with info regarding the website, instead of
    the developer README which tells one how to build the source.
    """
    print 'Copy user README.txt for installer.'
    shutil.copy(USER_README, DEV_README)

def revert_readme():
    """Revert the developer README."""
    print 'Reverting README.txt...'
    cmd = 'svn revert %s' % DEV_README
    shellcmd(cmd)
                
def shellcmd(cmd, verbose=True):
    """Call a shell command."""
    if verbose:
        print cmd
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError, err:
        msg = """
        Error while executing a shell command.
        %s
        """ % str(err)
        raise Exception(msg)
    
def build():
    # update end-user documentation
    copy_readme()
    shellcmd("svn stat %s"%DEV_README)

    # change to source directory
    cwd = os.getcwd()
    os.chdir(SRC_DIR)

    # build distribution
    remove_dirs()
    build_dist()
    build_dmg()

    # change back to original directory
    os.chdir(cwd)
    # restore developer documentation
    revert_readme()
    
if __name__ == '__main__':
    build()
