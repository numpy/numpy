from __future__ import division, print_function

import subprocess
import os

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p", "--pyver", dest="pyver",
                      help = "Python version (2.4, 2.5, etc...)")

    opts, args = parser.parse_args()
    pyver = opts.pyver

    if not pyver:
        pyver = "2.5"

    # Bootstrap
    subprocess.check_call(['python', 'prepare_bootstrap.py', '-p', pyver])

    # Build binaries
    subprocess.check_call(['python', 'build.py', '-p', pyver], 
                          cwd = 'bootstrap-%s' % pyver)

    # Build installer using nsis
    subprocess.check_call(['makensis', 'numpy-superinstaller.nsi'], 
                          cwd = 'bootstrap-%s' % pyver)
