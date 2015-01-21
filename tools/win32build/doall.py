from __future__ import division, print_function

import subprocess
import os

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p", "--pyver", dest="pyver",
                      help = "Python version (2.4, 2.5, etc...)")
    parser.add_option("-m", "--build-msi", dest="msi",
                      help = "0 or 1. If 1, build a msi instead of an exe.")
    
    opts, args = parser.parse_args()
    pyver = opts.pyver
    msi = '1' if opts.msi else '0'

    if not pyver:
        pyver = "2.5"

    # Bootstrap
    subprocess.check_call(['python', 'prepare_bootstrap.py', '-p', pyver, '-m', msi])

    # Build binaries
    subprocess.check_call(['python', 'build.py', '-p', pyver, '-m', msi],
                          cwd = 'bootstrap-%s' % pyver)

    # Build installer using nsis
    subprocess.check_call(['makensis', 'numpy-superinstaller.nsi'],
                          cwd = 'bootstrap-%s' % pyver)
