import subprocess
import os

PYVER = "2.5"

# Bootstrap
subprocess.check_call(['python', 'prepare_bootstrap.py'])

# Build binaries
subprocess.check_call(['python', 'build.py', '-p', PYVER], cwd = 'bootstrap-%s' % PYVER)

# Build installer using nsis
subprocess.check_call(['makensis', 'numpy-superinstaller.nsi'], cwd = 'bootstrap-%s' % PYVER)
