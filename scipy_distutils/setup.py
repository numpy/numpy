import sys
sys.path.insert(0,'..')
import os
d = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
if d == 'scipy_distutils':
    execfile('setup_scipy_distutils.py')
else:
    os.system('cd .. && ln -s %s scipy_distutils' % (d))
    execfile('setup_scipy_distutils.py')
    os.system('cd .. && rm -f scipy_distutils')
