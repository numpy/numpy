import sys
sys.path.insert(0,'..')
import os
if sys.prefix[:3]< '2.3' and __name__ == '__main__':
    __file__ = sys.argv[0]
d = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
if d == 'scipy_distutils':
    import scipy_distutils
    del sys.path[0]
    execfile('setup_scipy_distutils.py')
else:
    os.system('cd .. && ln -s %s scipy_distutils' % (d))
    import scipy_distutils
    del sys.path[0]
    execfile('setup_scipy_distutils.py')
    os.system('cd .. && rm -f scipy_distutils')
