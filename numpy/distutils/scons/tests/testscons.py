#! /usr/bin/env python 
# Last Change: Fri Dec 07 08:00 PM 2007 J
import os
from os.path import join
from shutil import rmtree

ROOTPATH = '/usr/media/src/dsp/'
INSTALLPATH = join(ROOTPATH, 'sconstest')
PYTHONPATH = join(INSTALLPATH, 'lib', 'python2.5', 'site-packages')

NUMPYPATH = join(ROOTPATH, 'numpy', 'numpy.scons')
SCIPYPATH = join(ROOTPATH, 'scipy', 'scipy.scons')

NUMPYBUILDPATH = join(NUMPYPATH, 'build')
SCIPYBUILDPATH = join(SCIPYPATH, 'build')

PERFLIB_OPTS = "ATLAS=None MKL=None"
COMPILER = None
FCOMPILER = "gnu95"
CFLAGS = "-Wall"
FFLAGS = "-Wall"

def remove_paths(paths):
    for p in paths:
        rmtree(p, True)
        #print "removing %s" % p

def clean():
    remove_paths([INSTALLPATH, NUMPYBUILDPATH, SCIPYBUILDPATH])

def _build_cmd(setupfile):
    cmd = ["PYTHONPATH=%s" % PYTHONPATH]
    cmd.append(PERFLIB_OPTS)
    cmd.append("CFLAGS=%s" % CFLAGS)
    cmd.append("FFLAGS=%s" % FFLAGS)
    #cmd.append("python")
    cmd.append(setupfile)
    cmd.append("scons")
    if COMPILER:
        cmd.append("--compiler=%s" % COMPILER)
    cmd.append("--fcompiler=%s" % FCOMPILER)
    cmd.append("install --prefix=%s" % INSTALLPATH)
    cmdstr = ' '.join(cmd)
    print cmdstr
    return cmdstr

def build_numpy():        
    setupfile = join(NUMPYPATH, 'setupscons.py')
    st = os.system(_build_cmd(setupfile))
    if st:
        raise RuntimeError("ST is %d" % st)
    
def build_scipy():
    setupfile = join(SCIPYPATH, 'setupscons.py')
    st = os.system(_build_cmd(setupfile))
    if st:
        raise RuntimeError("ST is %d" % st)
    
if __name__ == '__main__':
    #clean()
    build_numpy() 
    build_scipy() 
