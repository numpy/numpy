"""
This module converts code written for numpy.numarray to work
with numpy

FIXME:  finish this.

"""
#__all__ = ['convertfile', 'convertall', 'converttree']
__all__ = []

import warnings
warnings.warn("numpy.numarray.alter_code2 is not working yet.")
import sys

import os
import glob

def makenewfile(name, filestr):
    fid = file(name, 'w')
    fid.write(filestr)
    fid.close()

def getandcopy(name):
    fid = file(name)
    filestr = fid.read()
    fid.close()
    base, ext = os.path.splitext(name)
    makenewfile(base+'.orig', filestr)
    return filestr

def convertfile(filename):
    """Convert the filename given from using Numeric to using NumPy

    Copies the file to filename.orig and then over-writes the file
    with the updated code
    """
    filestr = getandcopy(filename)
    filestr = fromstr(filestr)
    makenewfile(filename, filestr)

def fromargs(args):
    filename = args[1]
    convertfile(filename)

def convertall(direc=os.path.curdir):
    """Convert all .py files to use NumPy (from Numeric) in the directory given

    For each file, a backup of <usesnumeric>.py is made as
    <usesnumeric>.py.orig.  A new file named <usesnumeric>.py
    is then written with the updated code.
    """
    files = glob.glob(os.path.join(direc,'*.py'))
    for afile in files:
        convertfile(afile)

def _func(arg, dirname, fnames):
    convertall(dirname)

def converttree(direc=os.path.curdir):
    """Convert all .py files in the tree given

    """
    os.path.walk(direc, _func, None)


if __name__ == '__main__':
    fromargs(sys.argv)
