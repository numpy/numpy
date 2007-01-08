"""
This module converts code written for numpy.oldnumeric to work
with numpy

FIXME:  Flesh this out.

Makes the following changes:
 * Converts typecharacters '1swu' to 'bhHI' respectively
   when used as typecodes
 * Changes import statements
 * Change typecode= to dtype=
 * Eliminates savespace=xxx keyword arguments
 *  Removes it when keyword is not given as well
 * replaces matrixmultiply with dot
 * converts functions that don't give axis= keyword that have changed
 * converts functions that don't give typecode= keyword that have changed
 * converts use of capitalized type-names
 * converts old function names in oldnumeric.linear_algebra,
   oldnumeric.random_array, and oldnumeric.fft

"""
#__all__ = ['convertfile', 'convertall', 'converttree']
__all__ = []

import warnings
warnings.warn("numpy.oldnumeric.alter_code2 is not working yet.")

import sys
import os
import re
import glob

# To convert typecharacters we need to
# Not very safe.  Disabled for now..
def replacetypechars(astr):
    astr = astr.replace("'s'","'h'")
    astr = astr.replace("'b'","'B'")
    astr = astr.replace("'1'","'b'")
    astr = astr.replace("'w'","'H'")
    astr = astr.replace("'u'","'I'")
    return astr

def changeimports(fstr, name, newname):
    importstr = 'import %s' % name
    importasstr = 'import %s as ' % name
    fromstr = 'from %s import ' % name
    fromall=0

    fstr = fstr.replace(importasstr, 'import %s as ' % newname)
    fstr = fstr.replace(importstr, 'import %s as %s' % (newname,name))

    ind = 0
    Nlen = len(fromstr)
    Nlen2 = len("from %s import " % newname)
    while 1:
        found = fstr.find(fromstr,ind)
        if (found < 0):
            break
        ind = found + Nlen
        if fstr[ind] == '*':
            continue
        fstr = "%sfrom %s import %s" % (fstr[:found], newname, fstr[ind:])
        ind += Nlen2 - Nlen
    return fstr, fromall

def replaceattr(astr):
    astr = astr.replace("matrixmultiply","dot")
    return astr

def replaceother(astr):
    astr = re.sub(r'typecode\s*=', 'dtype=', astr)
    astr = astr.replace('ArrayType', 'ndarray')
    astr = astr.replace('NewAxis', 'newaxis')
    return astr

import datetime
def fromstr(filestr):
    #filestr = replacetypechars(filestr)
    filestr, fromall1 = changeimports(filestr, 'numpy.oldnumeric', 'numpy')
    filestr, fromall1 = changeimports(filestr, 'numpy.core.multiarray', 'numpy')
    filestr, fromall1 = changeimports(filestr, 'numpy.core.umath', 'numpy')
    filestr, fromall3 = changeimports(filestr, 'LinearAlgebra',
                                      'numpy.linalg.old')
    filestr, fromall3 = changeimports(filestr, 'RNG', 'numpy.random.oldrng')
    filestr, fromall3 = changeimports(filestr, 'RNG.Statistics', 'numpy.random.oldrngstats')
    filestr, fromall3 = changeimports(filestr, 'RandomArray', 'numpy.random.oldrandomarray')
    filestr, fromall3 = changeimports(filestr, 'FFT', 'numpy.fft.old')
    filestr, fromall3 = changeimports(filestr, 'MA', 'numpy.core.ma')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    today = datetime.date.today().strftime('%b %d, %Y')
    name = os.path.split(sys.argv[0])[-1]
    filestr = '## Automatically adapted for '\
              'numpy %s by %s\n\n%s' % (today, name, filestr)
    return filestr

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
