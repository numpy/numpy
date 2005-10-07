
# This module converts code written for Numeric to run with scipy.base

# Makes the following changes:
#  * Converts typecharacters
#  * Changes import statements (warns of use of from Numeric import *)
#  * Changes import statements (using numerix) ...
#  * Makes search and replace changes to:
#    - .typecode()
#    - .iscontiguous()
#    - .byteswapped()
#    - .itemsize()
#  * Converts .flat to .ravel() except for .flat = xxx or .flat[xxx]
#  * Change typecode= to dtype=
#  * Eliminates savespace=xxx
#  * Replace xxx.spacesaver() with True
#  * Convert xx.savespace(?) to pass + ## xx.savespace(?)
#  * Convert a.shape = ? to a.reshape(?) 
#  * Prints warning for use of bool, int, float, copmlex, object, and unicode
#

__all__ = ['fromfile', 'fromstr']

import sys
import os
import re
import warnings

flatindex_re = re.compile('([.]flat(\s*?[[=]))')
int_re = re.compile('int\s*[(][^)]*[)]')
bool_re = re.compile('bool\s*[(][^)]*[)]')
float_re = re.compile('float\s*[(][^)]*[)]')
complex_re = re.compile('complex\s*[(][^)]*[)]')
unicode_re = re.compile('unicode\s*[(][^)]*[)]')

def replacetypechars(astr):
    astr = astr.replace("'s'","'h'")
    astr = astr.replace("'c'","'S1'")
    astr = astr.replace("'b'","'B'")
    astr = astr.replace("'1'","'b'")
    astr = astr.replace("'w'","'H'")
    astr = astr.replace("'u'","'I'")
    return astr

def changeimports(fstr, name, newname):
    importstr = 'import %s' % name
    importasstr = 'import %s as ' % name
    fromstr = 'from %s import ' % name
    fromallstr = 'from %s import *' % name
    fromall=0

    fstr = fstr.replace(importasstr, 'import %s as ' % newname)
    fstr = fstr.replace(importstr, 'import %s as %s' % (newname,name))
    if (fstr.find(fromallstr) >= 0):
        warnings.warn('Usage of %s found.' % fromallstr)
        fstr = fstr.replace(fromallstr, 'from %s import *' % newname)
        fromall=1

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
    astr = astr.replace(".typecode()",".dtypechar")
    astr = astr.replace(".iscontiguous()",".flags['CONTIGUOUS']")
    astr = astr.replace(".byteswapped()",".byteswap()")
    astr = astr.replace(".itemsize()",".itemsize")

    # preserve uses of flat that should be o.k.
    tmpstr = flatindex_re.sub("@@@@\\2",astr)
    # replace other uses of flat
    tmpstr = tmpstr.replace(".flat",".ravel()")
    # put back .flat where it was valid
    astr = tmpstr.replace("@@@@", ".flat")
    return astr

svspc = re.compile(r'(\S+\s*[(].+),\s*savespace\s*=.+\s*[)]')
svspc2 = re.compile(r'([^,(\s]+[.]spacesaver[(][)])')
svspc3 = re.compile(r'(\S+[.]savespace[(].*[)])')
shpe = re.compile(r'(\S+\s*)[.]shape\s*=[^=]\s*(.+)')
def replaceother(astr):
    astr = astr.replace("typecode=","dtype=")
    astr = astr.replace("UserArray","ndarray")
    astr = svspc.sub('\\1)',astr)
    astr = svspc2.sub('True',astr)
    astr = svspc3.sub('pass  ## \\1', astr)
    astr = shpe.sub('\\1=\\1.reshape(\\2)', astr)
    return astr

def warnofnewtypes(filestr):
    if int_re.search(filestr) or \
       float_re.search(filestr) or \
       complex_re.search(filestr) or \
       unicode_re.search(filestr) or \
       bool_re.search(filestr):
        warnings.warn("Use of builtin bool, int, float, complex, or unicode\n" \
                      "found when import * used -- these will be handled by\n" \
                      "new array scalars under scipy")
        
    return
    
import datetime
def fromstr(filestr):
    filestr = replacetypechars(filestr)
    filestr, fromall1 = changeimports(filestr, 'Numeric', 'scipy')
    filestr, fromall1 = changeimports(filestr, 'multiarray',
                                      'scipy.base.multiarray')
    filestr, fromall1 = changeimports(filestr, 'umath',
                                          'scipy.base.umath')
    filestr, fromall1 = changeimports(filestr, 'Precision', 'scipy.base')
    filestr, fromall2 = changeimports(filestr, 'numerix', 'scipy.base')
    filestr, fromall3 = changeimports(filestr, 'scipy_base', 'scipy.base')
    filestr, fromall3 = changeimports(filestr, 'MLab', 'scipy.basic.linalg')
    filestr, fromall3 = changeimports(filestr, 'LinearAlgebra', 'scipy.basic.linalg')
    filestr, fromall3 = changeimports(filestr, 'RNG', 'scipy.basic.random')
    filestr, fromall3 = changeimports(filestr, 'RandomArray', 'scipy.basic.random')
    filestr, fromall3 = changeimports(filestr, 'FFT', 'scipy.basdic.fft')
    filestr, fromall3 = changeimports(filestr, 'MA', 'scipy.base.ma')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    if fromall:
        warnofnewtypes(filestr)
    today = datetime.date.today().strftime('%b %d, %Y')
    name = os.path.split(sys.argv[0])[-1]
    filestr = '## Automatically adapted for '\
              'scipy %s by %s\n\n%s' % (today, name, filestr)
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
       
def fromfile(args):
    filename = args[1]
    filestr = getandcopy(filename)
    filestr = fromstr(filestr)
    makenewfile(filename, filestr)

if __name__ == '__main__':
    fromfile(sys.argv)
    
             

