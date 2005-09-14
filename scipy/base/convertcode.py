
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
#  * Converts useage of .flat.xxx to .ravel().xxx
#  * Prints warning of other usage of flat.
#  * Prints warning for use of bool, int, float, copmlex, object, and unicode


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
    astr = astr.replace("'s'","'h'")
    astr = astr.replace("'w'","'H'")
    astr = astr.replace("'u'","'I'")
    return astr

def changeimports(fstr, name):
    importstr = 'import %s' % name
    importasstr = 'import %s as ' % name
    fromstr = 'from %s import ' % name
    fromallstr = 'from %s import *' % name
    fromall=0

    fstr = fstr.replace(importasstr, 'import scipy.base as ')
    fstr = fstr.replace(importstr, 'import scipy.base as %s' % name)
    if (fstr.find(fromallstr) >= 0):
        warnings.warn('Usage of %s found.' % fromallstr)
        fstr = fstr.replace(fromallstr, 'from scipy.base import *')
        fromall=1

    ind = 0
    Nlen = len(fromstr)
    Nlen2 = len("from scipy.base import ")
    while 1:
        found = fstr.find(fromstr,ind)
        if (found < 0):
            break
        ind = found + Nlen
        if fstr[ind] == '*':
            continue
        fstr = "%sfrom scipy.base import %s" % (fstr[:found], fstr[ind:])
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

def warnofnewtypes(filestr):
    if int_re.search(filestr) or \
       float_re.search(filestr) or \
       complex_re.search(filestr) or \
       unicode_re.search(filestr) or \
       bool_re.search(filestr):
        warnings.warn("Use of builtin bool, int, float, complex, or unicode\n" \
                      "found when import * used -- these will be handled by\n" \
                      "new array scalars under scipy.base")
        
    return
    

def process(filestr):
    filestr = replacetypechars(filestr)
    filestr, fromall1 = changeimports(filestr, 'Numeric')
    filestr, fromall2 = changeimports(filestr, 'numerix')
    filestr, fromall3 = changeimports(filestr, 'scipy_base')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    #filestr = convertflat(filestr)
    if fromall:
        warnofnewtypes(filestr)
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
       
def main(args):
    filename = args[1]
    filestr = getandcopy(filename)
    filestr = process(filestr)
    makenewfile(filename, filestr)

if __name__ == '__main__':
    main(sys.argv)
    
             

