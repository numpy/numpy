"""
This module converts code written for Numeric to run with numpy

Makes the following changes:
 * Changes import statements (warns of use of from Numeric import *)
 * Changes import statements (using numerix) ...
 * Makes search and replace changes to:
   - .typecode()
   - .iscontiguous()
   - .byteswapped()
   - .itemsize()
   - .toscalar()
 * Converts .flat to .ravel() except for .flat = xxx or .flat[xxx]
 * Replace xxx.spacesaver() with True
 * Convert xx.savespace(?) to pass + ## xx.savespace(?)

 * Converts uses of 'b' to 'B' in the typecode-position of
   functions:
   eye, tri (in position 4)
   ones, zeros, identity, empty, array, asarray, arange,
   fromstring, indices, array_constructor (in position 2)

   and methods:
   astype --- only argument
    -- converts uses of '1', 's', 'w', and 'u' to
    -- 'b', 'h', 'H', and 'I'

 * Converts uses of type(...) is <type>
   isinstance(..., <type>)
"""
__all__ = ['convertfile', 'convertall']

import sys
import os
import re
import glob


_func4 = ['eye', 'tri']
_meth1 = ['astype']
_func2 = ['ones', 'zeros', 'identity', 'fromstring', 'indices',
         'empty', 'array', 'asarray', 'arange', 'array_constructor']

_chars = {'1':'b','s':'h','w':'H','u':'I'}

func_re = {}
meth_re = {}

for name in _func2:
    _astr = r"""(%s\s*[(][^,]*?[,][^'"]*?['"])b(['"][^)]*?[)])"""%name
    func_re[name] = re.compile(_astr, re.DOTALL)

for name in _func4:
    _astr = r"""(%s\s*[(][^,]*?[,][^,]*?[,][^,]*?[,][^'"]*?['"])b(['"][^)]*?[)])"""%name
    func_re[name] = re.compile(_astr, re.DOTALL)    

for name in _meth1:
    _astr = r"""(.%s\s*[(][^'"]*?['"])b(['"][^)]*?[)])"""%name
    func_re[name] = re.compile(_astr, re.DOTALL)

for char in _chars.keys():
    _astr = r"""(.astype\s*[(][^'"]*?['"])%s(['"][^)]*?[)])"""%char
    meth_re[char] = re.compile(_astr, re.DOTALL)

def fixtypechars(fstr):
    for name in _func2 + _func4 + _meth1:
        fstr = func_re[name].sub('\\1B\\2',fstr)
    for char in _chars.keys():
        fstr = meth_re[char].sub('\\1%s\\2'%_chars[char], fstr)
    return fstr

flatindex_re = re.compile('([.]flat(\s*?[[=]))')

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

istest_re = {}
_types = ['float', 'int', 'complex', 'ArrayType', 'FloatType',
          'IntType', 'ComplexType']
for name in _types:
    _astr = r'type\s*[(]([^)]*)[)]\s+(?:is|==)\s+(.*?%s)'%name
    istest_re[name] = re.compile(_astr)
def fixistesting(astr):
    for name in _types:
        astr = istest_re[name].sub('isinstance(\\1, \\2)', astr)
    return astr

def replaceattr(astr):
    astr = astr.replace(".typecode()",".dtype.char")
    astr = astr.replace(".iscontiguous()",".flags.contiguous")
    astr = astr.replace(".byteswapped()",".byteswap()")
    astr = astr.replace(".toscalar()", ".item()")
    astr = astr.replace(".itemsize()",".itemsize")
    # preserve uses of flat that should be o.k.
    tmpstr = flatindex_re.sub(r"@@@@\2",astr)
    # replace other uses of flat
    tmpstr = tmpstr.replace(".flat",".ravel()")
    # put back .flat where it was valid
    astr = tmpstr.replace("@@@@", ".flat")
    return astr

svspc2 = re.compile(r'([^,(\s]+[.]spacesaver[(][)])')
svspc3 = re.compile(r'(\S+[.]savespace[(].*[)])')
#shpe = re.compile(r'(\S+\s*)[.]shape\s*=[^=]\s*(.+)')
def replaceother(astr):
    astr = svspc2.sub('True',astr)
    astr = svspc3.sub(r'pass  ## \1', astr)
    #astr = shpe.sub('\\1=\\1.reshape(\\2)', astr)
    return astr

import datetime
def fromstr(filestr):
    filestr = fixtypechars(filestr)
    filestr = fixistesting(filestr)
    filestr, fromall1 = changeimports(filestr, 'Numeric', 'numpy.oldnumeric')
    filestr, fromall1 = changeimports(filestr, 'multiarray','numpy.oldnumeric')
    filestr, fromall1 = changeimports(filestr, 'umath', 'numpy.oldnumeric')
    filestr, fromall1 = changeimports(filestr, 'Precision', 'numpy.oldnumeric.precision')
    filestr, fromall1 = changeimports(filestr, 'UserArray', 'numpy.oldnumeric.user_array')
    filestr, fromall1 = changeimports(filestr, 'ArrayPrinter', 'numpy.oldnumeric.array_printer')
    filestr, fromall2 = changeimports(filestr, 'numerix', 'numpy.oldnumeric')
    filestr, fromall3 = changeimports(filestr, 'scipy_base', 'numpy.oldnumeric')
    filestr, fromall3 = changeimports(filestr, 'Matrix', 'numpy.oldnumeric.matrix')    
    filestr, fromall3 = changeimports(filestr, 'MLab', 'numpy.oldnumeric.mlab')
    filestr, fromall3 = changeimports(filestr, 'LinearAlgebra', 'numpy.oldnumeric.linear_algebra')
    filestr, fromall3 = changeimports(filestr, 'RNG', 'numpy.oldnumeric.rng')
    filestr, fromall3 = changeimports(filestr, 'RNG.Statistics', 'numpy.oldnumeric.rng_stats')
    filestr, fromall3 = changeimports(filestr, 'RandomArray', 'numpy.oldnumeric.random_array')
    filestr, fromall3 = changeimports(filestr, 'FFT', 'numpy.oldnumeric.fft')
    filestr, fromall3 = changeimports(filestr, 'MA', 'numpy.oldnumeric.ma')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    today = datetime.date.today().strftime('%b %d, %Y')
    name = os.path.split(sys.argv[0])[-1]
    filestr = '## Automatically adapted for '\
              'numpy.oldnumeric %s by %s\n\n%s' % (today, name, filestr)
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
    """Convert all .py files to use numpy.oldnumeric (from Numeric) in the directory given

    For each file, a backup of <usesnumeric>.py is made as
    <usesnumeric>.py.orig.  A new file named <usesnumeric>.py
    is then written with the updated code.
    """
    files = glob.glob(os.path.join(direc,'*.py'))
    for afile in files:
        convertfile(afile)

if __name__ == '__main__':
    fromargs(sys.argv)
