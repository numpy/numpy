"""
This module converts code written for numarray to run with numpy

Makes the following changes:
 * Changes import statements

   import numarray.package
       --> import numpy.numarray.package as numarray_package
           with all numarray.package in code changed to numarray_package

   import numarray --> import numpy.numarray as numarray
   import numarray.package as <yyy> --> import numpy.numarray.package as <yyy>

   from numarray import <xxx> --> from numpy.numarray import <xxx>
   from numarray.package import <xxx>
       --> from numpy.numarray.package import <xxx>

   package can be convolve, image, nd_image, mlab, linear_algebra, ma,
                  matrix, fft, random_array


 * Makes search and replace changes to:
   - .imaginary --> .imag
   - .flat --> .ravel() (most of the time)
   - .byteswapped() --> .byteswap(False)
   - .byteswap() --> .byteswap(True)
   - .info() --> numarray.info(self)
   - .isaligned() --> .flags.aligned
   - .isbyteswapped() --> (not .dtype.isnative)
   - .typecode() --> .dtype.char
   - .iscontiguous() --> .flags.contiguous
   - .is_c_array() --> .flags.carray and .dtype.isnative
   - .is_fortran_contiguous() --> .flags.fortran
   - .is_f_array() --> .dtype.isnative and .flags.farray
   - .itemsize() --> .itemsize
   - .nelements() --> .size
   - self.new(type) --> numarray.newobj(self, type)
   - .repeat(r) --> .repeat(r, axis=0)
   - .size() --> .size
   - self.type() -- numarray.typefrom(self)
   - .typecode() --> .dtype.char
   - .stddev() --> .std()
   - .togglebyteorder() --> numarray.togglebyteorder(self)
   - .getshape() --> .shape
   - .setshape(obj) --> .shape=obj
   - .getflat() --> .ravel()
   - .getreal() --> .real
   - .setreal() --> .real =
   - .getimag() --> .imag
   - .setimag() --> .imag =
   - .getimaginary() --> .imag
   - .setimaginary() --> .imag
   
"""
__all__ = ['fromfile', 'fromstr', 'convertfile', 'convertall']

import sys
import os
import re
import glob



def changeimports(fstr, name, newname):
    importstr = 'import %s' % name
    importasstr = 'import %s as ' % name
    fromstr = 'from %s import ' % name
    fromall=0

    name_ = name
    if ('.' in name):
        name_ = name.replace('.','_')

    fstr = fstr.replace(importasstr, 'import %s as ' % newname)
    fstr = fstr.replace(importstr, 'import %s as %s' % (newname,name_))
    if (name_ != name):
        fstr = fstr.replace(name, name_)

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

flatindex_re = re.compile('([.]flat(\s*?[[=]))')


def addimport(astr):
    # find the first line with import on it
    ind = astr.find('import')
    start = astr.rfind(os.linesep, 0, ind)
    astr = "%s%s%s%s" % (astr[:start], os.linesep,
                         "import numpy.numarray as numarray",
                         astr[start:])
    return astr

def replaceattr(astr):
    astr = astr.replace(".imaginary", ".imag")
    astr = astr.replace(".byteswapped()",".byteswap(False)")
    astr = astr.replace(".byteswap()", ".byteswap(True)")
    astr = astr.replace(".isaligned()", ".flags.aligned")
    astr = astr.replace(".iscontiguous()",".flags.contiguous")
    astr = astr.replace(".is_fortran_contiguous()",".flags.fortran")
    astr = astr.replace(".itemsize()",".itemsize")
    astr = astr.replace(".size()",".size")
    astr = astr.replace(".nelements()",".size")    
    astr = astr.replace(".typecode()",".dtype.char")
    astr = astr.replace(".stddev()",".std()")
    astr = astr.replace(".getshape()", ".shape")
    astr = astr.replace(".getflat()", ".ravel()")
    astr = astr.replace(".getreal", ".real")
    astr = astr.replace(".getimag", ".imag")
    astr = astr.replace(".getimaginary", ".imag")            

    # preserve uses of flat that should be o.k.
    tmpstr = flatindex_re.sub(r"@@@@\2",astr)
    # replace other uses of flat
    tmpstr = tmpstr.replace(".flat",".ravel()")
    # put back .flat where it was valid
    astr = tmpstr.replace("@@@@", ".flat")
    return astr

info_re = re.compile(r'(\S+)\s*[.]\s*info\s*[(]\s*[)]')
new_re = re.compile(r'(\S+)\s*[.]\s*new\s*[(]\s*(\S+)\s*[)]')
toggle_re = re.compile(r'(\S+)\s*[.]\s*togglebyteorder\s*[(]\s*[)]')
type_re = re.compile(r'(\S+)\s*[.]\s*type\s*[(]\s*[)]')

isbyte_re = re.compile(r'(\S+)\s*[.]\s*isbyteswapped\s*[(]\s*[)]')
iscarr_re = re.compile(r'(\S+)\s*[.]\s*is_c_array\s*[(]\s*[)]')
isfarr_re = re.compile(r'(\S+)\s*[.]\s*is_f_array\s*[(]\s*[)]')
repeat_re = re.compile(r'(\S+)\s*[.]\s*repeat\s*[(]\s*(\S+)\s*[)]')

setshape_re = re.compile(r'(\S+)\s*[.]\s*setshape\s*[(]\s*(\S+)\s*[)]')
setreal_re = re.compile(r'(\S+)\s*[.]\s*setreal\s*[(]\s*(\S+)\s*[)]')
setimag_re = re.compile(r'(\S+)\s*[.]\s*setimag\s*[(]\s*(\S+)\s*[)]')
setimaginary_re = re.compile(r'(\S+)\s*[.]\s*setimaginary\s*[(]\s*(\S+)\s*[)]')
def replaceother(astr):
    # self.info() --> numarray.info(self)
    # self.new(type) --> numarray.newobj(self, type)
    # self.togglebyteorder() --> numarray.togglebyteorder(self)
    # self.type() --> numarray.typefrom(self)
    (astr, n1) = info_re.subn('numarray.info(\\1)', astr)
    (astr, n2) = new_re.subn('numarray.newobj(\\1, \\2)', astr)
    (astr, n3) = toggle_re.subn('numarray.togglebyteorder(\\1)', astr)
    (astr, n4) = type_re.subn('numarray.typefrom(\\1)', astr)
    if (n1+n2+n3+n4 > 0):
        astr = addimport(astr)

    astr = isbyte_re.sub('not \\1.dtype.isnative', astr)
    astr = iscarr_re.sub('\\1.dtype.isnative and \\1.flags.carray', astr)
    astr = isfarr_re.sub('\\1.dtype.isnative and \\1.flags.farray', astr)
    astr = repeat_re.sub('\\1.repeat(\\2, axis=0)', astr)
    astr = setshape_re.sub('\\1.shape = \\2', astr)
    astr = setreal_re.sub('\\1.real = \\2', astr)
    astr = setimag_re.sub('\\1.imag = \\2', astr)
    astr = setimaginary_re.sub('\\1.imag = \\2', astr)    
    return astr
    
import datetime
def fromstr(filestr):
    filestr, fromall = changeimports(filestr, 'numarray', 'numpy.numarray')
    base = 'numarray'
    newbase = 'numpy.numarray'
    for sub in ['', 'convolve', 'image', 'nd_image', 'mlab', 'linear_algebra',
                'ma', 'matrix', 'fft', 'random_array']:
        if sub != '':
            sub = '.'+sub
        filestr, fromall = changeimports(filestr, base+sub, newbase+sub)

    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    today = datetime.date.today().strftime('%b %d, %Y')
    name = os.path.split(sys.argv[0])[-1]
    filestr = '## Automatically adapted for '\
              'numpy.numarray %s by %s\n\n%s' % (today, name, filestr)
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
