"""
This module adds the default axis argument to code which did not specify it
for the functions where the default was changed in NumPy.

The functions changed are

add -1  ( all second argument)
======
nansum
nanmax
nanmin
nanargmax
nanargmin
argmax
argmin
compress 3


add 0
======
take     3
repeat   3
sum         # might cause problems with builtin.
product
sometrue
alltrue
cumsum
cumproduct
average
ptp
cumprod
prod
std
mean
"""
__all__ = ['convertfile', 'convertall', 'converttree']

import sys
import os
import re
import glob


_args3 = ['compress', 'take', 'repeat']
_funcm1 = ['nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin',
           'argmax', 'argmin', 'compress']
_func0 = ['take', 'repeat', 'sum', 'product', 'sometrue', 'alltrue',
          'cumsum', 'cumproduct', 'average', 'ptp', 'cumprod', 'prod',
          'std', 'mean']

_all = _func0 + _funcm1
func_re = {}

for name in _all:
    _astr = r"""%s\s*[(]"""%name
    func_re[name] = re.compile(_astr)


import string
disallowed = '_' + string.uppercase + string.lowercase + string.digits

def _add_axis(fstr, name, repl):
    alter = 0
    if name in _args3:
        allowed_comma = 1
    else:
        allowed_comma = 0
    newcode = ""
    last = 0
    for obj in func_re[name].finditer(fstr):
        nochange = 0
        start, end = obj.span()
        if fstr[start-1] in disallowed:
            continue
        if fstr[start-1] == '.' \
           and fstr[start-6:start-1] != 'numpy' \
           and fstr[start-2:start-1] != 'N' \
           and fstr[start-9:start-1] != 'numarray' \
           and fstr[start-8:start-1] != 'numerix' \
           and fstr[start-8:start-1] != 'Numeric':
            continue
        if fstr[start-1] in ['\t',' ']:
            k = start-2
            while fstr[k] in ['\t',' ']:
                k -= 1
            if fstr[k-2:k+1] == 'def' or \
               fstr[k-4:k+1] == 'class':
                continue
        k = end
        stack = 1
        ncommas = 0
        N = len(fstr)
        while stack:
            if k>=N:
                nochange =1
                break
            if fstr[k] == ')':
                stack -= 1
            elif fstr[k] == '(':
                stack += 1
            elif stack == 1 and fstr[k] == ',':
                ncommas += 1
                if ncommas > allowed_comma:
                    nochange = 1
                    break
            k += 1
        if nochange:
            continue
        alter += 1
        newcode = "%s%s,%s)" % (newcode, fstr[last:k-1], repl)
        last = k
    if not alter:
        newcode = fstr
    else:
        newcode = "%s%s" % (newcode, fstr[last:])
    return newcode, alter

def _import_change(fstr, names):
    # Four possibilities
    #  1.) import numpy with subsequent use of numpy.<name>
    #        change this to import numpy.oldnumeric as numpy
    #  2.) import numpy as XXXX with subsequent use of
    #        XXXX.<name> ==> import numpy.oldnumeric as XXXX
    #  3.) from numpy import *
    #        with subsequent use of one of the names
    #  4.) from numpy import ..., <name>, ... (could span multiple
    #        lines.  ==> remove all names from list and
    #        add from numpy.oldnumeric import <name>

    num = 0
    # case 1
    importstr = "import numpy"
    ind = fstr.find(importstr)
    if (ind > 0):
        found = 0
        for name in names:
            ind2 = fstr.find("numpy.%s" % name, ind)
            if (ind2 > 0):
                found = 1
                break
        if found:
            fstr = "%s%s%s" % (fstr[:ind], "import numpy.oldnumeric as numpy",
                               fstr[ind+len(importstr):])
            num += 1

    # case 2
    importre = re.compile("""import numpy as ([A-Za-z0-9_]+)""")
    modules = importre.findall(fstr)
    if len(modules) > 0:
        for module in modules:
            found = 0
            for name in names:
                ind2 = fstr.find("%s.%s" % (module, name))
                if (ind2 > 0):
                    found = 1
                    break
            if found:
                importstr = "import numpy as %s" % module
                ind = fstr.find(importstr)
                fstr = "%s%s%s" % (fstr[:ind],
                                   "import numpy.oldnumeric as %s" % module,
                                   fstr[ind+len(importstr):])
                num += 1

    # case 3
    importstr = "from numpy import *"
    ind = fstr.find(importstr)
    if (ind > 0):
        found = 0
        for name in names:
            ind2 = fstr.find(name, ind)
            if (ind2 > 0) and fstr[ind2-1] not in disallowed:
                found = 1
                break
        if found:
            fstr = "%s%s%s" % (fstr[:ind],
                               "from numpy.oldnumeric import *",
                               fstr[ind+len(importstr):])
            num += 1

    # case 4
    ind = 0
    importstr = "from numpy import"
    N = len(importstr)
    while 1:
        ind = fstr.find(importstr, ind)
        if (ind < 0):
            break
        ind += N
        ptr = ind+1
        stack = 1
        while stack:
            if fstr[ptr] == '\\':
                stack += 1
            elif fstr[ptr] == '\n':
                stack -= 1
            ptr += 1
        substr = fstr[ind:ptr]
        found = 0
        substr = substr.replace('\n',' ')
        substr = substr.replace('\\','')
        importnames = [x.strip() for x in substr.split(',')]
        # determine if any of names are in importnames
        addnames = []
        for name in names:
            if name in importnames:
                importnames.remove(name)
                addnames.append(name)
        if len(addnames) > 0:
            fstr = "%s%s\n%s\n%s" % \
                   (fstr[:ind],
                    "from numpy import %s" % \
                    ", ".join(importnames),
                    "from numpy.oldnumeric import %s" % \
                    ", ".join(addnames),
                    fstr[ptr:])
            num += 1

    return fstr, num

def add_axis(fstr, import_change=False):
    total = 0
    if not import_change:
        for name in _funcm1:
            fstr, num = _add_axis(fstr, name, 'axis=-1')
            total += num
        for name in _func0:
            fstr, num = _add_axis(fstr, name, 'axis=0')
            total += num
        return fstr, total
    else:
        fstr, num = _import_change(fstr, _funcm1+_func0)
        return fstr, num


def makenewfile(name, filestr):
    fid = file(name, 'w')
    fid.write(filestr)
    fid.close()

def getfile(name):
    fid = file(name)
    filestr = fid.read()
    fid.close()
    return filestr

def copyfile(name, fstr):
    base, ext = os.path.splitext(name)
    makenewfile(base+'.orig', fstr)
    return

def convertfile(filename, import_change=False):
    """Convert the filename given from using Numeric to using NumPy

    Copies the file to filename.orig and then over-writes the file
    with the updated code
    """
    filestr = getfile(filename)
    newstr, total = add_axis(filestr, import_change)
    if total > 0:
        print "Changing ", filename
        copyfile(filename, filestr)
        makenewfile(filename, newstr)
        sys.stdout.flush()

def fromargs(args):
    filename = args[1]
    convertfile(filename)

def convertall(direc=os.path.curdir, import_change=False):
    """Convert all .py files in the directory given

    For each file, a backup of <usesnumeric>.py is made as
    <usesnumeric>.py.orig.  A new file named <usesnumeric>.py
    is then written with the updated code.
    """
    files = glob.glob(os.path.join(direc,'*.py'))
    for afile in files:
        convertfile(afile, import_change)

def _func(arg, dirname, fnames):
    convertall(dirname, import_change=arg)

def converttree(direc=os.path.curdir, import_change=False):
    """Convert all .py files in the tree given

    """
    os.path.walk(direc, _func, import_change)

if __name__ == '__main__':
    fromargs(sys.argv)
