"""
Provides get_source_info(<filename>) function to determine the format
(free|fixed|strict|pyf) of a Fortran file.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['get_source_info']

import re
import os
import sys

_has_f_extension = re.compile(r'.*[.](for|ftn|f77|f)\Z',re.I).match
_has_f_header = re.compile(r'-[*]-\s*fortran\s*-[*]-',re.I).search
_has_f90_header = re.compile(r'-[*]-\s*f90\s*-[*]-',re.I).search
_has_fix_header = re.compile(r'-[*]-\s*fix\s*-[*]-',re.I).search
_free_f90_start = re.compile(r'[^c*!]\s*[^\s\d\t]',re.I).match

def get_source_info(filename):
    """
    Determine if fortran file is
      - in fix format and contains Fortran 77 code    -> return False, True
      - in fix format and contains Fortran 90 code    -> return False, False
      - in free format and contains Fortran 90 code   -> return True, False
      - in free format and contains signatures (.pyf) -> return True, True
    """
    base,ext = os.path.splitext(filename)
    if ext=='.pyf':
        return True, True
    isfree = False
    isstrict = False
    f = open(filename,'r')
    firstline = f.readline()
    f.close()
    if _has_f_extension(filename) and \
       not (_has_f90_header(firstline) or _has_fix_header(firstline)):
        isstrict = True
    elif is_free_format(filename) and not _has_fix_header(firstline):
        isfree = True
    return isfree,isstrict

def is_free_format(file):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    isfree = False
    f = open(file,'r')
    line = f.readline()
    n = 10000 # the number of non-comment lines to scan for hints
    if _has_f_header(line):
        n = 0
    elif _has_f90_header(line):
        n = 0
        isfree = True
    contline = False
    while n>0 and line:
        line = line.rstrip()
        if line and line[0]!='!':
            n -= 1
            if line[0]!='\t' and _free_f90_start(line[:5]) or line[-1:]=='&':
                isfree = True
                break
        line = f.readline()
    f.close()
    return isfree

def simple_main():
    for filename in sys.argv[1:]:
        isfree, isstrict = get_source_info(filename)
        print '%s: isfree=%s, isstrict=%s'  % (filename, isfree, isstrict)
    
if __name__ == '__main__':
    simple_main()
