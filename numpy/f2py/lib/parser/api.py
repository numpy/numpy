"""
Public API for Fortran parser.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

# import all Statement classes:
from base_classes import EndStatement
from block_statements import *

# CHAR_BIT is used to convert object bit sizes to byte sizes
from utils import CHAR_BIT

def parse(input, isfree=None, isstrict=None, include_dirs = None):
    """ Parse input and return Statement tree.

    input            --- string or filename.
    isfree, isstrict --- specify input Fortran format.
                         Defaults are True, False, respectively, or
                         determined from input.
    include_dirs     --- list of include directories.
                         Default contains current working directory
                         and the directory of file name.
    """
    import os
    import re
    from readfortran import FortranFileReader, FortranStringReader
    from parsefortran import FortranParser
    if os.path.isfile(input):
        name,ext = os.path.splitext(input)
        if ext.lower() in ['.c']:
            # get signatures from C file comments starting with `/*f2py` and ending with `*/`.
            # TODO: improve parser to take line number offset making line numbers in
            #       parser messages correct.
            f2py_c_comments = re.compile('/[*]\s*f2py\s.*[*]/',re.I | re.M)
            f = open(filename,'r')
            c_input = ''
            for s1 in f2py_c_comments.findall(f.read()):
                c_input += s1[2:-2].lstrip()[4:] + '\n'
            f.close()
            if isfree is None: isfree = True
            if isstrict is None: isstrict = True
            return parse(c_input, isfree, isstrict, include_dirs)
        reader = FortranFileReader(input,
                                   include_dirs = include_dirs)
        if isfree is None: isfree = reader.isfree
        if isstrict is None: isstrict = reader.isstrict
        reader.set_mode(isfree, isstrict)
    elif isinstance(input, str):
        if isfree is None: isfree = True
        if isstrict is None: isstrict = False
        reader = FortranStringReader(input,
                                     isfree, isstrict,
                                     include_dirs = include_dirs)
    else:
        raise TypeError,'Expected string or filename input but got %s' % (type(input))
    parser = FortranParser(reader)
    parser.parse()
    parser.analyze()
    return parser.block

    
