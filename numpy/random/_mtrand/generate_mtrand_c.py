#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import sys
import re
import os

unused_internal_funcs = ['__Pyx_PrintItem',
                         '__Pyx_PrintNewline',
                         '__Pyx_ReRaise',
                         #'__Pyx_GetExcValue',
                         '__Pyx_ArgTypeTest',
                         '__Pyx_SetVtable',
                         '__Pyx_GetVtable',
                         '__Pyx_CreateClass']

if __name__ == '__main__':
    # Use cython here so that long docstrings are broken up.
    # This is needed for some VC++ compilers.
    os.system('cython mtrand.pyx')
    mtrand_c = open('mtrand.c', 'r')
    processed = open('mtrand_pp.c', 'w')
    unused_funcs_str = '(' + '|'.join(unused_internal_funcs) + ')'
    uifpat = re.compile(r'static \w+ \*?'+unused_funcs_str+r'.*/\*proto\*/')
    linepat = re.compile(r'/\* ".*/mtrand.pyx":')
    for linenum, line in enumerate(mtrand_c):
        m = re.match(r'^(\s+arrayObject\w*\s*=\s*[(])[(]PyObject\s*[*][)]',
                     line)
        if m:
            line = '%s(PyArrayObject *)%s' % (m.group(1), line[m.end():])
        m = uifpat.match(line)
        if m:
            line = ''
        m = re.search(unused_funcs_str, line)
        if m:
            print("%s was declared unused, but is used at line %d" % (m.group(),
                                                                    linenum+1), file=sys.stderr)
        line = linepat.sub(r'/* "mtrand.pyx":', line)
        processed.write(line)
    mtrand_c.close()
    processed.close()
    os.rename('mtrand_pp.c', 'mtrand.c')
