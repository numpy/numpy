#!/usr/bin/env python
"""
summarize.py

Show a summary about which NumPy functions are documented and which are not.

"""
from __future__ import division, absolute_import, print_function

import os, glob, re, sys, inspect, optparse
try:
    # Accessing collections abstract classes from collections
    # has been deprecated since Python 3.3
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc
sys.path.append(os.path.join(os.path.dirname(__file__), 'sphinxext'))
from sphinxext.phantom_import import import_phantom_module

from sphinxext.autosummary_generate import get_documented

CUR_DIR = os.path.dirname(__file__)
SOURCE_DIR = os.path.join(CUR_DIR, 'source', 'reference')

SKIP_LIST = """
# --- aliases:
alltrue sometrue bitwise_not cumproduct
row_stack column_stack product rank

# -- skipped:
core lib f2py dual doc emath ma rec char distutils oldnumeric numarray
testing version matlib

add_docstring add_newdoc add_newdocs fastCopyAndTranspose pkgload
conjugate disp

int0 object0 unicode0 uint0 string_ string0 void0

flagsobj

setup PackageLoader

lib.scimath.arccos lib.scimath.arcsin lib.scimath.arccosh lib.scimath.arcsinh
lib.scimath.arctanh lib.scimath.log lib.scimath.log2 lib.scimath.log10
lib.scimath.logn lib.scimath.power lib.scimath.sqrt

# --- numpy.random:
random random.info random.mtrand random.ranf random.sample random.random

# --- numpy.fft:
fft fft.Tester fft.bench fft.fftpack fft.fftpack_lite fft.helper
fft.info fft.test

# --- numpy.linalg:
linalg linalg.Tester
linalg.bench linalg.info linalg.lapack_lite linalg.linalg linalg.test

# --- numpy.ctypeslib:
ctypeslib ctypeslib.test

""".split()

def main():
    p = optparse.OptionParser(__doc__)
    p.add_option("-c", "--columns", action="store", type="int", dest="cols",
                 default=3, help="Maximum number of columns")
    options, args = p.parse_args()

    if len(args) != 0:
        p.error('Wrong number of arguments')

    # prepare
    fn = os.path.join(CUR_DIR, 'dump.xml')
    if os.path.isfile(fn):
        import_phantom_module(fn)

    # check
    documented, undocumented = check_numpy()

    # report
    in_sections = {}
    for name, locations in documented.items():
        for (filename, section, keyword, toctree) in locations:
            in_sections.setdefault((filename, section, keyword), []).append(name)

    print("Documented")
    print("==========\n")

    last_filename = None
    for (filename, section, keyword), names in sorted(in_sections.items()):
        if filename != last_filename:
            print("--- %s\n" % filename)
        last_filename = filename
        print(" ** ", section)
        print(format_in_columns(sorted(names), options.cols))
        print("\n")

    print("")
    print("Undocumented")
    print("============\n")
    print(format_in_columns(sorted(undocumented.keys()), options.cols))

def check_numpy():
    documented = get_documented(glob.glob(SOURCE_DIR + '/*.rst'))
    undocumented = {}

    import numpy, numpy.fft, numpy.linalg, numpy.random
    for mod in [numpy, numpy.fft, numpy.linalg, numpy.random,
                numpy.ctypeslib, numpy.emath, numpy.ma]:
        undocumented.update(get_undocumented(documented, mod, skip=SKIP_LIST))

    for d in (documented, undocumented):
        for k in d.keys():
            if k.startswith('numpy.'):
                d[k[6:]] = d[k]
                del d[k]

    return documented, undocumented

def get_undocumented(documented, module, module_name=None, skip=[]):
    """
    Find out which items in NumPy are not documented.

    Returns
    -------
    undocumented : dict of bool
        Dictionary containing True for each documented item name
        and False for each undocumented one.

    """
    undocumented = {}

    if module_name is None:
        module_name = module.__name__

    for name in dir(module):
        obj = getattr(module, name)
        if name.startswith('_'): continue

        full_name = '.'.join([module_name, name])

        if full_name in skip: continue
        if full_name.startswith('numpy.') and full_name[6:] in skip: continue
        if not (inspect.ismodule(obj) or isinstance(obj, collections_abc.Callable) or inspect.isclass(obj)):
            continue

        if full_name not in documented:
            undocumented[full_name] = True

    return undocumented

def format_in_columns(lst, max_columns):
    """
    Format a list containing strings to a string containing the items
    in columns.
    """
    lst = [str(_m) for _m in lst]
    col_len = max([len(_m) for _m in lst]) + 2
    ncols = 80//col_len
    if ncols > max_columns:
        ncols = max_columns
    if ncols <= 0:
        ncols = 1

    if len(lst) % ncols == 0:
        nrows = len(lst)//ncols
    else:
        nrows = 1 + len(lst)//ncols

    fmt = ' %%-%ds ' % (col_len-2)

    lines = []
    for n in range(nrows):
        lines.append("".join([fmt % x for x in lst[n::nrows]]))
    return "\n".join(lines)

if __name__ == "__main__": main()
