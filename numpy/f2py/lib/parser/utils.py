"""
Various utility functions.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['split_comma', 'specs_split_comma',
           'ParseError','AnalyzeError',
           'get_module_file','parse_bind','parse_result','is_name',
           'CHAR_BIT','str2stmt']

import re
import os, glob

class ParseError(Exception):
    pass

class AnalyzeError(Exception):
    pass

is_name = re.compile(r'^[a-z_]\w*$',re.I).match
name_re = re.compile(r'[a-z_]\w*',re.I).match
is_entity_decl = re.compile(r'^[a-z_]\w*',re.I).match
is_int_literal_constant = re.compile(r'^\d+(_\w+|)$').match

def split_comma(line, item = None, comma=','):
    items = []
    if item is None:
        for s in line.split(comma):
            s = s.strip()
            if not s: continue
            items.append(s)
        return items
    newitem = item.copy(line, True)
    apply_map = newitem.apply_map
    for s in newitem.get_line().split(comma):
        s = apply_map(s).strip()
        if not s: continue
        items.append(s)
    return items

def specs_split_comma(line, item = None, upper=False):
    specs0 = split_comma(line, item)
    specs = []
    for spec in specs0:
        i = spec.find('=')
        if i!=-1:
            kw = spec[:i].strip().upper()
            v  = spec[i+1:].strip()
            specs.append('%s = %s' % (kw, v))
        else:
            if upper:
                spec = spec.upper()
            specs.append(spec)
    return specs

def parse_bind(line, item = None):
    if not line.lower().startswith('bind'):
        return None, line
    if item is not None:
        newitem = item.copy(line, apply_map=True)
        newline = newitem.get_line()
    else:
        newitem = None
    newline = newline[4:].lstrip()
    i = newline.find(')')
    assert i!=-1,`newline`
    args = []
    for a in specs_split_comma(newline[1:i].strip(), newitem, upper=True):
        args.append(a)
    rest = newline[i+1:].lstrip()
    if item is not None:
        rest = newitem.apply_map(rest)
    return args, rest

def parse_result(line, item = None):
    if not line.lower().startswith('result'):
        return None, line
    line = line[6:].lstrip()
    i = line.find(')')
    assert i != -1,`line`
    name = line[1:i].strip()
    assert is_name(name),`name`
    return name, line[i+1:].lstrip()

def filter_stmts(content, classes):
    """ Pop and return classes instances from content.
    """
    stmts = []
    indices = []
    for i in range(len(content)):
        stmt = content[i]
        if isinstance(stmt, classes):
            stmts.append(stmt)
            indices.append(i)
    indices.reverse()
    for i in indices:
        del content[i]
    return stmts


def get_module_files(directory, _cache={}):
    if _cache.has_key(directory):
        return _cache[directory]
    module_line = re.compile(r'(\A|^)module\s+(?P<name>\w+)\s*(!.*|)$',re.I | re.M)
    d = {}
    for fn in glob.glob(os.path.join(directory,'*.f90')):
        f = open(fn,'r')
        for name in module_line.findall(f.read()):
            name = name[1]
            if d.has_key(name):
                print d[name],'already defines',name
                continue
            d[name] = fn
    _cache[directory] = d
    return d

def get_module_file(name, directory, _cache={}):
    fn = _cache.get(name, None)
    if fn is not None:
        return fn
    if name.endswith('_module'):
        f1 = os.path.join(directory,name[:-7]+'.f90')
        if os.path.isfile(f1):
            _cache[name] = fn
            return f1
    pattern = re.compile(r'\s*module\s+(?P<name>[a-z]\w*)', re.I).match
    for fn in glob.glob(os.path.join(directory,'*.f90')):
        f = open(fn,'r')
        for line in f:
            m = pattern(line)
            if m and m.group('name')==name:
                _cache[name] = fn
                f.close()
                return fn
        f.close()
    return

def str2stmt(string, isfree=True, isstrict=False):
    """ Convert Fortran code to Statement tree.
    """
    from readfortran import Line, FortranStringReader
    from parsefortran import FortranParser
    reader = FortranStringReader(string, isfree, isstrict)
    parser = FortranParser(reader)
    parser.parse()
    parser.analyze()
    block = parser.block
    while len(block.content)==1:
        block = block.content[0]
    return block

def get_char_bit():
    import numpy
    one = numpy.ubyte(1)
    two = numpy.ubyte(2)
    n = numpy.ubyte(2)
    i = 1
    while n>=two:
        n <<= one
        i += 1
    return i

CHAR_BIT = get_char_bit()
