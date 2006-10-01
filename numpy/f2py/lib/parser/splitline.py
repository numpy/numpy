#!/usr/bin/env python
"""
Defines LineSplitter and helper functions.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['String','string_replace_map','splitquote','splitparen']

import re

class String(str): pass
class ParenString(str): pass

def split2(line, lower=False):
    """
    Split line into non-string part and into a start of a string part.
    Returns 2-tuple. The second item either is empty string or start
    of a string part.
    """
    return LineSplitter(line,lower=lower).split2()

_f2py_str_findall = re.compile(r"_F2PY_STRING_CONSTANT_\d+_").findall
_is_name = re.compile(r'\w*\Z',re.I).match
_is_simple_str = re.compile(r'\w*\Z',re.I).match

def string_replace_map(line, lower=False,
                       _cache={'index':0,'pindex':0}):
    """
    1) Replaces string constants with symbol `'_F2PY_STRING_CONSTANT_<index>_'`
    2) Replaces (expression) with symbol `(F2PY_EXPR_TUPLE_<index>)`
    Returns a new line and the replacement map.
    """
    items = []
    string_map = {}
    rev_string_map = {}
    for item in splitquote(line, lower=lower)[0]:
        if isinstance(item, String) and not _is_simple_str(item[1:-1]):
            key = rev_string_map.get(item)
            if key is None:
                _cache['index'] += 1
                index = _cache['index']
                key = "_F2PY_STRING_CONSTANT_%s_" % (index)
                it = item[1:-1]
                string_map[key] = it
                rev_string_map[it] = key
            items.append(item[0]+key+item[-1])
        else:
            items.append(item)
    newline = ''.join(items)
    items = []
    expr_keys = []
    for item in splitparen(newline):
        if isinstance(item, ParenString) and not _is_name(item[1:-1]):
            key = rev_string_map.get(item)
            if key is None:
                _cache['pindex'] += 1
                index = _cache['pindex']
                key = 'F2PY_EXPR_TUPLE_%s' % (index)
                it = item[1:-1].strip()
                string_map[key] = it
                rev_string_map[it] = key
                expr_keys.append(key)
            items.append(item[0]+key+item[-1])
        else:
            items.append(item)
    found_keys = set()
    for k in expr_keys:
        v = string_map[k]
        l = _f2py_str_findall(v)
        if l:
            found_keys = found_keys.union(l)
            for k1 in l:
                v = v.replace(k1, string_map[k1])
            string_map[k] = v
    for k in found_keys:
        del string_map[k]
    return ''.join(items), string_map

def splitquote(line, stopchar=None, lower=False, quotechars = '"\''):
    """
    Fast LineSplitter
    """
    items = []
    i = 0
    while 1:
        try:
            char = line[i]; i += 1
        except IndexError:
            break
        l = []
        l_append = l.append
        nofslashes = 0
        if stopchar is None:
            # search for string start
            while 1:
                if char in quotechars and not nofslashes % 2:
                    stopchar = char
                    i -= 1
                    break
                if char=='\\':
                    nofslashes += 1
                else:
                    nofslashes = 0
                l_append(char)
                try:
                    char = line[i]; i += 1
                except IndexError:
                    break
            if not l: continue
            item = ''.join(l)
            if lower: item = item.lower()
            items.append(item)
            continue
        if char==stopchar:
            # string starts with quotechar
            l_append(char)
            try:
                char = line[i]; i += 1
            except IndexError:
                if l:
                    item = String(''.join(l))
                    items.append(item)
                break
        # else continued string
        while 1:
            if char==stopchar and not nofslashes % 2:
                l_append(char)
                stopchar = None
                break
            if char=='\\':
                nofslashes += 1
            else:
                nofslashes = 0
            l_append(char)
            try:
                char = line[i]; i += 1
            except IndexError:
                break
        if l:
            item = String(''.join(l))
            items.append(item)
    return items, stopchar

class LineSplitterBase:

    def __iter__(self):
        return self

    def next(self):
        item = ''
        while not item:
            item = self.get_item() # get_item raises StopIteration
        return item

class LineSplitter(LineSplitterBase):
    """ Splits a line into non strings and strings. E.g.
    abc=\"123\" -> ['abc=','\"123\"']
    Handles splitting lines with incomplete string blocks.
    """
    def __init__(self, line,
                 quotechar = None,
                 lower=False,
                 ):
        self.fifo_line = [c for c in line]
        self.fifo_line.reverse()
        self.quotechar = quotechar
        self.lower = lower

    def split2(self):
        """
        Split line until the first start of a string.
        """
        try:
            item1 = self.get_item()
        except StopIteration:
            return '',''
        i = len(item1)
        l = self.fifo_line[:]
        l.reverse()
        item2 = ''.join(l)
        return item1,item2

    def get_item(self):
        fifo_pop = self.fifo_line.pop
        try:
            char = fifo_pop()
        except IndexError:
            raise StopIteration
        fifo_append = self.fifo_line.append
        quotechar = self.quotechar
        l = []
        l_append = l.append
        
        nofslashes = 0
        if quotechar is None:
            # search for string start
            while 1:
                if char in '"\'' and not nofslashes % 2:
                    self.quotechar = char
                    fifo_append(char)
                    break
                if char=='\\':
                    nofslashes += 1
                else:
                    nofslashes = 0
                l_append(char)
                try:
                    char = fifo_pop()
                except IndexError:
                    break
            item = ''.join(l)
            if self.lower: item = item.lower()
            return item

        if char==quotechar:
            # string starts with quotechar
            l_append(char)
            try:
                char = fifo_pop()
            except IndexError:
                return String(''.join(l))
        # else continued string
        while 1:
            if char==quotechar and not nofslashes % 2:
                l_append(char)
                self.quotechar = None
                break
            if char=='\\':
                nofslashes += 1
            else:
                nofslashes = 0
            l_append(char)
            try:
                char = fifo_pop()
            except IndexError:
                break
        return String(''.join(l))

def splitparen(line,paren='()'):
    """
    Fast LineSplitterParen.
    """
    stopchar = None
    startchar, endchar = paren[0],paren[1]

    items = []
    i = 0
    while 1:
        try:
            char = line[i]; i += 1
        except IndexError:
            break
        nofslashes = 0
        l = []
        l_append = l.append
        if stopchar is None:
            # search for parenthesis start
            while 1:
                if char==startchar and not nofslashes % 2:
                    stopchar = endchar
                    i -= 1
                    break
                if char=='\\':
                    nofslashes += 1
                else:
                    nofslashes = 0
                l_append(char)
                try:
                    char = line[i]; i += 1
                except IndexError:
                    break
            item = ''.join(l)
        else:
            nofstarts = 0
            while 1:
                if char==stopchar and not nofslashes % 2 and nofstarts==1:
                    l_append(char)
                    stopchar = None
                    break
                if char=='\\':
                    nofslashes += 1
                else:
                    nofslashes = 0
                if char==startchar:
                    nofstarts += 1
                elif char==endchar:
                    nofstarts -= 1
                l_append(char)
                try:
                    char = line[i]; i += 1
                except IndexError:
                    break
            item = ParenString(''.join(l))
        items.append(item)
    return items

class LineSplitterParen(LineSplitterBase):
    """ Splits a line into strings and strings with parenthesis. E.g.
    a(x) = b(c,d) -> ['a','(x)',' = b','(c,d)']
    """
    def __init__(self, line, paren = '()'):
        self.fifo_line = [c for c in line]
        self.fifo_line.reverse()
        self.startchar = paren[0]
        self.endchar = paren[1]
        self.stopchar = None
        
    def get_item(self):
        fifo_pop = self.fifo_line.pop
        try:
            char = fifo_pop()
        except IndexError:
            raise StopIteration
        fifo_append = self.fifo_line.append
        startchar = self.startchar
        endchar = self.endchar
        stopchar = self.stopchar
        l = []
        l_append = l.append
        
        nofslashes = 0
        if stopchar is None:
            # search for parenthesis start
            while 1:
                if char==startchar and not nofslashes % 2:
                    self.stopchar = endchar
                    fifo_append(char)
                    break
                if char=='\\':
                    nofslashes += 1
                else:
                    nofslashes = 0
                l_append(char)
                try:
                    char = fifo_pop()
                except IndexError:
                    break
            item = ''.join(l)
            return item

        nofstarts = 0
        while 1:
            if char==stopchar and not nofslashes % 2 and nofstarts==1:
                l_append(char)
                self.stopchar = None
                break
            if char=='\\':
                nofslashes += 1
            else:
                nofslashes = 0
            if char==startchar:
                nofstarts += 1
            elif char==endchar:
                nofstarts -= 1
            l_append(char)
            try:
                char = fifo_pop()
            except IndexError:
                break
        return ParenString(''.join(l))
                
def test():
    splitter = LineSplitter('abc\\\' def"12\\"3""56"dfad\'a d\'')
    l = [item for item in splitter]
    assert l==['abc\\\' def','"12\\"3"','"56"','dfad','\'a d\''],`l`
    assert splitter.quotechar is None
    l,stopchar=splitquote('abc\\\' def"12\\"3""56"dfad\'a d\'')
    assert l==['abc\\\' def','"12\\"3"','"56"','dfad','\'a d\''],`l`
    assert stopchar is None

    splitter = LineSplitter('"abc123&')
    l = [item for item in splitter]
    assert l==['"abc123&'],`l`
    assert splitter.quotechar=='"'
    l,stopchar = splitquote('"abc123&')
    assert l==['"abc123&'],`l`
    assert stopchar=='"'
    
    splitter = LineSplitter(' &abc"123','"')
    l = [item for item in splitter]
    assert l==[' &abc"','123']
    assert splitter.quotechar is None
    l,stopchar = splitquote(' &abc"123','"')
    assert l==[' &abc"','123']
    assert stopchar is None
        
    l = split2('')
    assert l==('',''),`l`
    l = split2('12')
    assert l==('12',''),`l`
    l = split2('1"a"//"b"')
    assert l==('1','"a"//"b"'),`l`
    l = split2('"ab"')
    assert l==('','"ab"'),`l`

    splitter = LineSplitterParen('a(b) = b(x,y(1)) b\((a)\)')
    l = [item for item in splitter]
    assert l==['a', '(b)', ' = b', '(x,y(1))', ' b\\(', '(a)', '\\)'],`l`
    l = splitparen('a(b) = b(x,y(1)) b\((a)\)')
    assert l==['a', '(b)', ' = b', '(x,y(1))', ' b\\(', '(a)', '\\)'],`l`

    l = string_replace_map('a()')
    print l
    print 'ok'

if __name__ == '__main__':
    test()

