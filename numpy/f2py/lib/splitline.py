#!/usr/bin/env python
"""
Defines LineSplitter.

Copyright 2006 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision:$
$Date: 2000/07/31 07:04:03 $
Pearu Peterson
"""

__all__ = ['LineSplitter','String','split2','string_replace_map']

class String(str): pass

def split2(line, lower=False):
    """
    Split line into non-string part and into a start of a string part.
    Returns 2-tuple. The second item either is empty string or start
    of a string part.
    """
    return LineSplitter(line,lower=lower).split2()

def string_replace_map(line, lower=False, _cache={'index':0}):
    """
    Replaces string constants with name _F2PY_STRING_CONSTANT_<index>
    and returns a new line and a map
      {_F2PY_STRING_CONSTANT_<index>: <original string constant>}
    """
    items = []
    string_map = {}
    rev_string_map = {}
    for item in LineSplitter(line, lower=lower):
        if isinstance(item, String):
            key = rev_string_map.get(item)
            if key is None:
                _cache['index'] += 1
                index = _cache['index']
                key = '_F2PY_STRING_CONSTANT_%s' % (index)
                string_map[key] = item
                rev_string_map[item] = key
            items.append(key)
        else:
            items.append(item)
    return ''.join(items),string_map

class LineSplitter:
    """ Splits a line into non strings and strings. E.g.
    abc=\"123\" -> ['abc=','\"123\"']
    Handles splitting lines with incomplete string blocks.
    """
    def __init__(self, line, quotechar = None, lower=False):
        self.fifo_line = [c for c in line]
        self.fifo_line.reverse()
        self.quotechar = quotechar
        self.lower = lower

    def __iter__(self):
        return self

    def next(self):
        item = ''
        while not item:
            item = self.get_item() # get_item raises StopIteration
        return item

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
                
def test():
    splitter = LineSplitter('abc\\\' def"12\\"3""56"dfad\'a d\'')
    l = [item for item in splitter]
    assert l==['abc\\\' def','"12\\"3"','"56"','dfad','\'a d\''],`l`
    assert splitter.quotechar is None

    splitter = LineSplitter('"abc123&')
    l = [item for item in splitter]
    assert l==['"abc123&'],`l`
    assert splitter.quotechar=='"'

    splitter = LineSplitter(' &abc"123','"')
    l = [item for item in splitter]
    assert l==[' &abc"','123']
    assert splitter.quotechar is None

    l = split2('')
    assert l==('',''),`l`
    l = split2('12')
    assert l==('12',''),`l`
    l = split2('1"a"//"b"')
    assert l==('1','"a"//"b"'),`l`
    l = split2('"ab"')
    assert l==('','"ab"'),`l`
if __name__ == '__main__':
    test()

