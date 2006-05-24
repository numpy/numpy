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

__all__ = ['LineSplitter','String']

class String(str): pass

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

if __name__ == '__main__':
    test()

