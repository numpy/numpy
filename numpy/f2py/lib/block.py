#!/usr/bin/env python
"""
Defines Block classes.

Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006

"""

__all__ = ['Block','Module','PythonModule','Interface',
           'Subroutine','Function','Type']

import re
import sys

from readfortran import Line
from splitline import split2, string_replace_map

class Block:

    classes = []
    end_re = re.compile(r'\s*end\s*\Z', re.I)

    def __init__(self, parent):
        self.parent = parent
        self.isfix77 = parent.isfix77
        self.reader = parent.reader
        self.get_item = parent.get_item
        self.put_item = parent.put_item
        self.lower = not self.reader.ispyf

        self.content = []
        self.name = None

    def get_name(self):
        if self.__class__ is Block: return '__MAIN__'
        if self.name is None: return ''
        return self.name

    def __str__(self):
        tab = ''
        p = self.parent
        while isinstance(p, Block):
            tab += '  '
            p = p.parent
        name = self.get_name()
        l=[tab+'begin '+self.__class__.__name__ +' '+ name]
        for c in self.content:
            l.append(str(c))
        l.append(tab+'end '+self.__class__.__name__ +' '+ name)
        return '\n'.join(l)

    def isenditem(self, item):
        line,sline = split2(item.get_line())
        if sline: return False # end statement never contains strings 
        if self.__class__ is Block:
            # MAIN block does not define start/end line conditions,
            # so it should never end until all lines are read.
            # However, sometimes F77 programs lack the PROGRAM statement,
            # and here we fix that:
            if self.isfix77:
                m = self.end_re.match(line)
                if m:
                    message = self.reader.format_message(\
                        'WARNING',
                        'assuming the end of undefined PROGRAM statement',
                        item.span[0],item.span[1])
                    print >> sys.stderr, message
                    p = Program(self,Program.start_re.match('program UNDEFINED'))
                    p.content.extend(self.content)
                    self.content[:] = [p]
                    return None
            return False
        m = self.end_re.match(line)
        if not m: return False
        # check if the block start name matches with the block end name

        if m.groupdict().has_key('name'):
            end_name = m.group('name')
            name = self.get_name()
            if end_name and name != end_name:
                message = self.reader.format_message(\
                        'WARNING',
                        'expected the end of %r block but got end of %r'\
                        % (name, end_name),
                        item.span[0],item.span[1])
                print >> sys.stderr, message
        return True

    def isblock(self, item):
        line = item.get_line()
        for cls in self.classes:
            m = cls.start_re.match(line)
            if m:
                subblock = cls(self, m, item)
                self.content.append(subblock)
                subblock.fill()
                return True
        return False

    def fill(self):
        item = self.get_item()
        while item is not None:
            if isinstance(item, Line):
                # handle end of a block
                flag = self.isenditem(item)
                if flag: # end of block
                    break
                if flag is None: # fixing the end of undefined start
                    item = self.get_item()
                    continue
                # handle subblocks
                if self.isblock(item):
                    item = self.get_item()
                    continue                    

                # line contains something else
                self.content.append(item)
            item = self.get_item()
        return

class Program(Block):
    classes = []
    start_re = re.compile(r'\s*program\s*((?P<name>\w+)|)', re.I)
    end_re = re.compile(r'\s*end(\s*program(\s*(?P<name>\w+)|)|)\s*\Z', re.I)
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Module(Block):
    classes = []
    start_re = re.compile(r'\s*module\s*(?P<name>\w+)\s*\Z', re.I)
    end_re = re.compile(r'\s*end(\s*module(\s*(?P<name>\w+)|)|)\s*\Z', re.I)
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Interface(Block):
    classes = []
    start_re = re.compile(r'\s*interface(\s*(?P<name>\w+)|)', re.I)
    end_re = re.compile(r'\s*end(\s*interface(\s*(?P<name>\w+)|)|)\s*\Z', re.I)
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')


class PythonModule(Block):
    classes = []
    start_re = re.compile(r'\s*python\s*module\s*(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*python\s*module(\s*(?P<name>\w+)|)|)\s*\Z', re.I)
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')
    
class Subroutine(Block):
    classes = []
    start_re = re.compile(r'\s*subroutine\s*(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*subroutine(\s*(?P<name>\w+)|)|)\s*\Z', re.I)
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Function(Block):
    classes = []
    start_re = re.compile(r'\s*function\s*(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*function(\s*(?P<name>\w+)|)|)\s*\Z')
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Type(Block):
    classes = []
    start_re = re.compile(r'\s*type(?!\s*\()(.*::|)\s*(?P<name>\w+)\s*\Z', re.I)
    end_re = re.compile(r'\s*end(\s*type(\s*(?P<name>\w+)|)|)\s*\Z', re.I)
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')


class StatementBlock(Block):
    classes = []

    def fill(self):
        item = self.get_item()
        while item is not None:
            if isinstance(item, Line):
                # handle end of a block
                flag = self.isenditem(item)
                if flag: # end of block
                    break
                # handle subblocks
                if self.isblock(item):
                    item = self.get_item()
                    continue                    

                # line contains something else
                self.content.append(item)
            item = self.get_item()
        return

class DoBlock(StatementBlock):

    start_re = re.compile(r'\s*do\b', re.I)
    end_re = re.compile(r'\s*end\s*do(\s*(?P<name>.*)|)\s*\Z', re.I)
    
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = item.label

class IfThenBlock(StatementBlock):

    start_re = re.compile(r'\s*if\b.*?\bthen\s*\Z', re.I)
    #start_re = re.compile(r'\s*if\b', re.I)
    end_re = re.compile(r'\s*end\s*if(\s*(?P<name>.*)|)\s*\Z', re.I)
    
    def __init__(self, parent, start_re_match, item):
        Block.__init__(self, parent)
        self.name = item.label


# Initialize classes lists

basic_blocks = [Program,PythonModule,Module,Interface,Subroutine,Function,Type]
stmt_blocks = [DoBlock,IfThenBlock]

Block.classes.extend(basic_blocks + stmt_blocks)
Module.classes.extend(Block.classes[1:])
PythonModule.classes.extend(Module.classes)
Interface.classes.extend(Block.classes[1:])
Subroutine.classes.extend(Block.classes[1:])
Function.classes.extend(Subroutine.classes)
Type.classes.extend(Block.classes[3:])

StatementBlock.classes.extend(stmt_blocks)
