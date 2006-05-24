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

from readfortran import Line

class Block:

    classes = []

    def __init__(self, parent):
        self.parent = parent
        self.content = []

        self.get_item = parent.get_item
        self.put_item = parent.put_item
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

    def isendline(self, line):
        if self.__class__ is Block:
            # MAIN block does not define start/end line conditions,
            # so it never ends.
            return False
        m = self.end_re.match(line)
        if not m: return False
        return True

    def fill(self):
        item = self.get_item()
        while item is not None:
            if isinstance(item, Line):
                line = item.line
                if self.isendline(line):
                    break
                # check if line contains subblock start
                found_block = False
                for cls in self.classes:
                    m = cls.start_re.match(line)
                    if m:
                        found_block = True
                        subblock = cls(self, m)
                        self.content.append(subblock)
                        subblock.fill()
                        break
                if found_block:
                    item = self.get_item()
                    continue
                # line contains something else
                self.content.append(item)
            item = self.get_item()
        return

class Program(Block):
    classes = []
    start_re = re.compile(r'\s*program', re.I)


class Module(Block):
    classes = []
    start_re = re.compile(r'\s*module\s(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*module(\s*(?P<name>\w+)|)|)\s*\Z')
    def __init__(self, parent, start_re_match):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Interface(Block):
    classes = []
    start_re = re.compile(r'\s*interface(\s*(?P<name>\w+)|)', re.I)
    end_re = re.compile(r'\s*end(\s*interface(\s*(?P<name>\w+)|)|)\s*\Z')
    def __init__(self, parent, start_re_match):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')


class PythonModule(Block):
    classes = []
    start_re = re.compile(r'\s*python\s*module\s(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*python\s*module(\s*(?P<name>\w+)|)|)\s*\Z')
    def __init__(self, parent, start_re_match):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')
    
class Subroutine(Block):
    classes = []
    start_re = re.compile(r'\s*subroutine\s*(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*subroutine(\s*(?P<name>.*)|)|)\s*\Z')
    def __init__(self, parent, start_re_match):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Function(Block):
    classes = []
    start_re = re.compile(r'\s*function\s*(?P<name>\w+)', re.I)
    end_re = re.compile(r'\s*end(\s*function(\s*(?P<name>.*)|)|)\s*\Z')
    def __init__(self, parent, start_re_match):
        Block.__init__(self, parent)
        self.name = start_re_match.group('name')

class Type(Block):
    classes = []
    start_re = re.compile(r'\s*type(?!\s*\()', re.I)
    end_re = re.compile(r'\s*end(\s*type(\s*(?P<name>.*)|)|)\s*\Z')
    def __init__(self, parent, start_re_match):
        Block.__init__(self, parent)
        self.name = 'notimpl'#start_re_match.group('name')

# Initialize classes lists
Block.classes.extend([Program,PythonModule,Module,Interface,Subroutine,Function,Type])
Module.classes.extend([PythonModule,Interface,Subroutine,Function,Type])
PythonModule.classes.extend(Module.classes)
Interface.classes.extend([PythonModule,Module,Interface,Subroutine,Function,Type])
Subroutine.classes.extend([PythonModule,Module,Interface,Subroutine,Function,Type])
Subroutine.classes.extend(Function.classes)
Type.classes.extend([Type,Function,Subroutine,Interface])
