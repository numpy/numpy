#!/usr/bin/env python
"""
Defines Block classes.

Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006

"""

__all__ = ['Block','ModuleBlock','PythonModuleBlock','InterfaceBlock',
           'SubroutineBlock','FunctionBlock','TypeBlock', 'ProgramBlock',
           'BlockDataBlock','DoBlock','IfThenBlock','SelectBlock',
           'StatementBlock']

import re
import sys

from readfortran import Line
from splitline import string_replace_map
from stmt import statements, end_stmts, block_stmts

class Block:
    """
    Block instance has attributes:
      content - list of Line or Statement instances
      name    - name of the block, unnamed blocks are named
                with the line label
      parent  - Block or FortranParser instance
      item    - Line instance containing the block start statement
      get_item, put_item - methods to retrive/submit Line instaces
                from/to Fortran reader.
      isvalid - boolean, when False, the Block instance will be ignored.
    """

    classes = {}
    
    end_re = re.compile(r'end\Z', re.I).match

    def __init__(self, parent, item = None):
        """
        parent - Block or FortanParser instance having the
                 following attributes: reader, get_item, put_item
        item   - Line instance containing the beginning of block
                 statement.
        """
        self.parent = parent
        self.reader = parent.reader
        self.get_item = parent.get_item # get line function
        self.put_item = parent.put_item # put line function

        self.content = []
        self.name = None

        self.item = item
        if item is None:
            return
        stmt = self.stmt_cls(self, item)
        self.isvalid = stmt.isvalid
        if self.isvalid:
            self.content.append(stmt)
            self.name = stmt.name
            self.fill() # read block content

    def get_name(self):
        if self.__class__ is Block: return '__F2PY_MAIN__'
        if not hasattr(self,'name') or self.name is None: return ''
        return self.name

    def __str__(self):
        l=[]
        for c in self.content:
            l.append(str(c))
        return '\n'.join(l)

    def isstmt(self, item):
        """
        Check is item is blocks start statement, if it is, read the block.
        """
        line = item.get_line()
        mode = item.reader.mode
        classes = self.classes[mode] + statements[self.__class__.__name__]
        for cls in classes:
            if issubclass(cls, Block):
                match_cmd = cls.stmt_cls.start_re
            else:
                match_cmd = cls.start_re
            if match_cmd(line):
                subblock = cls(self, item)
                if subblock.isvalid:
                    self.content.append(subblock)
                    return True
        return False

    def isendblock(self, item):
        line = item.get_line()
        if self.__class__ is Block:
            # MAIN block does not define start/end line conditions,
            # so it should never end until all lines are read.
            # However, sometimes F77 programs lack the PROGRAM statement,
            # and here we fix that:
            if self.reader.isfix77:
                m = self.end_re(line)
                if m:
                    message = self.reader.format_message(\
                        'WARNING',
                        'assuming the end of undefined PROGRAM statement',
                        item.span[0],item.span[1])
                    print >> sys.stderr, message
                    l = Line('program UNDEFINED',(0,0),None,self.reader)
                    p = Program(self,l)
                    p.content.extend(self.content)
                    self.content[:] = [p]
                    return True
            return False
        cls = self.end_stmt_cls
        if cls.start_re(line):
            stmt = cls(self, item)
            if stmt.isvalid:
                self.content.append(stmt)
                return True
        return False

    def fill(self):
        """
        Fills blocks content until the end of block statement.
        """
        end_flag = self.__class__ is Block
        item = self.get_item()
        while item is not None:
            if isinstance(item, Line):
                # handle end of a block
                if self.isendblock(item):
                    end_flag = True
                    break
                elif not self.isstmt(item):
                    # put unknown item's to content.
                    self.content.append(item)
            item = self.get_item()
        if not end_flag:
            message = self.reader.format_message(\
                        'WARNING',
                        'failed to find the end of block for %s'\
                        % (self.__class__.__name__),
                        self.item.span[0],self.item.span[1])
            print >> sys.stderr, message
            sys.stderr.flush()
        return

class ProgramUnit(Block):
    """
    <main program>
    <external subprogram (function | subroutine)>
    <module>
    <block data>    
    """

class ProgramBlock(ProgramUnit):
    """
    program [name]
      <specification part>
      <execution part>
      <internal subprogram part>
    end [program [name]]
    """
    classes = {}
    
class ModuleBlock(ProgramUnit):
    """
    module <name>
      <specification part>
      <module subprogram part>
    end [module [name]]
    """
    classes = {}
    
class BlockDataBlock(ProgramUnit):
    """
    block data [name]
    end [block data [name]]
    """
    classes = {}
    
class InterfaceBlock(ProgramUnit):
    """
    abstract interface | interface [<generic-spec>]
      <interface specification>
    end interface [<generic spec>]
    """
    classes = {}

class PythonModuleBlock(ProgramUnit):
    """
    python module <name>
      ..
    end [python module [<name>]]
    """

class SubroutineBlock(ProgramUnit):
    """
    [prefix] subroutine <name> [ ( [<dummy-arg-list>] ) [<proc-language-binding-spec>]]
      <specification-part>
      <execution-part>
      <internal-subprogram part>
    end [subroutine [name]]
    """
    classes = {}
    
class FunctionBlock(ProgramUnit):
    classes = {}
    
class TypeBlock(Block):
    """
    type [[type-attr-spec-list] ::] <name> [(type-param-name-list)]
      <type-param-def-stmt>
      <private-or-sequence>
      <component-part>
      <type-bound-procedure-part>
    end type [name]
    """
    classes = {}

class StatementBlock(Block):
    """
    <start stmt-block>
      <statements>
    <end stmt-block>
    """
    classes = {}
    
class DoBlock(StatementBlock):

    begin_re = re.compile(r'do\b\s*(?P<label>\d*)', re.I).match

    def __init__(self, parent, item):
        label = self.begin_re(item.get_line()).group('label').strip()
        self.endlabel = label
        StatementBlock.__init__(self, parent, item)

    def isendblock(self, item):
        if self.endlabel:
            # Handle:
            #   do 1, i=1,n
            #   ..
            # 1 continue 
            if item.label==self.endlabel:
                # item may contain computational statemets
                self.content.append(item)
                # the same item label may be used for different block ends
                self.put_item(item)       
                return True
        else:
            return StatementBlock.isendblock(self, item)
        return False

class IfThenBlock(StatementBlock):

    pass
    
class SelectBlock(StatementBlock):

    pass

