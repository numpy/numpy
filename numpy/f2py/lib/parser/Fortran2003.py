#!/usr/bin/env python
"""
Fortran 2003 Syntax Rules.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

import re
from splitline import string_replace_map
import pattern_tools as pattern
from readfortran import FortranReaderBase

###############################################################################
############################## BASE CLASSES ###################################
###############################################################################

class NoMatchError(Exception):
    pass

class ParseError(Exception):
    pass

class Base(object):
    """ Base class for Fortran 2003 syntax rules.

    All Base classes have the following attributes:
      .string - original argument to construct a class instance, it's type
                is either str or FortranReaderBase.
      .item   - Line instance (holds label) or None.
    """
    subclasses = {}

    def __new__(cls, string, parent_cls = None):
        """
        """
        if parent_cls is None:
            parent_cls = [cls]
        elif cls not in parent_cls:
            parent_cls.append(cls)
        #print '__new__:',cls.__name__,`string`
        match = cls.__dict__.get('match', None)
        if isinstance(string, FortranReaderBase) and not issubclass(cls, BlockBase) \
               and match is not None:
            reader = string
            item = reader.get_item()
            if item is None: return
            try:
                obj = cls(item.line, parent_cls = parent_cls)
            except NoMatchError:
                obj = None
            if obj is None:
                reader.put_item(item)
                return
            obj.item = item
            return obj
        errmsg = '%s: %r' % (cls.__name__, string)
        if match is not None:
            try:
                result = cls.match(string)
            except NoMatchError, msg:
                if str(msg)==errmsg: # avoid recursion 1.
                    raise
                result = None
        else:
            result = None

        #print '__new__:result:',cls.__name__,`string,result`
        if isinstance(result, tuple):
            obj = object.__new__(cls)
            obj.string = string
            obj.item = None
            if hasattr(cls, 'init'): obj.init(*result)
            return obj
        elif isinstance(result, Base):
            return result
        elif result is None:
            for subcls in Base.subclasses.get(cls.__name__,[]):
                if subcls in parent_cls: # avoid recursion 2.
                    continue
                #print '%s:%s: %r' % (cls.__name__,subcls.__name__,string)
                try:
                    obj = subcls(string, parent_cls = parent_cls)
                except NoMatchError, msg:
                    obj = None
                if obj is not None:
                    return obj
        else:
            raise AssertionError,`result`
        raise NoMatchError,errmsg

##     def restore_reader(self):
##         self._item.reader.put_item(self._item)
##         return

    def init_list(self, *items):
        self.items = items
        return

    def tostr_list(self):
        return ', '.join(map(str,self.items))

    def torepr_list(self):
        return '%s(%s)' % (self.__class__.__name__,', '.join(map(repr,self.items)))

    def __str__(self): return self.tostr()

    def __repr__(self): return self.torepr()

    def __cmp__(self, other):
        if self is other: return 0
        if not isinstance(other, self.__class__): return -1
        return self.compare(other)

    def tofortran(self, tab='', isfix=None):
        return tab + str(self)


class BlockBase(Base):
    """
    <block-base> = [ <startcls> ]
                     [ <subcls> ]...
                     ...
                     [ <subcls> ]...
                     [ <endcls> ]
    """
    def match(startcls, subclasses, endcls, reader):
        assert isinstance(reader,FortranReaderBase),`reader`
        content = []
        if startcls is not None:
            try:
                obj = startcls(reader)
            except NoMatchError:
                obj = None
            if obj is None: return
            content.append(obj)
        if endcls is not None:
            classes = subclasses + [endcls]
        else:
            classes = subclasses[:]
        i = 0
        while 1:
            cls = classes[i]
            try:
                obj = cls(reader)
            except NoMatchError:
                obj = None
            if obj is None:
                j = i
                for cls in classes[i+1:]:
                    j += 1
                    try:
                        obj = cls(reader)
                    except NoMatchError:
                        obj = None
                    if obj is not None:
                        break
                if obj is not None:
                    i = j
            if obj is not None:
                content.append(obj)
                if endcls is not None and isinstance(obj, endcls): break
                continue
            if endcls is not None:
                item = reader.get_item()
                if item is not None:
                    reader.error('failed to parse with %s, skipping.' % ('|'.join([c.__name__ for c in classes[i:]])), item)
                    continue
                if hasattr(content[0],'name'):
                    reader.error('unexpected eof file while looking line for <%s> of %s.'\
                                 % (classes[-1].__name__.lower().replace('_','-'), content[0].name))
                else:
                    reader.error('unexpected eof file while looking line for <%s>.'\
                                 % (classes[-1].__name__.lower().replace('_','-')))
            break
        if not content: return
        if startcls is not None and endcls is not None:
            # check names of start and end statements:
            start_stmt = content[0]
            end_stmt = content[-1]
            if isinstance(end_stmt, endcls) and hasattr(end_stmt, 'name') and hasattr(start_stmt, 'name'):
                if end_stmt.name is not None:
                    if start_stmt.name != end_stmt.name:
                        end_stmt._item.reader.error('expected <%s-name> is %s but got %s. Ignoring.'\
                                                    % (end_stmt.type.lower(), start_stmt.name, end_stmt.name))
                else:
                    end_stmt.name = start_stmt.name
        return content,
    match = staticmethod(match)

    def init(self, content):
        self.content = content
        return
    def tostr(self):
        return self.tofortran()
    def torepr(self):
        return '%s(%s)' % (self.__class__.__name__,', '.join(map(repr, self.content)))

    def tofortran(self, tab='', isfix=None):
        l = []
        start = self.content[0]
        end = self.content[-1]
        extra_tab = ''
        if isinstance(end, EndStmtBase):
            extra_tab = '  '
        l.append(start.tofortran(tab=tab,isfix=isfix))
        for item in self.content[1:-1]:
            l.append(item.tofortran(tab=tab+extra_tab,isfix=isfix))
        if len(self.content)>1:
            l.append(end.tofortran(tab=tab,isfix=isfix))
        return '\n'.join(l)

##     def restore_reader(self):
##         content = self.content[:]
##         content.reverse()
##         for obj in content:
##             obj.restore_reader()
##         return

class SequenceBase(Base):
    """
    <sequence-base> = <obj>, <obj> [ , <obj> ]...
    """
    def match(separator, subcls, string):
        line, repmap = string_replace_map(string)
        if isinstance(separator, str):
            splitted = line.split(separator)
        else:
            splitted = separator[1].split(line)
            separator = separator[0]
        if len(splitted)<=1: return
        lst = []
        for p in splitted:
            lst.append(subcls(repmap(p.strip())))
        return separator, tuple(lst)
    match = staticmethod(match)
    def init(self, separator, items):
        self.separator = separator
        self.items = items
        return
    def tostr(self):
        s = self.separator
        if s==',': s = s + ' '
        elif s==' ': pass
        else: s = ' ' + s + ' '
        return s.join(map(str, self.items))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.separator, self.items)
    def compare(self, other):
        return cmp((self.separator,self.items),(other.separator,self.items))
    
class UnaryOpBase(Base):
    """
    <unary-op-base> = <unary-op> <rhs>
    """
    def init(self, op, rhs):
        self.op = op
        self.rhs = rhs
        return    
    def tostr(self):
        return '%s %s' % (self.op, self.rhs)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__,self.op, self.rhs)
    def match(op_pattern, rhs_cls, string):
        m = op_pattern.match(string)
        if not m: return
        #if not m: return rhs_cls(string)
        rhs = string[m.end():].lstrip()        
        if not rhs: return
        op = string[:m.end()].rstrip().upper()
        return op, rhs_cls(rhs)
    match = staticmethod(match)
    def compare(self, other):
        return cmp((self.op,self.rhs),(other.op,other.rhs))

class BinaryOpBase(Base):
    """
    <binary-op-base> = <lhs> <op> <rhs>
    <op> is searched from right by default.
    """
    def match(lhs_cls, op_pattern, rhs_cls, string, right=True):
        line, repmap = string_replace_map(string)
        if isinstance(op_pattern, str):
            if right:
                t = line.rsplit(op_pattern,1)
            else:
                t = line.split(op_pattern,1)
            if len(t)!=2: return
            lhs, rhs = t[0].rstrip(), t[1].lstrip()
            op = op_pattern
        else:
            if right:
                t = op_pattern.rsplit(line)
            else:
                t = op_pattern.lsplit(line)
            if t is None or len(t)!=3: return
            lhs, op, rhs = t
            lhs = lhs.rstrip()
            rhs = rhs.lstrip()
            op = op.upper()
        if not lhs: return
        if not rhs: return
        lhs_obj = lhs_cls(repmap(lhs))
        rhs_obj = rhs_cls(repmap(rhs))
        return lhs_obj, op, rhs_obj
    match = staticmethod(match)
    def init(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        return    
    def tostr(self):
        return '%s %s %s' % (self.lhs, self.op, self.rhs)
    def torepr(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,self.lhs, self.op, self.rhs)
    def compare(self, other):
        return cmp((self.op,self.lhs,self.rhs),(other.op,other.lhs,other.rhs))

class SeparatorBase(Base):
    """
    <separator-base> = [ <lhs> ] : [ <rhs> ]
    """
    def match(lhs_cls, rhs_cls, string, require_lhs=False, require_rhs=False):
        line, repmap = string_replace_map(string)
        if ':' not in line: return
        lhs,rhs = line.split(':',1)
        lhs = lhs.rstrip()
        rhs = rhs.lstrip()
        lhs_obj, rhs_obj = None, None
        if lhs:
            if lhs_cls is None: return
            lhs_obj = lhs_cls(repmap(lhs))
        elif require_lhs:
            return
        if rhs:
            if rhs_cls is None: return
            rhs_obj = rhs_cls(repmap(rhs))
        elif require_rhs:
            return
        return lhs_obj, rhs_obj
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        s = ''
        if self.items[0] is not None:
            s += '%s :' % (self.items[0])
        else:
            s += ':'
        if self.items[1] is not None:
            s += ' %s' % (self.items[1])
        return s
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class KeywordValueBase(Base):
    """
    <keyword-value-base> = [ <lhs> = ] <rhs>
    """
    def match(lhs_cls, rhs_cls, string, require_lhs = True, upper_lhs = False):
        if require_lhs and '=' not in string: return
        if isinstance(lhs_cls, (list, tuple)):
            for s in lhs_cls:
                try:
                    obj = KeywordValueBase.match(s, rhs_cls, string, require_lhs=require_lhs, upper_lhs=upper_lhs)
                except NoMatchError:
                    obj = None
                if obj is not None: return obj
            return obj
        lhs,rhs = string.split('=',1)
        lhs = lhs.rstrip()
        rhs = rhs.lstrip()
        if not rhs: return
        if not lhs:
            if require_lhs: return
            return None, rhs_cls(rhs)
        if isinstance(lhs_cls, str):
            if upper_lhs:
                lhs = lhs.upper()
            if lhs_cls!=lhs: return
            return lhs, rhs_cls(rhs)
        return lhs_cls(lhs),rhs_cls(rhs)
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        if self.items[0] is None: return str(self.items[1])
        return '%s = %s' % tuple(self.items)
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class BracketBase(Base):
    """
    <bracket-base> = <left-bracket-base> <something> <right-bracket>
    """
    def match(brackets, cls, string):
        i = len(brackets)/2
        left = brackets[:i]
        right = brackets[-i:]
        if string.startswith(left) and string.endswith(right):
            line = string[i:-i].strip()
            if not line: return
            return left,cls(line),right
        return
    match = staticmethod(match)
    def init(self,left,item,right):
        self.left = left
        self.item = item
        self.right = right
        return
    def tostr(self):
        if self.item is None:
            return '%s%s' % (self.left, self.right)
        return '%s%s%s' % (self.left, self.item, self.right)
    def torepr(self): return '%s(%r, %r, %r)' % (self.__class__.__name__, self.left, self.item, self.right)
    def compare(self, other):
        return cmp((self.left,self.item,self.right),(other.left,other.item,other.right))

class NumberBase(Base):
    """
    <number-base> = <number> [ _ <kind-param> ]
    """
    def match(number_pattern, string):
        m = number_pattern.match(string)
        if m is None: return
        return m.group('value').upper(),m.group('kind_param')
    match = staticmethod(match)
    def init(self, value, kind_param):
        self.value = value
        self.kind_param = kind_param
        return
    def tostr(self):
        if self.kind_param is None: return str(self.value)
        return '%s_%s' % (self.value, self.kind_param)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.kind_param)
    def compare(self, other):
        return cmp(self.value, other.value)

class CallBase(Base):
    """
    <call-base> = <lhs> ( [ <rhs> ] )
    """
    def match(lhs_cls, rhs_cls, string, upper_lhs = False, require_rhs=False):
        if not string.endswith(')'): return
        line, repmap = string_replace_map(string)
        i = line.find('(')
        if i==-1: return
        lhs = line[:i].rstrip()
        if not lhs: return
        rhs = line[i+1:-1].strip()
        lhs = repmap(lhs)
        if upper_lhs:
            lhs = lhs.upper()
        rhs = repmap(rhs)
        if isinstance(lhs_cls, str):
            if lhs_cls!=lhs: return
        else:
            lhs = lhs_cls(lhs)
        if rhs:
            if isinstance(rhs_cls, str):
                if rhs_cls!=rhs: return
            else:
                rhs = rhs_cls(rhs)
            return lhs, rhs
        elif require_rhs:
            return
        return lhs, None
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        if self.items[1] is None: return '%s()' % (self.items[0])
        return '%s(%s)' % (self.items[0], self.items[1])
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))
    def compare(self, other):
        return cmp(self.items,other.items)

class CALLBase(CallBase):
    """
    <CALL-base> = <LHS> ( [ <rhs> ] )
    """
    def match(lhs_cls, rhs_cls, string, require_rhs = False):
        return CallBase.match(lhs_cls, rhs_cls, string, upper_lhs=True, require_rhs = require_rhs)
    match = staticmethod(match)

class StringBase(Base):
    """
    <string-base> = <xyz>
    """
    def match(pattern, string):
        if isinstance(pattern, (list,tuple)):
            for p in pattern:
                obj = StringBase.match(p, string)
                if obj is not None: return obj
            return
        if isinstance(pattern, str):
            if len(pattern)==len(string) and pattern==string: return string,
            return
        if pattern.match(string): return string,
        return
    match = staticmethod(match)
    def init(self, string):
        self.string = string
        return
    def tostr(self): return str(self.string)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.string)
    def compare(self, other):
        return cmp(self.string,other.string)

class STRINGBase(StringBase):
    """
    <STRING-base> = <XYZ>
    """
    match = staticmethod(StringBase.match)
    def match(pattern, string):
        if isinstance(pattern, (list,tuple)):
            for p in pattern:
                obj = STRINGBase.match(p, string)
                if obj is not None: return obj
            return
        STRING = string.upper()
        if isinstance(pattern, str):
            if len(pattern)==len(string) and pattern==STRING: return STRING,
            return
        if pattern.match(STRING): return STRING,
        return
    match = staticmethod(match)

class StmtBase(Base):
    """
    [ <label> ] <stmt>
    """
    def tofortran(self, tab='', isfix=None):
        label = None
        if self.item is not None: label = self.item.label
        if isfix:
            colon = ''
            c = ' '
        else:
            colon = ':'
            c = ''
        if label:
            t = c + label + colon
            if isfix:
                while len(t)<6: t += ' '
            else:
                tab = tab[len(t):] or ' '
        else:
            t = ''
        return t + tab + str(self)

class EndStmtBase(StmtBase):
    """
    <end-stmt-base> = END [ <stmt> [ <stmt-name>] ]
    """
    def match(stmt_type, stmt_name, string, require_stmt_type=False):
        start = string[:3].upper()
        if start != 'END': return
        line = string[3:].lstrip()
        start = line[:len(stmt_type)].upper()
        if start:
            if start.replace(' ','') != stmt_type.replace(' ',''): return
            line = line[len(stmt_type):].lstrip()
        else:
            if require_stmt_type: return
            line = ''
        if line:
            if stmt_name is None: return
            return stmt_type, stmt_name(line)
        return stmt_type, None
    match = staticmethod(match)
    def init(self, stmt_type, stmt_name):
        self.type, self.name = stmt_type, stmt_name
        return
    def tostr(self):
        if self.name is not None:
            return 'END %s %s' % (self.type, self.name)
        return 'END %s' % (self.type)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.type, self.name)
    def compare(self, other):
        return cmp((self.type,self.name),(other.type,other.name))

def _ensure_single_spaces(string):
    return ' '.join([s for s in string.split(' ') if s])

class WORDClsBase(Base):
    """
    <WORD-cls> = <WORD> [ [ :: ] <cls> ]
    """
    def match(pattern, cls, string, check_colons=False, require_cls=False):
        if isinstance(pattern, (tuple,list)):
            for p in pattern:
                try:
                    obj = WORDClsBase.match(p, cls, string, check_colons=check_colons, require_cls=require_cls)
                except NoMatchError:
                    obj = None
                if obj is not None: return obj
            return
        if isinstance(pattern, str):
            if string[:len(pattern)].upper()!=pattern: return
            line = string[len(pattern):]
            if not line: return pattern, None
            if 'a'<=line[0].lower()<='z' or line[0]=='_': return
            line = line.lstrip()
            if not line:
                if require_cls: return
                return pattern, None
            if cls is None: return
            return pattern, cls(line)
        m = pattern.match(string)
        if m is None: return
        line = string[len(m.group()):]
        if pattern.value is not None:
            pattern_value = pattern.value
        else:
            pattern_value = m.group().upper()
        if not line: return pattern_value, None
        if 'a'<=line[0].lower()<='z' or line[0]=='_': return
        line = line.lstrip()
        if not line:
            if require_cls: return
            return pattern_value, None
        if cls is None: return
        return pattern_value, cls(line)
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        if self.items[1] is None: return str(self.items[0])
        s = str(self.items[1])
        if s and s[0] in '(*':
            return '%s%s' % (self.items[0], s)
        return '%s %s' % (self.items[0], s)
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

###############################################################################
############################### SECTION  1 ####################################
###############################################################################

#R101: <xyz-list> = <xyz> [ , <xyz> ]...
#R102: <xyz-name> = <name>
#R103: <scalar-xyz> = <xyz>

###############################################################################
############################### SECTION  2 ####################################
###############################################################################

class Program(BlockBase): # R201
    """
    <program> = <program-unit>
                  [ <program-unit> ] ...
    """
    subclass_names = []
    use_names = ['Program_Unit']
    def match(reader):
        return BlockBase.match(Program_Unit, [Program_Unit], None, reader)
    match = staticmethod(match)

class Program_Unit(Base): # R202
    """
    <program-unit> = <main-program>
                     | <external-subprogram>
                     | <module>
                     | <block-data>
    """
    subclass_names = ['Main_Program', 'External_Subprogram', 'Module', 'Block_Data']

class External_Subprogram(Base): # R203
    """
    <external-subprogram> = <function-subprogram>
                            | <subroutine-subprogram>
    """
    subclass_names = ['Function_Subprogram', 'Subroutine_Subprogram']


class Specification_Part(BlockBase): # R204
    """
    <specification-part> = [ <use-stmt> ]...
                             [ <import-stmt> ]...
                             [ <implicit-part> ]
                             [ <declaration-construct> ]...
    """
    subclass_names = []
    use_names = ['Use_Stmt', 'Import_Stmt', 'Implicit_Part', 'Declaration_Construct']
    def match(reader):
        return BlockBase.match(None, [Use_Stmt, Import_Stmt, Implicit_Part, Declaration_Construct], None, reader)
    match = staticmethod(match)

class Implicit_Part(Base): # R205
    """
    <implicit-part> = [ <implicit-part-stmt> ]...
                        <implicit-stmt>
    """
    subclass_names = []
    use_names = ['Implicit_Part_Stmt', 'Implicit_Stmt']

class Implicit_Part_Stmt(Base): # R206
    """
    <implicit-part-stmt> = <implicit-stmt>
                           | <parameter-stmt>
                           | <format-stmt>
                           | <entry-stmt>
    """
    subclass_names = ['Implicit_Stmt', 'Parameter_Stmt', 'Format_Stmt', 'Entry_Stmt']

class Declaration_Construct(Base): # R207
    """
    <declaration-construct> = <derived-type-def>
                              | <entry-stmt>
                              | <enum-def>
                              | <format-stmt>
                              | <interface-block>
                              | <parameter-stmt>
                              | <procedure-declaration-stmt>
                              | <specification-stmt>
                              | <type-declaration-stmt>
                              | <stmt-function-stmt>
    """
    subclass_names = ['Derived_Type_Def', 'Entry_Stmt', 'Enum_Def', 'Format_Stmt',
                      'Interface_Block', 'Parameter_Stmt', 'Procedure_Declaration_Stmt',
                      'Specification_Stmt', 'Type_Declaration_Stmt', 'Stmt_Function_Stmt']

class Execution_Part(BlockBase): # R208
    """
    <execution-part> = <executable-construct>
                       | [ <execution-part-construct> ]...

    <execution-part> shall not contain <end-function-stmt>, <end-program-stmt>, <end-subroutine-stmt>
    """
    subclass_names = []
    use_names = ['Executable_Construct_C201', 'Execution_Part_Construct_C201']
    def match(string): return BlockBase.match(Executable_Construct_C201, [Execution_Part_Construct_C201], None, string)
    match = staticmethod(match)

class Execution_Part_Construct(Base): # R209
    """
    <execution-part-construct> = <executable-construct>
                                 | <format-stmt>
                                 | <entry-stmt>
                                 | <data-stmt>
    """
    subclass_names = ['Executable_Construct', 'Format_Stmt', 'Entry_Stmt', 'Data_Stmt']

class Execution_Part_Construct_C201(Base):
    subclass_names = ['Executable_Construct_C201', 'Format_Stmt', 'Entry_Stmt', 'Data_Stmt']

class Internal_Subprogram_Part(Base): # R210
    """
    <internal-subprogram-part> = <contains-stmt>
                                   <internal-subprogram>
                                   [ <internal-subprogram> ]...
    """
    subclass_names = []
    use_names = ['Contains_Stmt', 'Internal_Subprogram']

class Internal_Subprogram(Base): # R211
    """
    <internal-subprogram> = <function-subprogram>
                            | <subroutine-subprogram>
    """
    subclass_names = ['Function_Subprogram', 'Subroutine_Subprogram']

class Specification_Stmt(Base):# R212
    """
    <specification-stmt> = <access-stmt>
                           | <allocatable-stmt>
                           | <asynchronous-stmt>
                           | <bind-stmt>
                           | <common-stmt>
                           | <data-stmt>
                           | <dimension-stmt>
                           | <equivalence-stmt>
                           | <external-stmt>
                           | <intent-stmt>
                           | <intrinsic-stmt>
                           | <namelist-stmt>
                           | <optional-stmt>
                           | <pointer-stmt>
                           | <protected-stmt>
                           | <save-stmt>
                           | <target-stmt>
                           | <volatile-stmt>
                           | <value-stmt>
    """
    subclass_names = ['Access_Stmt', 'Allocatable_Stmt', 'Asynchronous_Stmt','Bind_Stmt',
                      'Common_Stmt', 'Data_Stmt', 'Dimension_Stmt', 'Equivalence_Stmt',
                      'External_Stmt', 'Intent_Stmt', 'Intrinsic_Stmt', 'Namelist_Stmt',
                      'Optional_Stmt','Pointer_Stmt','Protected_Stmt','Save_Stmt',
                      'Target_Stmt','Volatile_Stmt', 'Value_Stmt']

class Executable_Construct(Base):# R213
    """
    <executable-construct> = <action-stmt>
                             | <associate-stmt>
                             | <case-construct>
                             | <do-construct>
                             | <forall-construct>
                             | <if-construct>
                             | <select-type-construct>
                             | <where-construct>
    """
    subclass_names = ['Action_Stmt', 'Associate_Stmt', 'Case_Construct', 'Do_Construct',
                      'Forall_Construct', 'If_Construct', 'Select_Type_Construct', 'Where_Construct']

class Executable_Construct_C201(Base):
    subclass_names = Executable_Construct.subclass_names[:]
    subclass_names[subclass_names.index('Action_Stmt')] = 'Action_Stmt_C201'
    

class Action_Stmt(Base):# R214
    """
    <action-stmt> = <allocate-stmt>
                    | <assignment-stmt>
                    | <backspace-stmt>
                    | <call-stmt>
                    | <close-stmt>
                    | <continue-stmt>
                    | <cycle-stmt>
                    | <deallocate-stmt>
                    | <endfile-stmt>
                    | <end-function-stmt>
                    | <end-program-stmt>
                    | <end-subroutine-stmt>
                    | <exit-stmt>
                    | <flush-stmt>
                    | <forall-stmt>
                    | <goto-stmt>
                    | <if-stmt>
                    | <inquire-stmt>
                    | <nullify-stmt>
                    | <open-stmt>
                    | <pointer-assignment-stmt>
                    | <print-stmt>
                    | <read-stmt>
                    | <return-stmt>
                    | <rewind-stmt>
                    | <stop-stmt>
                    | <wait-stmt>
                    | <where-stmt>
                    | <write-stmt>
                    | <arithmetic-if-stmt>
                    | <computed-goto-stmt>
    """
    subclass_names = ['Allocate_Stmt', 'Assignment_Stmt', 'Backspace_Stmt', 'Call_Stmt',
                      'Close_Stmt', 'Continue_Stmt', 'Cycle_Stmt', 'Deallocate_Stmt',
                      'Endfile_Stmt', 'End_Function_Stmt', 'End_Subroutine_Stmt', 'Exit_Stmt',
                      'Flush_Stmt', 'Forall_Stmt', 'Goto_Stmt', 'If_Stmt', 'Inquire_Stmt',
                      'Nullify_Stmt', 'Open_Stmt', 'Pointer_Assignment_Stmt', 'Print_Stmt',
                      'Read_Stmt', 'Return_Stmt', 'Rewind_Stmt', 'Stop_Stmt', 'Wait_Stmt',
                      'Where_Stmt', 'Write_Stmt', 'Arithmetic_If_Stmt', 'Computed_Goto_Stmt']

class Action_Stmt_C201(Base):
    """
    <action-stmt-c201> = <action-stmt>
    C201 is applied.
    """
    subclass_names = Action_Stmt.subclass_names[:]
    subclass_names.remove('End_Function_Stmt')
    subclass_names.remove('End_Subroutine_Stmt')
    #subclass_names.remove('End_Program_Stmt')

class Action_Stmt_C802(Base):
    """
    <action-stmt-c802> = <action-stmt>
    C802 is applied.
    """
    subclass_names = Action_Stmt.subclass_names[:]
    subclass_names.remove('End_Function_Stmt')
    subclass_names.remove('End_Subroutine_Stmt')
    subclass_names.remove('If_Stmt')

class Action_Stmt_C824(Base):
    """
    <action-stmt-c824> = <action-stmt>
    C824 is applied.
    """
    subclass_names = Action_Stmt.subclass_names[:]
    subclass_names.remove('End_Function_Stmt')
    subclass_names.remove('End_Subroutine_Stmt')
    subclass_names.remove('Continue_Stmt')
    subclass_names.remove('Goto_Stmt')
    subclass_names.remove('Return_Stmt')
    subclass_names.remove('Stop_Stmt')
    subclass_names.remove('Exit_Stmt')
    subclass_names.remove('Cycle_Stmt')
    subclass_names.remove('Arithmetic_If_Stmt')

class Keyword(Base): # R215
    """
    <keyword> = <name>
    """
    subclass_names = ['Name']

###############################################################################
############################### SECTION  3 ####################################
###############################################################################

#R301: <character> = <alphanumeric-character> | <special-character>
#R302: <alphanumeric-character> = <letter> | <digit> | <underscore>
#R303: <underscore> = _

class Name(StringBase): # R304
    """
    <name> = <letter> [ <alphanumeric_character> ]...
    """
    subclass_names = []
    def match(string): return StringBase.match(pattern.abs_name, string)
    match = staticmethod(match)

class Constant(Base): # R305
    """
    <constant> = <literal-constant>
                 | <named-constant>
    """
    subclass_names = ['Literal_Constant','Named_Constant']

class Literal_Constant(Base): # R306
    """
    <literal-constant> = <int-literal-constant>
                         | <real-literal-constant>
                         | <complex-literal-constant>
                         | <logical-literal-constant>
                         | <char-literal-constant>
                         | <boz-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant', 'Real_Literal_Constant','Complex_Literal_Constant',
                      'Logical_Literal_Constant','Char_Literal_Constant','Boz_Literal_Constant']

class Named_Constant(Base): # R307
    """
    <named-constant> = <name>
    """
    subclass_names = ['Name']

class Int_Constant(Base): # R308
    """
    <int-constant> = <constant>
    """
    subclass_names = ['Constant']

class Char_Constant(Base): # R309
    """
    <char-constant> = <constant>
    """
    subclass_names = ['Constant']

#R310: <intrinsic-operator> = <power-op> | <mult-op> | <add-op> | <concat-op> | <rel-op> | <not-op> | <and-op> | <or-op> | <equiv-op>
#R311: <defined-operator> = <defined-unary-op> | <defined-binary-op> | <extended-intrinsic-op>
#R312: <extended-intrinsic-op> = <intrinsic-op>

class Label(StringBase): # R313
    """
    <label> = <digit> [ <digit> [ <digit> [ <digit> [ <digit> ] ] ] ]
    """
    subclass_names = []
    def match(string): return StringBase.match(pattern.abs_label, string)
    match = staticmethod(match)

###############################################################################
############################### SECTION  4 ####################################
###############################################################################

class Type_Spec(Base): # R401
    """
    <type-spec> = <intrinsic-type-spec>
                  | <derived-type-spec>
    """
    subclass_names = ['Intrinsic_Type_Spec', 'Derived_Type_Spec']

class Type_Param_Value(StringBase): # R402
    """
    <type-param-value> = <scalar-int-expr>
                       | *
                       | :
    """
    subclass_names = ['Scalar_Int_Expr']
    use_names = []
    def match(string): return StringBase.match(['*',':'], string)
    match = staticmethod(match)

class Intrinsic_Type_Spec(WORDClsBase): # R403
    """
    <intrinsic-type-spec> = INTEGER [ <kind-selector> ]
                            | REAL [ <kind-selector> ]
                            | DOUBLE COMPLEX
                            | COMPLEX [ <kind-selector> ]
                            | CHARACTER [ <char-selector> ]
                            | LOGICAL [ <kind-selector> ]
    Extensions:
                            | DOUBLE PRECISION
                            | BYTE
    """
    subclass_names = []
    use_names = ['Kind_Selector','Char_Selector']
        
    def match(string):
        for w,cls in [('INTEGER',Kind_Selector),
                      ('REAL',Kind_Selector),
                      ('COMPLEX',Kind_Selector),
                      ('LOGICAL',Kind_Selector),
                      ('CHARACTER',Char_Selector),
                      (pattern.abs_double_complex_name, None),
                      (pattern.abs_double_precision_name, None),
                      ('BYTE', None),
                      ]:
            try:
                obj = WORDClsBase.match(w,cls,string)
            except NoMatchError:
                obj = None
            if obj is not None: return obj
        return
    match = staticmethod(match)


class Kind_Selector(Base): # R404
    """
    <kind-selector> = ( [ KIND = ] <scalar-int-initialization-expr> )
    Extensions:
                      | * <char-length>
    """
    subclass_names = []
    use_names = ['Char_Length','Scalar_Int_Initialization_Expr']

    def match(string):
        if string[0]+string[-1] != '()':
            if not string.startswith('*'): return
            return '*',Char_Length(string[1:].lstrip())
        line = string[1:-1].strip()
        if line[:4].upper()=='KIND':
            line = line[4:].lstrip()
            if not line.startswith('='): return
            line = line[1:].lstrip()
        return '(',Scalar_Int_Initialization_Expr(line),')'
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if len(self.items)==2: return '%s%s' % tuple(self.items)
        return '%sKIND = %s%s' % tuple(self.items)
    
    def torepr(self):
        if len(self.items)==2:
            return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.items[0], self.items[1], self.items[2])

class Signed_Int_Literal_Constant(NumberBase): # R405
    """
    <signed-int-literal-constant> = [ <sign> ] <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant'] # never used because sign is included in pattern
    def match(string):
        return NumberBase.match(pattern.abs_signed_int_literal_constant_named, string)
    match = staticmethod(match)

class Int_Literal_Constant(NumberBase): # R406
    """
    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    """
    subclass_names = []
    def match(string):
        return NumberBase.match(pattern.abs_int_literal_constant_named, string)
    match = staticmethod(match)

#R407: <kind-param> = <digit-string> | <scalar-int-constant-name>
#R408: <signed-digit-string> = [ <sign> ] <digit-string>
#R409: <digit-string> = <digit> [ <digit> ]...
#R410: <sign> = + | -

class Boz_Literal_Constant(Base): # R411
    """
    <boz-literal-constant> = <binary-constant>
                             | <octal-constant>
                             | <hex-constant>
    """
    subclass_names = ['Binary_Constant','Octal_Constant','Hex_Constant']

class Binary_Constant(STRINGBase): # R412
    """
    <binary-constant> = B ' <digit> [ <digit> ]... '
                        | B \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string): return STRINGBase.match(pattern.abs_binary_constant, string)
    match = staticmethod(match)

class Octal_Constant(STRINGBase): # R413
    """
    <octal-constant> = O ' <digit> [ <digit> ]... '
                       | O \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string): return STRINGBase.match(pattern.abs_octal_constant, string)
    match = staticmethod(match)

class Hex_Constant(STRINGBase): # R414
    """
    <hex-constant> = Z ' <digit> [ <digit> ]... '
                     | Z \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string): return STRINGBase.match(pattern.abs_hex_constant, string)
    match = staticmethod(match)

#R415: <hex-digit> = <digit> | A | B | C | D | E | F

class Signed_Real_Literal_Constant(NumberBase): # R416
    """
    <signed-real-literal-constant> = [ <sign> ] <real-literal-constant>
    """
    subclass_names = ['Real_Literal_Constant'] # never used
    def match(string):
        return NumberBase.match(pattern.abs_signed_real_literal_constant_named, string)
    match = staticmethod(match)

class Real_Literal_Constant(NumberBase): # R417
    """
    """
    subclass_names = []
    def match(string):
        return NumberBase.match(pattern.abs_real_literal_constant_named, string)
    match = staticmethod(match)

#R418: <significand> = <digit-string> . [ <digit-string> ]  | . <digit-string>
#R419: <exponent-letter> = E | D
#R420: <exponent> = <signed-digit-string>

class Complex_Literal_Constant(Base): # R421
    """
    <complex-literal-constant> = ( <real-part>, <imag-part> )
    """
    subclass_names = []
    use_names = ['Real_Part','Imag_Part']
    def match(string):
        if not string or string[0]+string[-1]!='()': return
        if not pattern.abs_complex_literal_constant.match(string):
            return
        r,i = string[1:-1].split(',')
        return Real_Part(r.strip()), Imag_Part(i.strip())
    match = staticmethod(match)
    def init(self,real,imag):
        self.real, self.imag = real, imag
        return
    def tostr(self): return '(%s, %s)' % (self.real, self.imag)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.real, self.imag)

class Real_Part(Base): # R422
    """
    <real-part> = <signed-int-literal-constant>
                  | <signed-real-literal-constant>
                  | <named-constant>
    """
    subclass_names = ['Signed_Int_Literal_Constant','Signed_Real_Literal_Constant','Named_Constant']

class Imag_Part(Base): # R423
    """
    <imag-part> = <real-part>
    """
    subclass_names = ['Signed_Int_Literal_Constant','Signed_Real_Literal_Constant','Named_Constant']

class Char_Selector(Base): # R424
    """
    <char-selector> = <length-selector>
                      | ( LEN = <type-param-value> , KIND = <scalar-int-initialization-expr> )
                      | ( <type-param-value> , [ KIND = ] <scalar-int-initialization-expr> )
                      | ( KIND = <scalar-int-initialization-expr> [ , LEN = <type-param-value> ] )
    """
    subclass_names = ['Length_Selector']
    use_names = ['Type_Param_Value','Scalar_Int_Initialization_Expr']
    def match(string):
        if string[0]+string[-1] != '()': return
        line, repmap = string_replace_map(string[1:-1].strip())
        if line[:3].upper()=='LEN':
            line = line[3:].lstrip()
            if not line.startswith('='): return
            line = line[1:].lstrip()
            i = line.find(',')
            if i==-1: return
            v = line[:i].rstrip()
            line = line[i+1:].lstrip()
            if line[:4].upper()!='KIND': return
            line = line[4:].lstrip()
            if not line.startswith('='): return
            line = line[1:].lstrip()
            v = repmap(v)
            line = repmap(line)
            return Type_Param_Value(v), Scalar_Int_Initialization_Expr(line)
        elif line[:4].upper()=='KIND':
            line = line[4:].lstrip()
            if not line.startswith('='): return
            line = line[1:].lstrip()
            i = line.find(',')
            if i==-1: return None,Scalar_Int_Initialization_Expr(line)
            v = line[i+1:].lstrip()
            line = line[:i].rstrip()
            if v[:3].upper()!='LEN': return
            v = v[3:].lstrip()
            if not v.startswith('='): return
            v = v[1:].lstrip()
            return Type_Param_Value(v), Scalar_Int_Initialization_Expr(line)
        else:
            i = line.find(',')
            if i==-1: return
            v = line[:i].rstrip()
            line = line[i+1:].lstrip()
            if line[:4].upper()=='KIND':
                line = line[4:].lstrip()
                if not line.startswith('='): return
                line = line[1:].lstrip()
            return Type_Param_Value(v), Scalar_Int_Initialization_Expr(line)
        return
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[0] is None:
            return '(KIND = %s)' % (self.items[1])
        return '(LEN = %s, KIND = %s)' % (self.items[0],self.items[1])
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Length_Selector(Base): # R425
    """
    <length -selector> = ( [ LEN = ] <type-param-value> )
                        | * <char-length> [ , ]
    """
    subclass_names = []
    use_names = ['Type_Param_Value','Char_Length']
    def match(string):
        if string[0]+string[-1] == '()':
            line = string[1:-1].strip()
            if line[:3].upper()=='LEN':
                line = line[3:].lstrip()
                if not line.startswith('='): return
                line = line[1:].lstrip()
            return '(',Type_Param_Value(line),')'
        if not string.startswith('*'): return
        line = string[1:].lstrip()
        if string[-1]==',': line = line[:-1].rstrip()
        return '*',Char_Length(line)
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if len(self.items)==2: return '%s%s' % tuple(self.items)
        return '%sLEN = %s%s' % tuple(self.items)
    def torepr(self):
        if len(self.items)==2:
            return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.items[0],self.items[1],self.items[2])

class Char_Length(BracketBase): # R426
    """
    <char-length> = ( <type-param-value> )
                    | <scalar-int-literal-constant>
    """
    subclass_names = ['Scalar_Int_Literal_Constant']
    use_names = ['Type_Param_Value']
    def match(string): return BracketBase.match('()',Type_Param_Value, string)
    match = staticmethod(match)

class Char_Literal_Constant(Base): # R427
    """
    <char-literal-constant> = [ <kind-param> _ ] ' <rep-char> '
                              | [ <kind-param> _ ] \" <rep-char> \"
    """
    subclass_names = []
    rep = pattern.char_literal_constant
    def match(string):
        if string[-1] not in '"\'': return
        if string[-1]=='"':
            abs_a_n_char_literal_constant_named = pattern.abs_a_n_char_literal_constant_named2
        else:
            abs_a_n_char_literal_constant_named = pattern.abs_a_n_char_literal_constant_named1
        line, repmap = string_replace_map(string)
        m = abs_a_n_char_literal_constant_named.match(line)
        if not m: return
        kind_param = m.group('kind_param')
        line = m.group('value')
        line = repmap(line)
        return line, kind_param
    match = staticmethod(match)
    def init(self, value, kind_param):
        self.value = value
        self.kind_param = kind_param
        return
    def tostr(self):
        if self.kind_param is None: return str(self.value)
        return '%s_%s' % (self.kind_param, self.value)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.kind_param)

class Logical_Literal_Constant(NumberBase): # R428
    """
    <logical-literal-constant> = .TRUE. [ _ <kind-param> ]
                                 | .FALSE. [ _ <kind-param> ]
    """
    subclass_names = []
    def match(string):
        return NumberBase.match(pattern.abs_logical_literal_constant_named, string)    
    match = staticmethod(match)

class Derived_Type_Def(Base): # R429
    """
    <derived-type-def> = <derived-type-stmt>
                           [ <type-param-def-stmt> ]...
                           [ <private-or-sequence> ]...
                           [ <component-part> ]
                           [ <type-bound-procedure-part> ]
                           <end-type-stmt>
    """
    subclass_names = []
    use_names = ['Derived_Type_Stmt', 'Type_Param_Def_Stmt', 'Private_Or_Sequence',
                 'Component_Part', 'Type_Bound_Procedure_Part', 'End_Type_Stmt']

class Derived_Type_Stmt(StmtBase): # R430
    """
    <derived-type-stmt> = TYPE [ [ , <type-attr-spec-list> ] :: ] <type-name> [ ( <type-param-name-list> ) ]
    """
    subclass_names = []
    use_names = ['Type_Attr_Spec_List', 'Type_Name', 'Type_Param_Name_List']
    def match(string):
        if string[:4].upper()!='TYPE': return
        line = string[4:].lstrip()
        i = line.find('::')
        attr_specs = None
        if i!=-1:
            if line.startswith(','):
                l = line[1:i].strip()
                if not l: return
                attr_specs = Type_Attr_Spec_List(l)
            line = line[i+2:].lstrip()
        m = pattern.name.match(line)
        if m is None: return
        name = Type_Name(m.group())
        line = line[m.end():].lstrip()
        if not line: return attr_specs, name, None
        if line[0]+line[-1]!='()': return
        return attr_specs, name, Type_Param_Name_List(line[1:-1].strip())
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
    def tostr(self):
        s = 'TYPE'
        if self.items[0] is not None:
            s += ', %s :: %s' % (self.items[0], self.items[1])
        else:
            s += ' :: %s' % (self.items[1])
        if self.items[2] is not None:
            s += '(%s)' % (self.items[2])
        return s
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class Type_Name(Name): # C424
    """
    <type-name> = <name>
    <type-name> shall not be DOUBLEPRECISION or the name of intrinsic type
    """
    subclass_names = []
    use_names = []
    def match(string):
        if pattern.abs_intrinsic_type_name.match(string): return
        return Name.match(string)
    match = staticmethod(match)

class Type_Attr_Spec(Base): # R431
    """
    <type-attr-spec> = <access-spec>
                       | EXTENDS ( <parent-type-name> )
                       | ABSTRACT
                       | BIND (C)
    """
    subclass_names = ['Access_Spec']
    use_names = ['Parent_Type_Name']
    def match(string):
        if len(string)==8 and string.upper()=='ABSTRACT': return 'ABSTRACT',None
        if string[:4].upper()=='BIND':
            line = string[4:].lstrip()
            if not line: return
            if line[0]+line[-1]!='()': return
            if line[1:-1].strip().upper()=='C': return 'BIND','C'
            return
        if string[:7].upper()=='EXTENDS':
            line = string[7:].lstrip()
            if not line: return
            if line[0]+line[-1]!='()': return
            line = line[1:-1].strip()
            if not line: return
            return 'EXTENDS', Parent_Type_Name(line)
    match = staticmethod(match)
    def init(self,*args):
        self.items = args
    def tostr(self):
        if self.items[1] is None: return str(self.items[0])
        return '%s(%s)' % tuple(self.items)
    def torepr(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class Private_Or_Sequence(Base): # R432
    """
    <private-or-sequence> = <private-components-stmt>
                            | <sequence-stmt>
    """
    subclass_names = ['Private_Components_Stmt', 'Sequence_Stmt']

class End_Type_Stmt(EndStmtBase): # R433
    """
    <end-type-stmt> = END TYPE [ <type-name> ]
    """
    subclass_names = []
    use_names = ['Type_Name']
    def match(string): return EndStmtBase.match('TYPE',Type_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Sequence_Stmt(STRINGBase): # R434
    """
    <sequence-stmt> = SEQUENCE
    """
    subclass_names = []
    def match(string): return STRINGBase.match('SEQUENCE', string)
    match = staticmethod(match)

class Type_Param_Def_Stmt(StmtBase): # R435
    """
    <type-param-def-stmt> = INTEGER [ <kind-selector> ] , <type-param-attr-spec> :: <type-param-decl-list>
    """
    subclass_names = []
    use_names = ['Kind_Selector', 'Type_Param_Attr_Spec', 'Type_Param_Decl_List']
    def match(string):
        if string[:7].upper()!='INTEGER': return
        line, repmap = string_replace_map(string[7:].lstrip())
        if not line: return
        i = line.find(',')
        if i==-1: return
        kind_selector = repmap(line[:i].rstrip()) or None
        line = repmap(line[i+1:].lstrip())
        i = line.find('::')
        if i==-1: return
        l1 = line[:i].rstrip()
        l2 = line[i+2:].lstrip()
        if not l1 or not l2: return
        if kind_selector: kind_selector = Kind_Selector(kind_selector)
        return kind_selector, Type_Param_Attr_Spec(l1), Type_Param_Decl_List(l2)
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
    def tostr(self):
        s = 'INTEGER'
        if self.items[0] is not None:
            s += '%s, %s :: %s' % (self.items[0], self.items[1], self.items[2])
        else:
            s += ', %s :: %s' % (self.items[1], self.items[2])
        return s
    def torepr(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))        


class Type_Param_Decl(BinaryOpBase): # R436
    """
    <type-param-decl> = <type-param-name> [ = <scalar-int-initialization-expr> ]
    """
    subclass_names = ['Type_Param_Name']
    use_names = ['Scalar_Int_Initialization_Expr']
    def match(string):
        if '=' not in string: return
        lhs,rhs = string.split('=',1)
        lhs = lhs.rstrip()
        rhs = rhs.lstrip()
        if not lhs or not rhs: return
        return Type_Param_Name(lhs),'=',Scalar_Int_Initialization_Expr(rhs)
    match = staticmethod(match)
    
class Type_Param_Attr_Spec(STRINGBase): # R437
    """
    <type-param-attr-spec> = KIND
                             | LEN
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['KIND', 'LEN'], string)
    match = staticmethod(match)


class Component_Part(BlockBase): # R438
    """
    <component-part> = [ <component-def-stmt> ]...
    """
    subclass_names = []
    use_names = ['Component_Def_Stmt']
    def match(reader):
        content = []
        while 1:
            try:
                obj = Component_Def_Stmt(reader)
            except NoMatchError:
                obj = None
            if obj is None:
                break
            content.append(obj)
        if content:
            return content,
        return
    match = staticmethod(match)

    def tofortran(self, tab='', isfix=None):
        l = []
        for item in self.content:
            l.append(item.tofortran(tab=tab,isfix=isfix))
        return '\n'.join(l)

class Component_Def_Stmt(Base): # R439
    """
    <component-def-stmt> = <data-component-def-stmt>
                           | <proc-component-def-stmt>
    """
    subclass_names = ['Data_Component_Def_Stmt', 'Proc_Component_Def_Stmt']

class Data_Component_Def_Stmt(StmtBase): # R440
    """
    <data-component-def-stmt> = <declaration-type-spec> [ [ , <component-attr-spec-list> ] :: ] <component-decl-list>
    """
    subclass_names = []
    use_names = ['Declaration_Type_Spec', 'Component_Attr_Spec_List', 'Component_Decl_List']

class Dimension_Component_Attr_Spec(CALLBase):
    """
    <dimension-component-attr-spec> = DIMENSION ( <component-array-spec> )
    """
    subclass_names = []
    use_names = ['Component_Array_Spec']
    def match(string): return CALLBase.match('DIMENSION', Component_Array_Spec, string)
    match = staticmethod(match)

class Component_Attr_Spec(STRINGBase): # R441
    """
    <component-attr-spec> = POINTER
                            | DIMENSION ( <component-array-spec> )
                            | ALLOCATABLE
                            | <access-spec>
    """
    subclass_names = ['Access_Spec', 'Dimension_Component_Attr_Spec']
    use_names = []
    def match(string): return STRINGBase.match(['POINTER', 'ALLOCATABLE'], string)
    match = staticmethod(match)

class Component_Decl(Base): # R442
    """
    <component-decl> = <component-name> [ ( <component-array-spec> ) ] [ * <char-length> ] [ <component-initialization> ]
    """
    subclass_names = []
    use_names = ['Component_Name', 'Component_Array_Spec', 'Char_Length', 'Component_Initialization']
    def match(string):
        m = pattern.name.match(string)
        if m is None: return
        name = Component_Name(m.group())
        newline = string[m.end():].lstrip()
        if not newline: return name, None, None, None
        array_spec = None
        char_length = None
        init = None
        if newline.startswith('('):
            line, repmap = string_replace_map(newline)
            i = line.find(')')
            if i==-1: return
            array_spec = Component_Array_Spec(repmap(line[1:i].strip()))
            newline = repmap(line[i+1:].lstrip())
        if newline.startswith('*'):
            line, repmap = string_replace_map(newline)
            i = line.find('=')
            if i!=-1:
                char_length = repmap(line[1:i].strip())
                newline = repmap(newline[i:].lstrip())
            else:
                char_length = repmap(newline[1:].strip())
                newline = ''
            char_length = Char_Length(char_length)
        if newline.startswith('='):
            init = Component_Initialization(newline)
        else:
            assert newline=='',`newline`
        return name, array_spec, char_length, init
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        s = str(self.items[0])
        if self.items[1] is not None:
            s += '(' + str(self.items[1]) + ')'
        if self.items[2] is not None:
            s += '*' + str(self.items[2])
        if self.items[3] is not None:
            s += ' ' + str(self.items[3])
        return s
    def torepr(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class Component_Array_Spec(Base): # R443
    """
    <component-array-spec> = <explicit-shape-spec-list>
                             | <deferred-shape-spec-list>
    """
    subclass_names = ['Explicit_Shape_Spec_List', 'Deferred_Shape_Spec_List']

class Component_Initialization(Base): # R444
    """
    <component-initialization> =  = <initialization-expr>
                                 | => <null-init>
    """
    subclass_names = []
    use_names = ['Initialization_Expr', 'Null_Init']
    def match(string):
        if string.startswith('=>'):
            return '=>', Null_Init(string[2:].lstrip())
        if string.startswith('='):
            return '=', Initialization_Expr(string[2:].lstrip())
        return
    match = staticmethod(match)
    def init(self, op, rhs):
        self.op = op
        self.rhs = rhs
        return
    def tostr(self): return '%s %s' % (self.op, self.rhs)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.op, self.rhs)

class Proc_Component_Def_Stmt(StmtBase): # R445
    """
    <proc-component-def-stmt> = PROCEDURE ( [ <proc-interface> ] ) , <proc-component-attr-spec-list> :: <proc-decl-list>
    """
    subclass_names = []
    use_names = ['Proc_Interface', 'Proc_Component_Attr_Spec_List', 'Proc_Decl_List']

class Proc_Component_Attr_Spec(Base): # R446
    """
    <proc-component-attr-spec> = POINTER
                                 | PASS [ ( <arg-name> ) ]
                                 | NOPASS
                                 | <access-spec>
    """
    subclass_names = []
    use_names = ['Arg_Name', 'Access_Spec']

class Private_Components_Stmt(StmtBase): # R447
    """
    <private-components-stmt> = PRIVATE
    """
    subclass_names = []

class Type_Bound_Procedure_Part(Base): # R448
    """
    <type-bound-procedure-part> = <contains-stmt>
                                      [ <binding-private-stmt> ]
                                      <proc-binding-stmt>
                                      [ <proc-binding-stmt> ]...
    """
    subclass_names = []
    use_names = ['Contains_Stmt', 'Binding_Private_Stmt', 'Proc_Binding_Stmt']

class Binding_Private_Stmt(StmtBase): # R449
    """
    <binding-private-stmt> = PRIVATE
    """
    subclass_names = []

class Proc_Binding_Stmt(Base): # R450
    """
    <proc-binding-stmt> = <specific-binding>
                          | <generic-binding>
                          | <final-binding>
    """
    subclass_names = ['Specific_Binding', 'Generic_Binding', 'Final_Binding']

class Specific_Binding(Base): # R451
    """
    <specific-binding> = PROCEDURE [ ( <interface-name> ) ] [ [ , <binding-attr-list> ] :: ] <binding-name> [ => <procedure-name> ]
    """
    subclass_names = []
    use_names = ['Interface_Name', 'Binding_Attr_List', 'Binding_Name', 'Procedure_Name']

class Generic_Binding(Base): # R452
    """
    <generic-binding> = GENERIC [ , <access-spec> ] :: <generic-spec> => <binding-name-list>
    """
    subclass_names = []
    use_names = ['Access_Spec', 'Generic_Spec', 'Binding_Name_List']

class Binding_Attr(Base): # R453
    """
    <binding-attr> = PASS [ ( <arg-name> ) ]
                     | NOPASS
                     | NON_OVERRIDABLE
                     | <access-spec>
    """
    subclass_names = []
    use_names = ['Arg_Name', 'Access_Spec']

class Final_Binding(Base): # R454
    """
    <final-binding> = FINAL [ :: ] <final-subroutine-name-list>
    """
    subclass_names = []
    use_names = ['Final_Subroutine_Name_List']

class Derived_Type_Spec(CallBase): # R455
    """
    <derived-type-spec> = <type-name> [ ( <type-param-spec-list> ) ]
    """
    subclass_names = ['Type_Name']
    use_names = ['Type_Param_Spec_List']
    def match(string): return CallBase.match(Type_Name, Type_Param_Spec_List, string)
    match = staticmethod(match)

class Type_Param_Spec(KeywordValueBase): # R456
    """
    <type-param-spec> = [ <keyword> = ] <type-param-value>
    """
    subclass_names = ['Type_Param_Value']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Keyword, Type_Param_Value, string)
    match = staticmethod(match)

class Structure_Constructor_2(KeywordValueBase): # R457.b
    """
    <structure-constructor-2> = [ <keyword> = ] <component-data-source>
    """
    subclass_names = ['Component_Data_Source']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Keyword, Component_Data_Source, string)
    match = staticmethod(match)

class Structure_Constructor(CallBase): # R457
    """
    <structure-constructor> = <derived-type-spec> ( [ <component-spec-list> ] )
                            | <structure-constructor-2>
    """
    subclass_names = ['Structure_Constructor_2']
    use_names = ['Derived_Type_Spec', 'Component_Spec_List']
    def match(string): return CallBase.match(Derived_Type_Spec, Component_Spec_List, string)
    match = staticmethod(match)

class Component_Spec(KeywordValueBase): # R458
    """
    <component-spec> = [ <keyword> = ] <component-data-source>
    """
    subclass_names = ['Component_Data_Source']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Keyword, Component_Data_Source, string)
    match = staticmethod(match)

class Component_Data_Source(Base): # R459
    """
    <component-data-source> = <expr>
                              | <data-target>
                              | <proc-target>
    """
    subclass_names = ['Proc_Target', 'Data_Target', 'Expr']

class Enum_Def(Base): # R460
    """
    <enum-def> = <enum-def-stmt>
                     <enumerator-def-stmt>
                     [ <enumerator-def-stmt> ]...
                     <end-enum-stmt>
    """
    subclass_names = []
    use_names = ['Enum_Def_Stmt', 'Enumerator_Def_Stmt', 'End_Enum_Stmt']

class Enum_Def_Stmt(StmtBase): # R461
    """
    <enum-def-stmt> = ENUM, BIND(C)
    """
    subclass_names = []

class Enumerator_Def_Stmt(StmtBase): # R462
    """
    <enumerator-def-stmt> = ENUMERATOR [ :: ] <enumerator-list>
    """
    subclass_names = []
    use_names = ['Enumerator_List']

class Enumerator(BinaryOpBase): # R463
    """
    <enumerator> = <named-constant> [ = <scalar-int-initialization-expr> ]
    """
    subclass_names = ['Named_Constant']
    use_names = ['Scalar_Int_Initialization_Expr']
    def match(string):
        if '=' not in string: return
        lhs,rhs = string.split('=',1)
        return Named_Constant(lhs.rstrip()),'=',Scalar_Int_Initialization_Expr(rhs.lstrip())
    match = staticmethod(match)

class End_Enum_Stmt(EndStmtBase): # R464
    """
    <end-enum-stmt> = END ENUM
    """
    subclass_names = []
    def match(string): return EndStmtBase.match('ENUM',None, string, requite_stmt_type=True)
    match = staticmethod(match)

class Array_Constructor(BracketBase): # R465
    """
    <array-constructor> = (/ <ac-spec> /)
                          | <left-square-bracket> <ac-spec> <right-square-bracket>

    """
    subclass_names = []
    use_names = ['Ac_Spec']
    def match(string):
        try:
            obj = BracketBase.match('(//)', Ac_Spec, string)
        except NoMatchError:
            obj = None
        if obj is None:
            obj = BracketBase.match('[]', Ac_Spec, string)
        return obj
    match = staticmethod(match)

class Ac_Spec(Base): # R466
    """
    <ac-spec> = <type-spec> ::
                | [ <type-spec> :: ] <ac-value-list>
    """
    subclass_names = ['Ac_Value_List']
    use_names = ['Type_Spec']
    def match(string):
        if string.endswith('::'):
            return Type_Spec(string[:-2].rstrip()),None
        line, repmap = string_replace_map(string)
        i = line.find('::')
        if i==-1: return
        ts = line[:i].rstrip()
        line = line[i+2:].lstrip()
        ts = repmap(ts)
        line = repmap(line)
        return Type_Spec(ts),Ac_Value_List(line)
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[0] is None:
            return str(self.items[1])
        if self.items[1] is None:
            return str(self.items[0]) + ' ::'
        return str(self.items[0]) + ' :: ' + str(self.items[1])
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

# R467: <left-square-bracket> = [
# R468: <right-square-bracket> = ]

class Ac_Value(Base): # R469
    """
    <ac-value> = <expr>
                 | <ac-implied-do>
    """
    subclass_names = ['Ac_Implied_Do','Expr']

class Ac_Implied_Do(Base): # R470
    """
    <ac-implied-do> = ( <ac-value-list> , <ac-implied-do-control> )
    """
    subclass_names = []
    use_names = ['Ac_Value_List','Ac_Implied_Do_Control']    
    def match(string):
        if string[0]+string[-1] != '()': return
        line, repmap = string_replace_map(string[1:-1].strip())
        i = line.rfind('=')
        if i==-1: return
        j = line[:i].rfind(',')
        assert j!=-1
        s1 = repmap(line[:j].rstrip())
        s2 = repmap(line[j+1:].lstrip())
        return Ac_Value_List(s1),Ac_Implied_Do_Control(s2)
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return '(%s, %s)' % tuple(self.items)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Ac_Implied_Do_Control(Base): # R471
    """
    <ac-implied-do-control> = <ac-do-variable> = <scalar-int-expr> , <scalar-int-expr> [ , <scalar-int-expr> ]    
    """
    subclass_names = []
    use_names = ['Ac_Do_Variable','Scalar_Int_Expr']
    def match(string):
        i = string.find('=')
        if i==-1: return
        s1 = string[:i].rstrip()
        line, repmap = string_replace_map(string[i+1:].lstrip())
        t = line.split(',')
        if not (2<=len(t)<=3): return
        t = [Scalar_Int_Expr(s.strip()) for s in t]
        return Ac_Do_Variable(s1), t
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return '%s = %s' % (self.items[0], ', '.join(map(str,self.items[1])))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Ac_Do_Variable(Base): # R472
    """
    <ac-do-variable> = <scalar-int-variable>
    <ac-do-variable> shall be a named variable    
    """
    subclass_names = ['Scalar_Int_Variable']

###############################################################################
############################### SECTION  5 ####################################
###############################################################################

class Type_Declaration_Stmt(Base): # R501
    """
    <type-declaration-stmt> = <declaration-type-spec> [ [ , <attr-spec> ]... :: ] <entity-decl-list>
    """
    subclass_names = []
    use_names = ['Declaration_Type_Spec', 'Attr_Spec_List', 'Entity_Decl_List']

    def match(string):
        line, repmap = string_replace_map(string)
        i = line.find('::')
        if i!=-1:
            j = line[:i].find(',')
            if j!=-1:
                i = j
        else:
            if line[:6].upper()=='DOUBLE':
                m = re.search(r'\s[a-z_]',line[6:].lstrip(),re.I)
                if m is None: return
                i = m.start() + len(line)-len(line[6:].lstrip())
            else:
                m = re.search(r'\s[a-z_]',line,re.I)
                if m is None: return
                i = m.start()
        type_spec = Declaration_Type_Spec(repmap(line[:i].rstrip()))
        if type_spec is None: return
        line = line[i:].lstrip()
        if line.startswith(','):
            i = line.find('::')
            if i==-1: return
            attr_specs = Attr_Spec_List(repmap(line[1:i].strip()))
            if attr_specs is None: return
            line = line[i:]
        else:
            attr_specs = None
        if line.startswith('::'):
            line = line[2:].lstrip()
        entity_decls = Entity_Decl_List(repmap(line))
        if entity_decls is None: return
        return type_spec, attr_specs, entity_decls
    match = staticmethod(match)
    def init(self, *args):
        self.type_spec, self.attr_specs, self.entity_decls = args
        return
    def tostr(self):
        if self.attr_specs is None:
            return '%s :: %s' % (self.type_spec, self.entity_decls)
        else:
            return '%s, %s :: %s' % (self.type_spec, self.attr_specs, self.entity_decls)
    def torepr(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.type_spec, self.attr_specs, self.entity_decls)

class Declaration_Type_Spec(Base): # R502
    """
    <declaration-type-spec> = <intrinsic-type-spec>
                              | TYPE ( <derived-type-spec> )
                              | CLASS ( <derived-type-spec> )
                              | CLASS ( * )
    """
    subclass_names = ['Intrinsic_Type_Spec']
    use_names = ['Derived_Type_Spec']

    def match(string):
        if string[-1] != ')': return
        start = string[:4].upper()
        if start == 'TYPE':
            line = string[4:].lstrip()
            if not line.startswith('('): return
            return 'TYPE',Derived_Type_Spec(line[1:-1].strip())
        start = string[:5].upper()
        if start == 'CLASS':
            line = string[5:].lstrip()
            if not line.startswith('('): return
            line = line[1:-1].strip()
            if line=='*': return 'CLASS','*'
            return 'CLASS', Derived_Type_Spec(line)
        return
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return '%s(%s)' % tuple(map(str,self.items))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Dimension_Attr_Spec(CALLBase): # R503.d
    """
    <dimension-attr-spec> = DIMENSION ( <array-spec> )
    """
    subclass_names = []
    use_names = ['Array_Spec']
    def match(string): return CALLBase.match('DIMENSION', Array_Spec, string)
    match = staticmethod(match)

class Intent_Attr_Spec(CALLBase): # R503.f
    """
    <intent-attr-spec> = INTENT ( <intent-spec> )
    """
    subclass_names = []
    use_names = ['Intent_Spec']
    def match(string): return CALLBase.match('INTENT', Intent_Spec, string)
    match = staticmethod(match)

class Attr_Spec(STRINGBase): # R503
    """
    <attr-spec> = <access-spec>
                  | ALLOCATABLE
                  | ASYNCHRONOUS
                  | DIMENSION ( <array-spec> )
                  | EXTERNAL
                  | INTENT ( <intent-spec> )
                  | INTRINSIC
                  | <language-binding-spec>
                  | OPTIONAL
                  | PARAMETER
                  | POINTER
                  | PROTECTED
                  | SAVE
                  | TARGET
                  | VALUE
                  | VOLATILE
    """
    subclass_names = ['Access_Spec', 'Language_Binding_Spec',
                      'Dimension_Attr_Spec', 'Intent_Attr_Spec']
    use_names = []
    def match(string): return STRINGBase.match(pattern.abs_attr_spec, string)
    match = staticmethod(match)

class Entity_Decl(Base): # R504
    """
    <entity-decl> = <object-name> [ ( <array-spec> ) ] [ * <char-length> ] [ <initialization> ]
                    | <function-name> [ * <char-length> ]
    """
    subclass_names = []
    use_names = ['Object_Name', 'Array_Spec', 'Char_Length', 'Initialization', 'Function_Name']
    def match(string):
        m = pattern.name.match(string)
        if m is None: return
        name = Name(m.group())
        newline = string[m.end():].lstrip()
        if not newline: return name, None, None, None
        array_spec = None
        char_length = None
        init = None
        if newline.startswith('('):
            line, repmap = string_replace_map(newline)
            i = line.find(')')
            if i==-1: return
            array_spec = Array_Spec(repmap(line[1:i].strip()))
            newline = repmap(line[i+1:].lstrip())
        if newline.startswith('*'):
            line, repmap = string_replace_map(newline)
            i = line.find('=')
            if i!=-1:
                char_length = repmap(line[1:i].strip())
                newline = repmap(newline[i:].lstrip())
            else:
                char_length = repmap(newline[1:].strip())
                newline = ''
            char_length = Char_Length(char_length)
        if newline.startswith('='):
            init = Initialization(newline)
        else:
            assert newline=='',`newline`
        return name, array_spec, char_length, init
    match = staticmethod(match)
    def init(self, *args):
        self.name, self.array_spec, self.char_length, self.init = args
        return
    def tostr(self):
        s = str(self.name)
        if self.array_spec is not None:
            s += '(' + str(self.array_spec) + ')'
        if self.char_length is not None:
            s += '*' + str(self.char_length)
        if self.init is not None:
            s += ' ' + str(self.init)
        return s
    def torepr(self):
        return '%s(%r, %r, %r, %r)' \
               % (self.__class__.__name__, self.name, self.array_spec, self.char_length, self.init)
    
class Object_Name(Base): # R505
    """
    <object-name> = <name>
    """
    subclass_names = ['Name']

class Initialization(Base): # R506
    """
    <initialization> =  = <initialization-expr>
                       | => <null-init> 
    """
    subclass_names = []
    use_names = ['Initialization_Expr', 'Null_Init']
    def match(string):
        if string.startswith('=>'):
            return '=>', Null_Init(string[2:].lstrip())
        if string.startswith('='):
            return '=', Initialization_Expr(string[2:].lstrip())
        return
    match = staticmethod(match)
    def init(self, op, rhs):
        self.op = op
        self.rhs = rhs
        return
    def tostr(self): return '%s %s' % (self.op, self.rhs)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.op, self.rhs)

class Null_Init(STRINGBase): # R507
    """
    <null-init> = <function-reference>

    <function-reference> shall be a reference to the NULL intrinsic function with no arguments.
    """
    subclass_names = ['Function_Reference']
    def match(string): return STRINGBase.match('NULL', string)
    match = staticmethod(match)

class Access_Spec(STRINGBase): # R508
    """
    <access-spec> = PUBLIC
                    | PRIVATE
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['PUBLIC','PRIVATE'], string)
    match = staticmethod(match)

class Language_Binding_Spec(Base): # R509
    """
    <language-binding-spec> = BIND ( C [ , NAME = <scalar-char-initialization-expr> ] )
    """
    subclass_names = []
    use_names = ['Scalar_Char_Initialization_Expr']
    def match(string):
        start = string[:4].upper()
        if start != 'BIND': return
        line = string[4:].lstrip()
        if not line or line[0]+line[-1]!='()': return
        line = line[1:-1].strip()
        if not line: return
        start = line[0].upper()
        if start!='C': return
        line = line[1:].lstrip()
        if not line: return None,
        if not line.startswith(','): return
        line = line[1:].lstrip()
        start = line[:4].upper()
        if start!='NAME': return
        line=line[4:].lstrip()
        if not line.startswith('='): return
        return Scalar_Char_Initialization_Expr(line[1:].lstrip()),
    match = staticmethod(match)
    def init(self, name):
        self.name = name
    def tostr(self):
        if self.name is None: return 'BIND(C)'
        return 'BIND(C, NAME = %s)' % (self.name)
    def torepr(self):
        return '%s(%r)' % (self.__class__.__name__, self.name)

class Array_Spec(Base): # R510
    """
    <array-spec> = <explicit-shape-spec-list>
                   | <assumed-shape-spec-list>
                   | <deferred-shape-spec-list>
                   | <assumed-size-spec>
    """
    subclass_names = ['Assumed_Size_Spec', 'Explicit_Shape_Spec_List', 'Assumed_Shape_Spec_List',
                      'Deferred_Shape_Spec_List']

class Explicit_Shape_Spec(SeparatorBase): # R511
    """
    <explicit-shape-spec> = [ <lower-bound> : ] <upper-bound>
    """
    subclass_names = []
    use_names = ['Lower_Bound', 'Upper_Bound']
    def match(string):
        line, repmap = string_replace_map(string)
        if ':' not in line:
            return None, Upper_Bound(string)
        lower,upper = line.split(':',1)
        lower = lower.rstrip()
        upper = upper.lstrip()
        if not upper: return
        if not lower: return
        return Lower_Bound(repmap(lower)), Upper_Bound(repmap(upper))
    match = staticmethod(match)
    def tostr(self):
        if self.items[0] is None: return str(self.items[1])
        return SeparatorBase.tostr(self)
    
class Lower_Bound(Base): # R512
    """
    <lower-bound> = <specification-expr>
    """
    subclass_names = ['Specification_Expr']

class Upper_Bound(Base): # R513
    """
    <upper-bound> = <specification-expr>
    """
    subclass_names = ['Specification_Expr']

class Assumed_Shape_Spec(SeparatorBase): # R514
    """
    <assumed-shape-spec> = [ <lower-bound> ] :
    """
    subclass_names = []
    use_names = ['Lower_Bound']
    def match(string): return SeparatorBase.match(Lower_Bound, None, string)
    match = staticmethod(match)

class Deferred_Shape_Spec(SeparatorBase): # R515
    """
    <deferred_shape_spec> = :
    """
    subclass_names = []
    def match(string):
        if string==':': return None,None
        return
    match = staticmethod(match)

class Assumed_Size_Spec(Base): # R516
    """
    <assumed-size-spec> = [ <explicit-shape-spec-list> , ] [ <lower-bound> : ] *
    """
    subclass_names = []
    use_names = ['Explicit_Shape_Spec_List', 'Lower_Bound']
    def match(string):
        if not string.endswith('*'): return
        line = string[:-1].rstrip()
        if not line: return None,None
        if line.endswith(':'):
            line, repmap = string_replace_map(line[:-1].rstrip())
            i = line.rfind(',')
            if i==-1:
                return None, Lower_Bound(repmap(line))
            return Explicit_Shape_Spec_List(repmap(line[:i].rstrip())), Lower_Bound(repmap(line[i+1:].lstrip()))
        if not line.endswith(','): return
        line = line[:-1].rstrip()
        return Explicit_Shape_Spec_List(line), None
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        s = ''
        if self.items[0] is not None:
            s += str(self.items[0]) + ', '
        if self.items[1] is not None:
            s += str(self.items[1]) + ' : '
        s += '*'
        return s
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))
    
class Intent_Spec(STRINGBase): # R517
    """
    <intent-spec> = IN
                    | OUT
                    | INOUT
    """
    subclass_names = []
    def match(string): return STRINGBase.match(pattern.abs_intent_spec, string)
    match = staticmethod(match)

class Access_Stmt(StmtBase): # R518
    """
    <access-stmt> = <access-spec> [ [ :: ] <access-id-list> ]
    """
    subclass_names = []
    use_names = ['Access_Spec', 'Access_Id_List']

class Access_Id(Base): # R519
    """
    <access-id> = <use-name>
                  | <generic-spec>
    """
    subclass_names = ['Use_Name', 'Generic_Spec']

class Allocatable_Stmt(StmtBase): # R520
    """
    <allocateble-stmt> = ALLOCATABLE [ :: ] <object-name> [ ( <deferred-shape-spec-list> ) ] [ , <object-name> [ ( <deferred-shape-spec-list> ) ] ]...
    """
    subclass_names = []
    use_names = ['Object_Name', 'Deferred_Shape_Spec_List']

class Asynchronous_Stmt(StmtBase): # R521
    """
    <asynchronous-stmt> = ASYNCHRONOUS [ :: ] <object-name-list>
    """
    subclass_names = []
    use_names = ['Object_Name_List']

class Bind_Stmt(StmtBase): # R522
    """
    <bind-stmt> = <language-binding-spec> [ :: ] <bind-entity-list>
    """
    subclass_names = []
    use_names = ['Language_Binding_Spec', 'Bind_Entity_List']

class Bind_Entity(Base): # R523
    """
    <bind-entity> = <entity-name>
                    | / <common-block-name> /
    """
    subclass_names = ['Entity_Name']
    use_names = ['Common_Block_Name']

class Data_Stmt(StmtBase): # R524
    """
    <data-stmt> = DATA <data-stmt-set> [ [ , ] <data-stmt-set> ]...
    """
    subclass_names = []
    use_names = ['Data_Stmt_Set']

class Data_Stmt_Set(Base): # R525
    """
    <data-stmt-set> = <data-stmt-object-list> / <data-stmt-value-list> /
    """
    subclass_names = []
    use_names = ['Data_Stmt_Object_List', 'Data_Stmt_Value_List']

class Data_Stmt_Object(Base): # R526
    """
    <data-stmt-object> = <variable>
                         | <data-implied-do>
    """
    subclass_names = ['Variable', 'Data_Implied_Do']

class Data_Implied_Do(Base): # R527
    """
    <data-implied-do> = ( <data-i-do-object-list> , <data-i-do-variable> = <scalar-int-expr > , <scalar-int-expr> [ , <scalar-int-expr> ] )
    """
    subclass_names = []
    use_names = ['Data_I_Do_Object_List', 'Data_I_Do_Variable', 'Scalar_Int_Expr']

class Data_I_Do_Object(Base): # R528
    """
    <data-i-do-object> = <array-element>
                         | <scalar-structure-component>
                         | <data-implied-do>
    """
    subclass_names = ['Array_Element', 'Scalar_Structure_Component', 'Data_Implied_Do']

class Data_I_Do_Variable(Base): # R529
    """
    <data-i-do-variable> = <scalar-int-variable>
    """
    subclass_names = ['Scalar_Int_Variable']

class Data_Stmt_Value(Base): # R530
    """
    <data-stmt-value> = [ <data-stmt-repeat> * ] <data-stmt-constant>
    """
    subclass_names = []
    use_names = ['Data_Stmt_Repeat', 'Data_Stmt_Constant']

class Data_Stmt_Repeat(Base): # R531
    """
    <data-stmt-repeat> = <scalar-int-constant>
                         | <scalar-int-constant-subobject>
    """
    subclass_names = ['Scalar_Int_Constant', 'Scalar_Int_Constant_Subobject']

class Data_Stmt_Constant(Base): # R532
    """
    <data-stmt-constant> = <scalar-constant>
                           | <scalar-constant-subobject>
                           | <signed-int-literal-constant>
                           | <signed-real-literal-constant>
                           | <null-init>
                           | <structure-constructor>
    """
    subclass_names = ['Scalar_Constant', 'Scalar_Constant_Subobject',
                      'Signed_Int_Literal_Constant', 'Signed_Real_Literal_Constant',
                      'Null_Init', 'Structure_Constructor']

class Int_Constant_Subobject(Base): # R533
    """
    <int-constant-subobject> = <constant-subobject>
    """
    subclass_names = ['Constant_Subobject']

class Constant_Subobject(Base): # R534
    """
    <constant-subobject> = <designator>
    """
    subclass_names = ['Designator']

class Dimension_Stmt(StmtBase): # R535
    """
    <dimension-stmt> = DIMENSION [ :: ] <array-name> ( <array-spec> ) [ , <array-name> ( <array-spec> ) ]...
    """
    subclass_names = []
    use_names = ['Array_Name', 'Array_Spec']
    def match(string):
        if string[:9].upper()!='DIMENSION': return
        line, repmap = string_replace_map(string[9:].lstrip())
        if line.startswith('::'): line = line[2:].lstrip()
        decls = []
        for s in line.split(','):
            s = s.strip()
            if not s.endswith(')'): return
            i = s.find('(')
            if i==-1: return
            decls.append((Array_Name(repmap(s[:i].rstrip())), Array_Spec(repmap(s[i+1:-1].strip()))))
        if not decls: return
        return decls,
    match = staticmethod(match)
    def init(self, items):
        self.items = items
        return
    def tostr(self):
        return 'DIMENSION :: ' + ', '.join(['%s(%s)' % ns for ns in self.items])
    def torepr(self):
        return '%s(%r)' % (self.__class__.__name__, self.items)

class Intent_Stmt(StmtBase): # R536
    """
    <intent-stmt> = INTENT ( <intent-spec> ) [ :: ] <dummy-arg-name-list>
    """
    subclass_names = []
    use_names = ['Intent_Spec', 'Dummy_Arg_Name_List']

class Optional_Stmt(StmtBase): # R537
    """
    <optional-stmt> = OPTIONAL [ :: ] <dummy-arg-name-list>
    """
    subclass_names = []
    use_names = ['Dummy_Arg_Name_List']

class Parameter_Stmt(StmtBase, CALLBase): # R538
    """
    <parameter-stmt> = PARAMETER ( <named-constant-def-list> )
    """
    subclass_names = []
    use_names = ['Named_Constant_Def_List']
    def match(string): return CALLBase.match('PARAMETER', Named_Constant_Def_List, string, require_rhs=True)
    match = staticmethod(match)

class Named_Constant_Def(KeywordValueBase): # R539
    """
    <named-constant-def> = <named-constant> = <initialization-expr>
    """
    subclass_names = []
    use_names = ['Named_Constant', 'Initialization_Expr']
    def match(string): return KeywordValueBase.match(Named_Constant, Initialization_Expr, string)
    match = staticmethod(match)
    
class Pointer_Stmt(StmtBase): # R540
    """
    <pointer-stmt> = POINTER [ :: ] <pointer-decl-list>
    """
    subclass_names = []
    use_names = ['Pointer_Decl_List']

class Pointer_Decl(CallBase): # R541
    """
    <pointer-decl> = <object-name> [ ( <deferred-shape-spec-list> ) ]
                     | <proc-entity-name>
    """
    subclass_names = ['Proc_Entity_Name', 'Object_Name']
    use_names = ['Deferred_Shape_Spec_List']
    def match(string): return CallBase.match(Object_Name, Deferred_Shape_Spec_List, string, require_rhs=True)
    match = staticmethod(match)

class Protected_Stmt(StmtBase): # R542
    """
    <protected-stmt> = PROTECTED [ :: ] <entity-name-list>
    """
    subclass_names = []
    use_names = ['Entity_Name_List']

class Save_Stmt(StmtBase): # R543
    """
    <save-stmt> = SAVE [ [ :: ] <saved-entity-list> ]
    """
    subclass_names = []
    use_names = ['Saved_Entity_List']

class Saved_Entity(Base): # R544
    """
    <saved-entity> = <object-name>
                     | <proc-pointer-name>
                     | / <common-block-name> /
    """
    subclass_names = ['Object_Name', 'Proc_Pointer_Name']
    use_names = ['Common_Block_Name']

class Proc_Pointer_Name(Base): # R545
    """
    <proc-pointer-name> = <name>
    """
    subclass_names = ['Name']

class Target_Stmt(StmtBase): # R546
    """
    <target-stmt> = TARGET [ :: ] <object-name> [ ( <array-spec> ) ] [ , <object-name> [ ( <array-spec> ) ]]
    """
    subclass_names = []
    use_names = ['Object_Name', 'Array_Spec']

class Value_Stmt(StmtBase): # R547
    """
    <value-stmt> = VALUE [ :: ] <dummy-arg-name-list>
    """
    subclass_names = []
    use_names = ['Dummy_Arg_Name_List']

class Volatile_Stmt(StmtBase): # R548
    """
    <volatile-stmt> = VOLATILE [ :: ] <object-name-list>
    """
    subclass_names = []
    use_names = ['Object_Name_List']

class Implicit_Stmt(StmtBase, WORDClsBase): # R549
    """
    <implicit-stmt> = IMPLICIT <implicit-spec-list>
                      | IMPLICIT NONE
    """
    subclass_names = []
    use_names = ['Implicit_Spec_List']
    def match(string):
        for w,cls in [(pattern.abs_implicit_none, None),
                      ('IMPLICIT', Implicit_Spec_List)]:
            try:
                obj = WORDClsBase.match(w, cls, string)
            except NoMatchError:
                obj = None
            if obj is not None: return obj
        return
    match = staticmethod(match)

class Implicit_Spec(CallBase): # R550
    """
    <implicit-spec> = <declaration-type-spec> ( <letter-spec-list> )
    """
    subclass_names = []
    use_names = ['Declaration_Type_Spec', 'Letter_Spec_List']
    def match(string):
        if not string.endswith(')'): return
        i = string.rfind('(')
        if i==-1: return
        s1 = string[:i].rstrip()
        s2 = string[i+1:-1].strip()
        if not s1 or not s2: return
        return Declaration_Type_Spec(s1), Letter_Spec_List(s2)
    match = staticmethod(match)

class Letter_Spec(Base): # R551
    """
    <letter-spec> = <letter> [ - <letter> ]
    """
    subclass_names = []
    def match(string):
        if len(string)==1:
            lhs = string.upper()
            if 'A'<=lhs<='Z': return lhs, None
            return
        if '-' not in string: return
        lhs,rhs = string.split('-',1)
        lhs = lhs.strip().upper()
        rhs = rhs.strip().upper()
        if not len(lhs)==len(rhs)==1: return
        if not ('A'<=lhs<=rhs<='Z'): return
        return lhs,rhs
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        if self.items[1] is None: return str(self.items[0])
        return '%s - %s' % tuple(self.items)
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class Namelist_Stmt(StmtBase): # R552
    """
    <namelist-stmt> = NAMELIST / <namelist-group-name> / <namelist-group-object-list> [ [ , ] / <namelist-group-name> / <namelist-group-object-list> ]
    """
    subclass_names = []
    use_names = ['Namelist_Group_Name', 'Namelist_Group_Object_List']

class Namelist_Group_Object(Base): # R553
    """
    <namelist-group-object> = <variable-name>
    """
    subclass_names = ['Variable_Name']

class Equivalence_Stmt(StmtBase, WORDClsBase): # R554
    """
    <equivalence-stmt> = EQUIVALENCE <equivalence-set-list>
    """
    subclass_names = []
    use_names = ['Equivalence_Set_List']
    def match(string): return WORDClsBase.match('EQUIVALENCE', Equivalence_Set_List, string)
    match = staticmethod(match)

class Equivalence_Set(Base): # R555
    """
    <equivalence-set> = ( <equivalence-object> , <equivalence-object-list> )
    """
    subclass_names = []
    use_names = ['Equivalence_Object', 'Equivalence_Object_List']
    def match(string):
        if not string or string[0]+string[-1]!='()': return
        line = string[1:-1].strip()
        if not line: return
        l = Equivalence_Object_List(line)
        obj = l.items[0]
        l.items = l.items[1:]
        if not l.items: return
        return obj, l
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self): return '(%s, %s)' % tuple(self.items)
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class Equivalence_Object(Base): # R556
    """
    <equivalence-object> = <variable-name>
                           | <array-element>
                           | <substring>
    """
    subclass_names = ['Variable_Name', 'Array_Element', 'Substring']

class Common_Stmt(StmtBase): # R557
    """
    <common-stmt> = COMMON [ / [ <common-block-name> ] / ] <common-block-object-list> [ [ , ] / [ <common-block-name> ] / <common-block-object-list> ]...
    """
    subclass_names = []
    use_names = ['Common_Block_Name', 'Common_Block_Object_List']
    def match(string):
        if string[:6].upper()!='COMMON': return
        line = string[6:]
        if not line or 'A'<=line[0].upper()<='Z' or line[0]=='_': return
        line, repmap = string_replace_map(line.lstrip())
        items = []
        if line.startswith('/'):
            i = line.find('/',1)
            if i==-1: return
            name = line[1:i].strip() or None
            if name is not None: name = Common_Block_Name(name)
            line = line[i+1:].lstrip()
            i = line.find('/')
            if i==-1:
                lst = Common_Block_Object_List(repmap(line))
                line = ''
            else:
                l = line[:i].rstrip()
                if l.endswith(','): l = l[:-1].rstrip()
                if not l: return
                lst = Common_Block_Object_List(repmap(l))
                line = line[i:].lstrip()
        else:
            name = None
            i = line.find('/')
            if i==-1:
                lst = Common_Block_Object_List(repmap(line))
                line = ''
            else:
                l = line[:i].rstrip()
                if l.endswith(','): l = l[:-1].rstrip()
                if not l: return
                lst = Common_Block_Object_List(repmap(l))
                line = line[i:].lstrip()
        items.append((name, lst))
        while line:
            if line.startswith(','): line = line[1:].lstrip()
            if not line.startswith('/'): return
            i = line.find('/',1)
            name = line[1:i].strip() or None
            if name is not None: name = Common_Block_Name(name)
            line = line[i+1:].lstrip()
            i = line.find('/')
            if i==-1:
                lst = Common_Block_Object_List(repmap(line))
                line = ''
            else:
                l = line[:i].rstrip()
                if l.endswith(','): l = l[:-1].rstrip()
                if not l: return
                lst = Common_Block_Object_List(repmap(l))
                line = line[i:].lstrip()
            items.append((name, lst))
        return items,
    match = staticmethod(match)
    def init(self, items):
        self.items = items
        return
    def tostr(self):
        s = 'COMMON'
        for (name, lst) in self.items:
            if name is not None:
                s += ' /%s/ %s' % (name, lst)
            else:
                s += ' // %s' % (lst)
        return s
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.items)

class Common_Block_Object(CallBase): # R558
    """
    <common-block-object> = <variable-name> [ ( <explicit-shape-spec-list> ) ]
                            | <proc-pointer-name>
    """
    subclass_names = ['Proc_Pointer_Name','Variable_Name']
    use_names = ['Variable_Name', 'Explicit_Shape_Spec_List']
    def match(string): return CallBase.match(Variable_Name, Explicit_Shape_Spec_List, string, require_rhs=True)
    match = staticmethod(match)

###############################################################################
############################### SECTION  6 ####################################
###############################################################################

class Variable(Base): # R601
    """
    <variable> = <designator>
    """
    subclass_names = ['Designator']

class Variable_Name(Base): # R602
    """
    <variable-name> = <name>
    """
    subclass_names = ['Name']

class Designator(Base): # R603
    """
    <designator> = <object-name>
                   | <array-element>
                   | <array-section>
                   | <structure-component>
                   | <substring>
    <substring-range> = [ <scalar-int-expr> ] : [ <scalar-int-expr> ]
    <structure-component> = <data-ref>
    """
    subclass_names = ['Object_Name','Array_Section','Array_Element','Structure_Component',
                      'Substring'
                      ]

class Logical_Variable(Base): # R604
    """
    <logical-variable> = <variable>
    """
    subclass_names = ['Variable']

class Default_Logical_Variable(Base): # R605
    """
    <default-logical-variable> = <variable>
    """
    subclass_names = ['Variable']

class Char_Variable(Base): # R606
    """
    <char-variable> = <variable>
    """
    subclass_names = ['Variable']

class Default_Char_Variable(Base): # R607
    """
    <default-char-variable> = <variable>
    """
    subclass_names = ['Variable']


class Int_Variable(Base): # R608
    """
    <int-variable> = <variable>
    """
    subclass_names = ['Variable']


class Substring(CallBase): # R609
    """
    <substring> = <parent-string> ( <substring-range> )
    """
    subclass_names = []
    use_names = ['Parent_String','Substring_Range']
    def match(string): return CallBase.match(Parent_String, Substring_Range, string, require_rhs=True)
    match = staticmethod(match)

class Parent_String(Base): # R610
    """
    <parent-string> = <scalar-variable-name>
                      | <array-element>
                      | <scalar-structure-component>
                      | <scalar-constant>    
    """
    subclass_names = ['Scalar_Variable_Name', 'Array_Element', 'Scalar_Structure_Component', 'Scalar_Constant']

class Substring_Range(SeparatorBase): # R611
    """
    <substring-range> = [ <scalar-int-expr> ] : [ <scalar-int-expr> ]
    """
    subclass_names = []
    use_names = ['Scalar_Int_Expr']
    def match(string):
        return SeparatorBase.match(Scalar_Int_Expr, Scalar_Int_Expr, string)
    match = staticmethod(match)

class Data_Ref(SequenceBase): # R612
    """
    <data-ref> = <part-ref> [ % <part-ref> ]...
    """
    subclass_names = ['Part_Ref']
    use_names = []
    def match(string): return SequenceBase.match(r'%', Part_Ref, string)
    match = staticmethod(match)

class Part_Ref(CallBase): # R613
    """
    <part-ref> = <part-name> [ ( <section-subscript-list> ) ]
    """
    subclass_names = ['Part_Name']
    use_names = ['Section_Subscript_List']
    def match(string):
        return CallBase.match(Part_Name, Section_Subscript_List, string, require_rhs=True)
    match = staticmethod(match)

class Structure_Component(Base): # R614
    """
    <structure-component> = <data-ref>
    """
    subclass_names = ['Data_Ref']

class Type_Param_Inquiry(BinaryOpBase): # R615
    """
    <type-param-inquiry> = <designator> % <type-param-name>
    """
    subclass_names = []
    use_names = ['Designator','Type_Param_Name']
    def match(string):
        return BinaryOpBase.match(\
            Designator, pattern.percent_op.named(), Type_Param_Name, string)
    match = staticmethod(match)

class Array_Element(Base): # R616
    """
    <array-element> = <data-ref>
    """
    subclass_names = ['Data_Ref']

class Array_Section(CallBase): # R617
    """
    <array-section> = <data-ref> [ ( <substring-range> ) ]
    """
    subclass_names = ['Data_Ref']
    use_names = ['Substring_Range']
    def match(string): return CallBase.match(Data_Ref, Substring_Range, string, require_rhs=True)
    match = staticmethod(match)

class Subscript(Base): # R618
    """
    <subscript> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Section_Subscript(Base): # R619
    """
    <section-subscript> = <subscript>
                          | <subscript-triplet>
                          | <vector-subscript>
    """
    subclass_names = ['Subscript_Triplet', 'Vector_Subscript', 'Subscript']

class Subscript_Triplet(Base): # R620
    """
    <subscript-triplet> = [ <subscript> ] : [ <subscript> ] [ : <stride> ]
    """
    subclass_names = []
    use_names = ['Subscript','Stride']
    def match(string):
        line, repmap = string_replace_map(string)
        t = line.split(':')
        if len(t)<=1 or len(t)>3: return
        lhs_obj,rhs_obj, stride_obj = None, None, None
        if len(t)==2:
            lhs,rhs = t[0].rstrip(),t[1].lstrip()
        else:
            lhs,rhs,stride = t[0].rstrip(),t[1].strip(),t[2].lstrip()
            if stride:
                stride_obj = Stride(repmap(stride))
        if lhs:
            lhs_obj = Subscript(repmap(lhs))
        if rhs:
            rhs_obj = Subscript(repmap(rhs))
        return lhs_obj, rhs_obj, stride_obj
    match = staticmethod(match)
    def init(self, lhs, rhs, stride):
        self.lhs, self.rhs, self.stride =lhs, rhs, stride
        return
    def tostr(self):
        s = ''
        if self.lhs is not None:
            s += str(self.lhs) + ' :'
        else:
            s += ':'
        if self.rhs is not None:
            s += ' ' + str(self.rhs)
        if self.stride is not None:
            s += ' : ' + str(self.stride)
        return s
    def torepr(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,self.lhs, self.rhs, self.stride)        

class Stride(Base): # R621
    """
    <stride> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Vector_Subscript(Base): # R622
    """
    <vector-subscript> = <int-expr>
    """
    subclass_names = ['Int_Expr']

class Allocate_Stmt(StmtBase): # R623
    """
    <allocate-stmt> = ALLOCATE ( [ <type-spec> :: ] <allocation-list> [ , <alloc-opt-list> ] )
    """
    subclass_names = []
    use_names = ['Type_Spec', 'Allocation_List', 'Alloc_Opt_List']
    
class Alloc_Opt(KeywordValueBase):# R624
    """
    <alloc-opt> = STAT = <stat-variable>
                  | ERRMSG = <errmsg-variable>
                  | SOURCE = <source-expr>
    """
    subclass_names = []
    use_names = ['Stat_Variable', 'Errmsg_Variable', 'Source_Expr']
    def match(string):
        for (k,v) in [('STAT', Stat_Variable),
                      ('ERRMSG', Errmsg_Variable),
                      ('SOURCE', Source_Expr)
                      ]:
            try:
                obj = KeywordValueBase.match(k, v, string, upper_lhs = True)
            except NoMatchError:
                obj = None
            if obj is not None: return obj
        return
    match = staticmethod(match)


class Stat_Variable(Base):# R625
    """
    <stat-variable> = <scalar-int-variable>
    """
    subclass_names = ['Scalar_Int_Variable']

class Errmsg_Variable(Base):# R626
    """
    <errmsg-variable> = <scalar-default-char-variable>
    """
    subclass_names = ['Scalar_Default_Char_Variable']

class Source_Expr(Base):# R627
    """
    <source-expr> = <expr>
    """
    subclass_names = ['Expr']

class Allocation(Base):# R628
    """
    <allocation> = <allocate-object> [ <allocate-shape-spec-list> ]
                 | <variable-name>
    """
    subclass_names = ['Variable_Name']
    use_names = ['Allocate_Object', 'Allocate_Shape_Spec_List']

class Allocate_Object(Base): # R629
    """
    <allocate-object> = <variable-name>
                        | <structure-component>
    """
    subclass_names = ['Variable_Name', 'Structure_Component']

class Allocate_Shape_Spec(SeparatorBase): # R630
    """
    <allocate-shape-spec> = [ <lower-bound-expr> : ] <upper-bound-expr>
    """
    subclass_names = []
    use_names = ['Lower_Bound_Expr', 'Upper_Bound_Expr']
    def match(string):
        line, repmap = string_replace_map(string)
        if ':' not in line: return None, Upper_Bound_Expr(string)
        lower,upper = line.split(':',1)
        lower = lower.rstrip()
        upper = upper.lstrip()
        if not upper: return
        if not lower: return
        return Lower_Bound_Expr(repmap(lower)), Upper_Bound_Expr(repmap(upper))
    match = staticmethod(match)
    def tostr(self):
        if self.items[0] is None: return str(self.items[1])
        return SeparatorBase.tostr(self)


class Lower_Bound_Expr(Base): # R631
    """
    <lower-bound-expr> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Upper_Bound_Expr(Base): # R632
    """
    <upper-bound-expr> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Nullify_Stmt(StmtBase, CALLBase): # R633
    """
    <nullify-stmt> = NULLIFY ( <pointer-object-list> )
    """
    subclass_names = []
    use_names = ['Pointer_Object_List']
    def match(string): return CALLBase.match('NULLIFY', Pointer_Object_List, string, require_rhs=True)
    match = staticmethod(match)
    
class Pointer_Object(Base): # R634
    """
    <pointer-object> = <variable-name>
                       | <structure-component>
                       | <proc-pointer-name>
    """
    subclass_names = ['Variable_Name', 'Structure_Component', 'Proc_Pointer_Name']

class Deallocate_Stmt(StmtBase): # R635
    """
    <deallocate-stmt> = DEALLOCATE ( <allocate-object-list> [ , <dealloc-opt-list> ] )
    """
    subclass_names = []
    use_names = ['Allocate_Object_List', 'Dealloc_Opt_List']

class Dealloc_Opt(KeywordValueBase): # R636
    """
    <dealloc-opt> = STAT = <stat-variable>
                    | ERRMSG = <errmsg-variable>
    """
    subclass_names = []
    use_names = ['Stat_Variable', 'Errmsg_Variable']
    def match(string):
        for (k,v) in [('STAT', Stat_Variable),
                      ('ERRMSG', Errmsg_Variable),
                      ]:
            try:
                obj = KeywordValueBase.match(k, v, string, upper_lhs = True)
            except NoMatchError:
                obj = None
            if obj is not None: return obj
        return
    match = staticmethod(match)

class Scalar_Char_Initialization_Expr(Base):
    subclass_names = ['Char_Initialization_Expr']

###############################################################################
############################### SECTION  7 ####################################
###############################################################################

class Primary(Base): # R701
    """
    <primary> = <constant>
                | <designator>
                | <array-constructor>
                | <structure-constructor>
                | <function-reference>
                | <type-param-inquiry>
                | <type-param-name>
                | ( <expr> )
    """
    subclass_names = ['Constant', 'Parenthesis', 'Designator','Array_Constructor',
                      'Structure_Constructor',
                      'Function_Reference', 'Type_Param_Inquiry', 'Type_Param_Name', 
                       ]

class Parenthesis(BracketBase): # R701.h
    """
    <parenthesis> = ( <expr> )
    """
    subclass_names = []
    use_names = ['Expr']
    def match(string): return BracketBase.match('()', Expr, string)
    match = staticmethod(match)

class Level_1_Expr(UnaryOpBase): # R702
    """
    <level-1-expr> = [ <defined-unary-op> ] <primary>
    <defined-unary-op> = . <letter> [ <letter> ]... .
    """
    subclass_names = ['Primary']
    use_names = []
    def match(string):
        if pattern.non_defined_binary_op.match(string):
            raise NoMatchError,'%s: %r' % (Level_1_Expr.__name__, string)
        return UnaryOpBase.match(\
            pattern.defined_unary_op.named(),Primary,string)
    match = staticmethod(match)

#R703: <defined-unary-op> = . <letter> [ <letter> ]... . 

class Mult_Operand(BinaryOpBase): # R704
    """
    <mult-operand> = <level-1-expr> [ <power-op> <mult-operand> ]
    <power-op> = **
    """
    subclass_names = ['Level_1_Expr']
    use_names = ['Mult_Operand']
    def match(string):
        return BinaryOpBase.match(\
            Level_1_Expr,pattern.power_op.named(),Mult_Operand,string,right=False)
    match = staticmethod(match)

class Add_Operand(BinaryOpBase): # R705
    """
    <add-operand> = [ <add-operand> <mult-op> ] <mult-operand>
    <mult-op>  = *
                 | /
    """
    subclass_names = ['Mult_Operand']
    use_names = ['Add_Operand','Mult_Operand']
    def match(string):
        return BinaryOpBase.match(Add_Operand,pattern.mult_op.named(),Mult_Operand,string)
    match = staticmethod(match)

class Level_2_Expr(BinaryOpBase): # R706
    """
    <level-2-expr> = [ [ <level-2-expr> ] <add-op> ] <add-operand>
    <level-2-expr> = [ <level-2-expr> <add-op> ] <add-operand>
                     | <level-2-unary-expr>
    <add-op>   = +
                 | -
    """
    subclass_names = ['Level_2_Unary_Expr']
    use_names = ['Level_2_Expr']
    def match(string):
        return BinaryOpBase.match(\
            Level_2_Expr,pattern.add_op.named(),Add_Operand,string)
    match = staticmethod(match)

class Level_2_Unary_Expr(UnaryOpBase): # R706.c
    """
    <level-2-unary-expr> = [ <add-op> ] <add-operand>
    """
    subclass_names = ['Add_Operand']
    use_names = []
    def match(string): return UnaryOpBase.match(pattern.add_op.named(),Add_Operand,string)
    match = staticmethod(match)

#R707: <power-op> = **
#R708: <mult-op> = * | /
#R709: <add-op> = + | -

class Level_3_Expr(BinaryOpBase): # R710
    """
    <level-3-expr> = [ <level-3-expr> <concat-op> ] <level-2-expr>
    <concat-op>    = //
    """
    subclass_names = ['Level_2_Expr']
    use_names =['Level_3_Expr']
    def match(string):
        return BinaryOpBase.match(\
            Level_3_Expr,pattern.concat_op.named(),Level_2_Expr,string)
    match = staticmethod(match)

#R711: <concat-op> = //

class Level_4_Expr(BinaryOpBase): # R712
    """
    <level-4-expr> = [ <level-3-expr> <rel-op> ] <level-3-expr>
    <rel-op> = .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. | == | /= | < | <= | > | >=
    """
    subclass_names = ['Level_3_Expr']
    use_names = []
    def match(string):
        return BinaryOpBase.match(\
            Level_3_Expr,pattern.rel_op.named(),Level_3_Expr,string)
    match = staticmethod(match)

#R713: <rel-op> = .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. | == | /= | < | <= | > | >=

class And_Operand(UnaryOpBase): # R714
    """
    <and-operand> = [ <not-op> ] <level-4-expr>
    <not-op> = .NOT.
    """
    subclass_names = ['Level_4_Expr']
    use_names = []
    def match(string):
        return UnaryOpBase.match(\
            pattern.not_op.named(),Level_4_Expr,string)
    match = staticmethod(match)

class Or_Operand(BinaryOpBase): # R715
    """
    <or-operand> = [ <or-operand> <and-op> ] <and-operand>    
    <and-op> = .AND.
    """
    subclass_names = ['And_Operand']
    use_names = ['Or_Operand','And_Operand']
    def match(string):
        return BinaryOpBase.match(\
            Or_Operand,pattern.and_op.named(),And_Operand,string)
    match = staticmethod(match)

class Equiv_Operand(BinaryOpBase): # R716
    """
    <equiv-operand> = [ <equiv-operand> <or-op> ] <or-operand>
    <or-op>  = .OR.
    """
    subclass_names = ['Or_Operand']
    use_names = ['Equiv_Operand']
    def match(string):
        return BinaryOpBase.match(\
            Equiv_Operand,pattern.or_op.named(),Or_Operand,string)
    match = staticmethod(match)


class Level_5_Expr(BinaryOpBase): # R717
    """
    <level-5-expr> = [ <level-5-expr> <equiv-op> ] <equiv-operand>
    <equiv-op> = .EQV.
               | .NEQV.
    """
    subclass_names = ['Equiv_Operand']
    use_names = ['Level_5_Expr']
    def match(string):
        return BinaryOpBase.match(\
            Level_5_Expr,pattern.equiv_op.named(),Equiv_Operand,string)
    match = staticmethod(match)

#R718: <not-op> = .NOT.
#R719: <and-op> = .AND.
#R720: <or-op> = .OR.
#R721: <equiv-op> = .EQV. | .NEQV.

class Expr(BinaryOpBase): # R722
    """
    <expr> = [ <expr> <defined-binary-op> ] <level-5-expr>
    <defined-binary-op> = . <letter> [ <letter> ]... .
    TODO: defined_binary_op must not be intrinsic_binary_op!!
    """
    subclass_names = ['Level_5_Expr']
    use_names = ['Expr']
    def match(string):
        return BinaryOpBase.match(Expr, pattern.defined_binary_op.named(), Level_5_Expr,
                                   string)
    match = staticmethod(match)

#R723: <defined-binary-op> = . <letter> [ <letter> ]... .

class Logical_Expr(Base): # R724
    """
    <logical-expr> = <expr>
    """
    subclass_names = ['Expr']

class Char_Expr(Base): # R725
    """
    <char-expr> = <expr>
    """
    subclass_names = ['Expr']

class Default_Char_Expr(Base): # R726
    """
    <default-char-expr> = <expr>
    """
    subclass_names = ['Expr']

class Int_Expr(Base): # R727
    """
    <int-expr> = <expr>
    """
    subclass_names = ['Expr']

class Numeric_Expr(Base): # R728
    """
    <numeric-expr> = <expr>
    """
    subclass_names = ['Expr']

class Specification_Expr(Base): # R729
    """
    <specification-expr> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Initialization_Expr(Base): # R730
    """
    <initialization-expr> = <expr>
    """
    subclass_names = ['Expr']

class Char_Initialization_Expr(Base): # R731
    """
    <char-initialization-expr> = <char-expr>
    """
    subclass_names = ['Char_Expr']

class Int_Initialization_Expr(Base): # R732
    """
    <int-initialization-expr> = <int-expr>
    """
    subclass_names = ['Int_Expr']

class Logical_Initialization_Expr(Base): # R733
    """
    <logical-initialization-expr> = <logical-expr>
    """
    subclass_names = ['Logical_Expr']

class Assignment_Stmt(StmtBase, BinaryOpBase): # R734
    """
    <assignment-stmt> = <variable> = <expr>
    """
    subclass_names = []
    use_names = ['Variable', 'Expr']
    def match(string):
        return BinaryOpBase.match(Variable, '=', Expr, string, right=False)
    match = staticmethod(match)
    
class Pointer_Assignment_Stmt(StmtBase): # R735
    """
    <pointer-assignment-stmt> = <data-pointer-object> [ ( <bounds-spec-list> ) ] => <data-target>
                                | <data-pointer-object> ( <bounds-remapping-list> ) => <data-target>
                                | <proc-pointer-object> => <proc-target>
    """
    subclass_names = []
    use_names = ['Data_Pointer_Object', 'Bounds_Spec_List', 'Data_Target', 'Bounds_Remapping_List',
                 'Proc_Pointer_Object', 'Proc_Target']

class Data_Pointer_Object(Base): # R736
    """
    <data-pointer-object> = <variable-name>
                            | <variable> % <data-pointer-component-name>
    """
    subclass_names = ['Variable_Name']
    use_names = ['Variable', 'Data_Pointer_Component_Name']

class Bounds_Spec(SeparatorBase): # R737
    """
    <bounds-spec> = <lower-bound-expr> :
    """
    subclass_names = []
    use_names = ['Lower_Bound_Expr']
    def match(string): return SeparatorBase.match(Lower_Bound_Expr, None, string, require_lhs=True)
    match = staticmethod(match)

class Bounds_Remapping(SeparatorBase): # R738
    """
    <bounds-remapping> = <lower-bound-expr> : <upper-bound-expr>
    """
    subclass_names = []
    use_classes = ['Lower_Bound_Expr', 'Upper_Bound_Expr']
    def match(string): return SeparatorBase.match(Lower_Bound_Expr, Upper_Bound_Expr, string, require_lhs=True, require_rhs=True)
    match = staticmethod(match)

class Data_Target(Base): # R739
    """
    <data-target> = <variable>
                    | <expr>
    """
    subclass_names = ['Variable','Expr']

class Proc_Pointer_Object(Base): # R740
    """
    <proc-pointer-object> = <proc-pointer-name>
                            | <proc-component-ref>
    """
    subclass_names = ['Proc_Pointer_Name', 'Proc_Component_Ref']

class Proc_Component_Ref(BinaryOpBase): # R741
    """
    <proc-component-ref> = <variable> % <procedure-component-name>
    """
    subclass_names = []
    use_names = ['Variable','Procedure_Component_Name']
    def match(string):
        return BinaryOpBase.match(Variable, r'%', Procedure_Component_Name, string)            
    match = staticmethod(match)

class Proc_Target(Base): # R742
    """
    <proc-target> = <expr>
                    | <procedure-name>
                    | <proc-component-ref>
    """
    subclass_names = ['Proc_Component_Ref', 'Procedure_Name', 'Expr']


class Where_Stmt(StmtBase): # R743
    """
    <where-stmt> = WHERE ( <mask-expr> ) <where-assignment-stmt>
    """
    subclass_names = []
    use_names = ['Mask_Expr', 'Where_Assignment_Stmt']

class Where_Construct(Base): # R744
    """
    <where-construct> = <where-construct-stmt>
                              [ <where-body-construct> ]...
                            [ <masked-elsewhere-stmt>
                              [ <where-body-construct> ]...
                            ]...
                            [ <elsewhere-stmt>
                              [ <where-body-construct> ]... ]
                            <end-where-stmt>
    """
    subclass_names = []
    use_names = ['Where_Construct_Stmt', 'Where_Body_Construct',
                 'Elsewhere_Stmt', 'End_Where_Stmt'
                 ]

class Where_Construct_Stmt(StmtBase): # R745
    """
    <where-construct-stmt> = [ <where-construct-name> : ] WHERE ( <mask-expr> )
    """
    subclass_names = []
    use_names = ['Where_Construct_Name', 'Mask_Expr']

class Where_Body_Construct(Base): # R746
    """
    <where-body-construct> = <where-assignment-stmt>
                             | <where-stmt>
                             | <where-construct>
    """
    subclass_names = ['Where_Assignment_Stmt', 'Where_Stmt', 'Where_Construct']

class Where_Assignment_Stmt(Base): # R747
    """
    <where-assignment-stmt> = <assignment-stmt>
    """
    subclass_names = ['Assignment_Stmt']

class Mask_Expr(Base): # R748
    """
    <mask-expr> = <logical-expr>
    """
    subclass_names = ['Logical_Expr']

class Masked_Elsewhere_Stmt(StmtBase): # R749
    """
    <masked-elsewhere-stmt> = ELSEWHERE ( <mask-expr> ) [ <where-construct-name> ]
    """
    subclass_names = []
    use_names = ['Mask_Expr', 'Where_Construct_Name']

class Elsewhere_Stmt(StmtBase): # R750
    """
    <elsewhere-stmt> = ELSEWHERE [ <where-construct-name> ]
    """
    subclass_names = []
    use_names = ['Where_Construct_Name']

class End_Where_Stmt(EndStmtBase): # R751
    """
    <end-where-stmt> = END WHERE [ <where-construct-name> ]
    """
    subclass_names = []
    use_names = ['Where_Construct_Name']
    def match(string): return EndStmtBase.match('WHERE',Where_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)


class Forall_Construct(Base): # R752
    """
    <forall-construct> = <forall-construct-stmt>
                             [ <forall-body-construct> ]...
                             <end-forall-stmt>
    """
    subclass_names = []
    use_names = ['Forall_Construct_Stmt', 'Forall_Body_Construct', 'End_Forall_Stmt']

class Forall_Construct_Stmt(StmtBase): # R753
    """
    <forall-construct-stmt> = [ <forall-construct-name> : ] FORALL <forall-header>
    """
    subclass_names = []
    use_names = ['Forall_Construct_Name', 'Forall_Header']

class Forall_Header(Base): # R754
    """
    <forall-header> = ( <forall-triplet-spec-list> [ , <scalar-mask-expr> ] )
    """
    subclass_names = []
    use_names = ['Forall_Triplet_Spec_List', 'Scalar_Mask_Expr']

class Forall_Triplet_Spec(Base): # R755
    """
    <forall-triplet-spec> = <index-name> = <subscript> : <subscript> [ : <stride> ]
    """
    subclass_names = []
    use_names = ['Index_Name', 'Subscript', 'Stride']

class Forall_Body_Construct(Base): # R756
    """
    <forall-body-construct> = <forall-assignment-stmt>
                              | <where-stmt>
                              | <where-construct>
                              | <forall-construct>
                              | <forall-stmt>
    """
    subclass_names = ['Forall_Assignment_Stmt', 'Where_Stmt', 'Where_Construct',
                      'Forall_Construct', 'Forall_Stmt']

class Forall_Assignment_Stmt(Base): # R757
    """
    <forall-assignment-stmt> = <assignment-stmt>
                               | <pointer-assignment-stmt>
    """
    subclass_names = ['Assignment_Stmt', 'Pointer_Assignment_Stmt']

class End_Forall_Stmt(EndStmtBase): # R758
    """
    <end-forall-stmt> = END FORALL [ <forall-construct-name> ]
    """
    subclass_names = []
    use_names = ['Forall_Construct_Name']
    def match(string): return EndStmtBase.match('FORALL',Forall_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Forall_Stmt(StmtBase): # R759
    """
    <forall-stmt> = FORALL <forall-header> <forall-assignment-stmt>
    """
    subclass_names = []
    use_names = ['Forall_Header', 'Forall_Assignment_Stmt']

###############################################################################
############################### SECTION  8 ####################################
###############################################################################

class Block(BlockBase): # R801
    """
    block = [ <execution-part-construct> ]...
    """
    subclass_names = []
    use_names = ['Execution_Part_Construct']
    def match(string): return BlockBase.match(None, [Execution_Part_Construct], None, string)
    match = staticmethod(match)

class If_Construct(BlockBase): # R802
    """
    <if-construct> = <if-then-stmt>
                           <block>
                         [ <else-if-stmt>
                           <block>
                         ]...
                         [ <else-stmt>
                           <block>
                         ]
                         <end-if-stmt>
    """
    subclass_names = []
    use_names = ['If_Then_Stmt', 'Block', 'Else_If_Stmt', 'Else_Stmt', 'End_If_Stmt']

    def match(reader):
        content = []
        try:
            obj = If_Then_Stmt(reader)
        except NoMatchError:
            obj = None
        if obj is None: return
        content.append(obj)
        obj = Block(reader)
        if obj is None: return # todo: restore reader
        content.append(obj)
        while 1:
            try:
                obj = Else_If_Stmt(reader)
            except NoMatchError:
                obj = None
            if obj is not None:
                content.append(obj)
                obj = Block(reader)
                if obj is None: return # todo: restore reader
                content.append(obj)
                continue
            try:
                obj = Else_Stmt(reader)
            except NoMatchError:
                obj = None
            if obj is not None:
                content.append(obj)
                obj = Block(reader)
                if obj is None: return # todo: restore reader
                content.append(obj)
            break
        try:
            obj = End_If_Stmt(reader)
        except NoMatchError:
            obj = None
        if obj is None: return # todo: restore reader
        content.append(obj)
        return content,
    match = staticmethod(match)

    def tofortran(self, tab='', isfix=None):
        l = []
        start = self.content[0]
        end = self.content[-1]
        l.append(start.tofortran(tab=tab,isfix=isfix))
        for item in self.content[1:-1]:
            if isinstance(item, (Else_If_Stmt, Else_Stmt)):
                l.append(item.tofortran(tab=tab,isfix=isfix))
            else:
                l.append(item.tofortran(tab=tab+'  ',isfix=isfix))
        l.append(end.tofortran(tab=tab,isfix=isfix))
        return '\n'.join(l)


class If_Then_Stmt(StmtBase): # R803
    """
    <if-then-stmt> = [ <if-construct-name> : ] IF ( <scalar-logical-expr> ) THEN
    """
    subclass_names = []
    use_names = ['If_Construct_Name', 'Scalar_Logical_Expr']
    def match(string):
        if string[:2].upper()!='IF': return
        if string[-4:].upper()!='THEN': return
        line = string[2:-4].strip()
        if not line: return
        if line[0]+line[-1]!='()': return
        return Scalar_Logical_Expr(line[1:-1].strip()),
    match = staticmethod(match)
    def init(self, expr):
        self.expr = expr
        return
    def tostr(self): return 'IF (%s) THEN' % (self.expr)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.expr)

class Else_If_Stmt(StmtBase): # R804
    """
    <else-if-stmt> = ELSE IF ( <scalar-logical-expr> ) THEN [ <if-construct-name> ]
    """
    subclass_names = []
    use_names = ['Scalar_Logical_Expr', 'If_Construct_Name']

    def match(string):
        if string[:4].upper()!='ELSE': return
        line = string[4:].lstrip()
        if line[:2].upper()!='IF': return
        line = line[2:].lstrip()
        if not line.startswith('('): return
        i = line.rfind(')')
        if i==-1: return
        expr = line[1:i].strip()
        line = line[i+1:].lstrip()
        if line[:4].upper()!='THEN': return
        line = line[4:].lstrip()
        if line: return Scalar_Logical_Expr(expr), If_Construct_Name(line)
        return Scalar_Logical_Expr(expr), None
    match = staticmethod(match)
    def init(self, *args):
        self.expr, self.iflabel = args
        return
    def tostr(self):
        if self.iflabel is None:
            return 'ELSE IF (%s) THEN' % (self.expr)
        return 'ELSE IF (%s) THEN %s' % (self.expr, self.iflabel)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.expr, self.iflabel)

class Else_Stmt(StmtBase): # R805
    """
    <else-stmt> = ELSE [ <if-construct-name> ]
    """
    subclass_names = []
    use_names = ['If_Construct_Name']
    def match(string):
        if string[:4].upper()!='ELSE': return
        line = string[4:].lstrip()
        if line: return If_Construct_Name(line),
        return None,
    match = staticmethod(match)
    def init(self, iflabel):
        self.iflabel = iflabel
        return
    def tostr(self):
        if self.iflabel is None:
            return 'ELSE'
        return 'ELSE %s' % (self.expr, self.iflabel)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.iflabel)

class End_If_Stmt(EndStmtBase): # R806
    """
    <end-if-stmt> = END IF [ <if-construct-name> ]
    """
    subclass_names = []
    use_names = ['If_Construct_Name']
    def match(string): return EndStmtBase.match('IF',If_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class If_Stmt(StmtBase): # R807
    """
    <if-stmt> = IF ( <scalar-logical-expr> ) <action-stmt>
    """
    subclass_names = []
    use_names = ['Scalar_Logical_Expr', 'Action_Stmt_C802']
    def match(string):
        if string[:2].upper() != 'IF': return
        line, repmap = string_replace_map(string)
        line = line[2:].lstrip()
        if not line.startswith('('): return
        i = line.find(')')
        if i==-1: return
        expr = repmap(line[1:i].strip())
        stmt = repmap(line[i+1:].lstrip())
        return Scalar_Logical_Expr(expr), Action_Stmt_C802(stmt)
    match = staticmethod(match)
    def init(self, *args):
        self.expr, self.stmt = args
        return
    def tostr(self): return 'IF (%s) %s' % (self.expr, self.stmt)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.expr, self.stmt)
    
class Case_Construct(Base): # R808
    """
    <case-construct> = <select-case-stmt>
                           [ <case-stmt>
                             <block>
                           ]..
                           <end-select-stmt>
    """
    subclass_names = []
    use_names = ['Select_Case_Stmt', 'Case_Stmt', 'End_Select_Stmt']

class Select_Case_Stmt(StmtBase): # R809
    """
    <select-case-stmt> = [ <case-construct-name> : ] SELECT CASE ( <case-expr> )
    """
    subclass_names = []
    use_names = ['Case_Construct_Name', 'Case_Expr']

class Case_Stmt(StmtBase): # R810
    """
    <case-stmt> = CASE <case-selector> [ <case-construct-name> ]
    """
    subclass_names = []
    use_names = ['Case_Selector', 'Case_Construct_Name']

class End_Select_Stmt(EndStmtBase): # R811
    """
    <end-select-stmt> = END SELECT [ <case-construct-name> ]
    """
    subclass_names = []
    use_names = ['Case_Construct_Name']
    def match(string): return EndStmtBase.match('SELECT',Case_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Case_Expr(Base): # R812
    """
    <case-expr> = <scalar-int-expr>
                  | <scalar-char-expr>
                  | <scalar-logical-expr>
    """
    subclass_names = []
    subclass_names = ['Scalar_Int_Expr', 'Scalar_Char_Expr', 'Scalar_Logical_Expr']

class Case_Selector(Base): # R813
    """
    <case-selector> = ( <case-value-range-list> )
                      | DEFAULT
    """
    subclass_names = []
    use_names = ['Case_Value_Range_List']

class Case_Value_Range(SeparatorBase): # R814
    """
    <case-value-range> = <case-value>
                         | <case-value> :
                         | : <case-value>
                         | <case-value> : <case-value>
    """
    subclass_names = ['Case_Value']
    def match(string): return SeparatorBase.match(Case_Value, Case_Value, string)
    match = staticmethod(match)

class Case_Value(Base): # R815
    """
    <case-value> = <scalar-int-initialization-expr>
                   | <scalar-char-initialization-expr>
                   | <scalar-logical-initialization-expr>
    """
    subclass_names = ['Scalar_Int_Initialization_Expr', 'Scalar_Char_Initialization_Expr', 'Scalar_Logical_Initialization_Expr']


class Associate_Construct(Base): # R816
    """
    <associate-construct> = <associate-stmt>
                                <block>
                                <end-associate-stmt>
    """
    subclass_names = []
    use_names = ['Associate_Stmt', 'Block', 'End_Associate_Stmt']

class Associate_Stmt(StmtBase): # R817
    """
    <associate-stmt> = [ <associate-construct-name> : ] ASSOCIATE ( <association-list> )
    """
    subclass_names = []
    use_names = ['Associate_Construct_Name', 'Association_List']

class Association(BinaryOpBase): # R818
    """
    <association> = <associate-name> => <selector>
    """
    subclass_names = []
    use_names = ['Associate_Name', 'Selector']
    def match(string): return BinaryOpBase.match(Assiciate_Name, '=>', Selector, string)
    match = staticmethod(match)

class Selector(Base): # R819
    """
    <selector> = <expr>
                 | <variable>
    """
    subclass_names = ['Expr', 'Variable']

class End_Associate_Stmt(EndStmtBase): # R820
    """
    <end-associate-stmt> = END ASSOCIATE [ <associate-construct-name> ]
    """
    subclass_names = []
    use_names = ['Associate_Construct_Name']
    def match(string): return EndStmtBase.match('ASSOCIATE',Associate_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Select_Type_Construct(Base): # R821
    """
    <select-type-construct> = <select-type-stmt>
                                  [ <type-guard-stmt>
                                    <block>
                                  ]...
                                  <end-select-type-stmt>
    """
    subclass_names = []
    use_names = ['Select_Type_Stmt', 'Type_Guard_Stmt', 'Block', 'End_Select_Type_Stmt']

class Select_Type_Stmt(StmtBase): # R822
    """
    <select-type-stmt> = [ <select-construct-name> : ] SELECT TYPE ( [ <associate-name> => ] <selector> )
    """
    subclass_names = []
    use_names = ['Select_Construct_Name', 'Associate_Name', 'Selector']

class Type_Guard_Stmt(StmtBase): # R823
    """
    <type-guard-stmt> = TYPE IS ( <type-spec> ) [ <select-construct-name> ]
                        | CLASS IS ( <type-spec> ) [ <select-construct-name> ]
                        | CLASS DEFAULT [ <select-construct-name> ]
    """
    subclass_names = []
    use_names = ['Type_Spec', 'Select_Construct_Name']

class End_Select_Type_Stmt(EndStmtBase): # R824
    """
    <end-select-type-stmt> = END SELECT [ <select-construct-name> ]
    """
    subclass_names = []
    use_names = ['Select_Construct_Name']
    def match(string): return EndStmtBase.match('SELECT',Select_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Do_Construct(Base): # R825
    """
    <do-construct> = <block-do-construct>
                     | <nonblock-do-construct>
    """
    subclass_names = ['Block_Do_Construct', 'Nonblock_Do_Construct']

class Block_Do_Construct(BlockBase): # R826
    """
    <block-do-construct> = <do-stmt>
                               <do-block>
                               <end-do>
    """
    subclass_names = []
    use_names = ['Do_Stmt', 'Do_Block', 'End_Do']
    def match(reader):
        assert isinstance(reader,FortranReaderBase),`reader`
        content = []
        try:
            obj = Do_Stmt(reader)
        except NoMatchError:
            obj = None
        if obj is None: return
        content.append(obj)
        if isinstance(obj, Label_Do_Stmt):
            label = str(obj.dolabel)
            while 1:
                try:
                    obj = Execution_Part_Construct(reader)
                except NoMatchError:
                    obj = None
                if obj is None: break
                content.append(obj)
                if isinstance(obj, Continue_Stmt) and obj.item.label==label:
                    return content,
            return
            raise RuntimeError,'Expected continue stmt with specified label'
        else:
            obj = End_Do(reader)
            content.append(obj)
            raise NotImplementedError
        return content,
    match = staticmethod(match)

    def tofortran(self, tab='', isfix=None):
        if not isinstance(self.content[0], Label_Do_Stmt):
            return BlockBase.tofortran(tab, isfix)
        l = []
        start = self.content[0]
        end = self.content[-1]
        extra_tab = '  '
        l.append(start.tofortran(tab=tab,isfix=isfix))
        for item in self.content[1:-1]:
            l.append(item.tofortran(tab=tab+extra_tab,isfix=isfix))
        if len(self.content)>1:
            l.append(end.tofortran(tab=tab,isfix=isfix))
        return '\n'.join(l)

class Do_Stmt(Base): # R827
    """
    <do-stmt> = <label-do-stmt>
                | <nonlabel-do-stmt>
    """
    subclass_names = ['Label_Do_Stmt', 'Nonlabel_Do_Stmt']

class Label_Do_Stmt(StmtBase): # R828
    """
    <label-do-stmt> = [ <do-construct-name> : ] DO <label> [ <loop-control> ]
    """
    subclass_names = []
    use_names = ['Do_Construct_Name', 'Label', 'Loop_Control']
    def match(string):
        if string[:2].upper()!='DO': return
        line = string[2:].lstrip()
        m = pattern.label.match(line)
        if m is None: return
        label = m.group()
        line = line[m.end():].lstrip()
        if line: return Label(label), Loop_Control(line)
        return Label(label), None
    match = staticmethod(match)
    def init(self,*args):
        self.dolabel, self.loop_control = args
    def tostr(self):
        if self.loop_control is None: return 'DO %s' % (self.dolabel)
        return 'DO %s %s' % (self.dolabel, self.loop_control)
    
class Nonlabel_Do_Stmt(StmtBase): # R829
    """
    <nonlabel-do-stmt> = [ <do-construct-name> : ] DO [ <loop-control> ]
    """
    subclass_names = []
    use_names = ['Do_Construct_Name', 'Loop_Control']

class Loop_Control(Base): # R830
    """
    <loop-control> = [ , ] <do-variable> = <scalar-int-expr> , <scalar-int-expr> [ , <scalar-int-expr> ]
                     | [ , ] WHILE ( <scalar-logical-expr> )
    """
    subclass_names = []
    use_names = ['Do_Variable', 'Scalar_Int_Expr', 'Scalar_Logical_Expr']
    def match(string):
        if string.startswith(','):
            line, repmap = string_replace_map(string[1:].lstrip())
        else:
            line, repmap = string_replace_map(string)
        if line[:5].upper()=='WHILE' and line[5:].lstrip().startswith('('):
            l = line[5:].lstrip()
            i = l.find(')')
            if i!=-1 and i==len(l)-1:
                return Scalar_Logical_Expr(repmap(l[1:i].strip())),
        if line.count('=')!=1: return
        var,rhs = line.split('=')
        rhs = [s.strip() for s in rhs.lstrip().split(',')]
        if not 2<=len(rhs)<=3: return
        return Variable(repmap(var.rstrip())),map(Scalar_Int_Expr, map(repmap,rhs))
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
    def tostr(self):
        if len(self.items)==1: return ', WHILE (%s)' % (self.items[0])
        return ', %s = %s' % (self.items[0], ', '.join(map(str,self.items[1])))
    def torepr(self):
        if len(self.items)==1: return '%s(%r)' % (self.__class__.__name__, self.items[0])
        return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])
    
class Do_Variable(Base): # R831
    """
    <do-variable> = <scalar-int-variable>
    """
    subclass_names = ['Scalar_Int_Variable']

class Do_Block(Base): # R832
    """
    <do-block> = <block>
    """
    subclass_names = ['Block']

class End_Do(Base): # R833
    """
    <end-do> = <end-do-stmt>
               | <continue-stmt>
    """
    subclass_names = ['End_Do_Stmt', 'Continue_Stmt']

class End_Do_Stmt(EndStmtBase): # R834
    """
    <end-do-stmt> = END DO [ <do-construct-name> ]
    """
    subclass_names = []
    use_names = ['Do_Construct_Name']
    def match(string): return EndStmtBase.match('DO',Do_Construct_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Nonblock_Do_Construct(Base): # R835
    """
    <nonblock-do-stmt> = <action-term-do-construct>
                         | <outer-shared-do-construct>
    """
    subclass_names = ['Action_Term_Do_Construct', 'Outer_Shared_Do_Construct']

class Action_Term_Do_Construct(BlockBase): # R836
    """
    <action-term-do-construct> = <label-do-stmt>
                                     <do-body>
                                     <do-term-action-stmt>
    """
    subclass_names = []
    use_names = ['Label_Do_Stmt', 'Do_Body', 'Do_Term_Action_Stmt']
    def match(reader):
        content = []
        for cls in [Label_Do_Stmt, Do_Body, Do_Term_Action_Stmt]:
            obj = cls(reader)
            if obj is None: # todo: restore reader
                return
            content.append(obj)
        return content,
    match = staticmethod(match)

class Do_Body(BlockBase): # R837
    """
    <do-body> = [ <execution-part-construct> ]...
    """
    subclass_names = []
    use_names = ['Execution_Part_Construct']
    def match(string): return BlockBase.match(None, [Execution_Part_Construct], None, string)
    match = staticmethod(match)

class Do_Term_Action_Stmt(StmtBase): # R838
    """
    <do-term-action-stmt> = <action-stmt>
    C824: <do-term-action-stmt> shall not be <continue-stmt>, <goto-stmt>, <return-stmt>, <stop-stmt>,
                          <exit-stmt>, <cycle-stmt>, <end-function-stmt>, <end-subroutine-stmt>,
                          <end-program-stmt>, <arithmetic-if-stmt>
    """
    subclass_names = ['Action_Stmt_C824']

class Outer_Shared_Do_Construct(BlockBase): # R839
    """
    <outer-shared-do-construct> = <label-do-stmt>
                                      <do-body>
                                      <shared-term-do-construct>
    """
    subclass_names = []
    use_names = ['Label_Do_Stmt', 'Do_Body', 'Shared_Term_Do_Construct']
    def match(reader):
        content = []
        for cls in [Label_Do_Stmt, Do_Body, Shared_Term_Do_Construct]:
            obj = cls(reader)
            if obj is None: # todo: restore reader
                return
            content.append(obj)
        return content,
    match = staticmethod(match)

class Shared_Term_Do_Construct(Base): # R840
    """
    <shared-term-do-construct> = <outer-shared-do-construct>
                                 | <inner-shared-do-construct>
    """
    subclass_names = ['Outer_Shared_Do_Construct', 'Inner_Shared_Do_Construct']

class Inner_Shared_Do_Construct(BlockBase): # R841
    """
    <inner-shared-do-construct> = <label-do-stmt>
                                      <do-body>
                                      <do-term-shared-stmt>
    """
    subclass_names = []
    use_names = ['Label_Do_Stmt', 'Do_Body', 'Do_Term_Shared_Stmt']

    def match(reader):
        content = []
        for cls in [Label_Do_Stmt, Do_Body, Do_Term_Shared_Stmt]:
            obj = cls(reader)
            if obj is None: # todo: restore reader
                return
            content.append(obj)
        return content,
    match = staticmethod(match)

class Do_Term_Shared_Stmt(StmtBase): # R842
    """
    <do-term-shared-stmt> = <action-stmt>
    C826: see C824 above.
    """
    subclass_names = ['Action_Stmt']

class Cycle_Stmt(StmtBase, WORDClsBase): # R843
    """
    <cycle-stmt> = CYCLE [ <do-construct-name> ]
    """
    subclass_names = []
    use_names = ['Do_Construct_Name']
    def match(string): return WORDClsBase.match('CYCLE', Do_Construct_Name, string)
    match = staticmethod(match)

class Exit_Stmt(StmtBase, WORDClsBase): # R844
    """
    <exit-stmt> = EXIT [ <do-construct-name> ]
    """
    subclass_names = []
    use_names = ['Do_Construct_Name']
    def match(string): return WORDClsBase.match('EXIT', Do_Construct_Name, string)
    match = staticmethod(match)

class Goto_Stmt(StmtBase): # R845
    """
    <goto-stmt> = GO TO <label>
    """
    subclass_names = []
    use_names = ['Label']
    def match(string):
        if string[:2].upper() != 'GO': return
        line = string[2:].lstrip()
        if line[:2].upper() != 'TO': return
        return Label(line[2:].lstrip()),
    match = staticmethod(match)
    def init(self, value): self.value = value
    def tostr(self): return 'GO TO %s' % (self.value)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.value)

class Computed_Goto_Stmt(StmtBase): # R846
    """
    <computed-goto-stmt> = GO TO ( <label-list> ) [ , ] <scalar-int-expr>
    """
    subclass_names = []
    use_names = ['Label_List', 'Scalar_Int_Expr']

class Arithmetic_If_Stmt(StmtBase): # R847
    """
    <arithmetic-if-stmt> = IF ( <scalar-numeric-expr> ) <label> , <label> , <label>
    """
    subclass_names = []
    use_names = ['Scalar_Numeric_Expr', 'Label']
    def match(string):
        if string[:2].upper() != 'IF': return
        line = string[2:].lstrip()
        if not line.startswith('('): return
        i = line.rfind(')')
        if i==-1: return
        labels = line[i+1:].lstrip().split(',')
        if len(labels) != 3: return
        labels = [Label(l.strip()) for l in labels]
        return Scalar_Numeric_Expr(line[1:i].strip()), labels
    match = staticmethod(match)
    def init(self, *args):
        self.expr, self.labels = args
        return
    def tostr(self): return 'IF (%s) %s' % (self.expr, ', '.join(map(str,self.labels)))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.expr, ', '.join(map(repr,self.labels)))

class Continue_Stmt(StmtBase, STRINGBase): # R848
    """
    <continue-stmt> = CONTINUE
    """
    subclass_names = []
    def match(string): return STRINGBase.match('CONTINUE', string)
    match = staticmethod(match)


class Stop_Stmt(StmtBase, WORDClsBase): # R849
    """
    <stop-stmt> = STOP [ <stop-code> ]
    """
    subclass_names = []
    use_names = ['Stop_Code']
    def match(string): return WORDClsBase.match('STOP', Stop_Code, string)
    match = staticmethod(match)

class Stop_Code(StringBase): # R850
    """
    <stop-code> = <scalar-char-constant>
                  | <digit> [ <digit> [ <digit> [ <digit> [ <digit> ] ] ] ]
    """
    subclass_names = ['Scalar_Char_Constant']
    def match(string): return StringBase.match(pattern.abs_label, string)
    match = staticmethod(match)


###############################################################################
############################### SECTION  9 ####################################
###############################################################################

class Io_Unit(StringBase): # R901
    """
    <io-unit> = <file-unit-number>
                | *
                | <internal-file-variable>
    """
    subclass_names = ['File_Unit_Number', 'Internal_File_Variable']
    def match(string): return StringBase.match('*', string)
    match = staticmethod(match)

class File_Unit_Number(Base): # R902
    """
    <file-unit-number> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Internal_File_Variable(Base): # R903
    """
    <internal-file-variable> = <char-variable>
    C901: <char-variable> shall not be an array section with a vector subscript.
    """
    subclass_names = ['Char_Variable']

class Open_Stmt(StmtBase): # R904
    """
    <open-stmt> = OPEN ( <connect-spec-list> )
    """
    subclass_names = []
    use_names = ['Connect_Spec_List']

class Connect_Spec(Base): # R905
    """
    <connect-spec> = [ UNIT = ] <file-unit-number>
                     | ACCESS = <scalar-default-char-expr>
                     | ACTION = <scalar-default-char-expr>
                     | ASYNCHRONOUS = <scalar-default-char-expr>
                     | BLANK = <scalar-default-char-expr>
                     | DECIMAL = <scalar-default-char-expr>
                     | DELIM = <scalar-default-char-expr>
                     | ENCODING = <scalar-default-char-expr>
                     | ERR = <label>
                     | FILE = <file-name-expr>
                     | FORM = <scalar-default-char-expr>
                     | IOMSG = <iomsg-variable>
                     | IOSTAT = <scalar-int-variable>
                     | PAD = <scalar-default-char-expr>
                     | POSITION = <scalar-default-char-expr>
                     | RECL = <scalar-int-expr>
                     | ROUND = <scalar-default-char-expr>
                     | SIGN = <scalar-default-char-expr>
                     | STATUS = <scalar-default-char-expr>
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Scalar_Default_Char_Expr', 'Label', 'File_Name_Expr', 'Iomsg_Variable',
                 'Scalar_Int_Expr', 'Scalar_Int_Variable']

class File_Name_Expr(Base): # R906
    """
    <file-name-expr> = <scalar-default-char-expr>
    """
    subclass_names = ['Scalar_Default_Char_Expr']

class Iomsg_Variable(Base): # R907
    """
    <iomsg-variable> = <scalar-default-char-variable>
    """
    subclass_names = ['Scalar_Default_Char_Variable']

class Close_Stmt(StmtBase): # R908
    """
    <close-stmt> = CLOSE ( <close-spec-list> )
    """
    subclass_names = []
    use_names = ['Close_Spec_List']

class Close_Spec(Base): # R909
    """
    <close-spec> = [ UNIT = ] <file-unit-number>
                   | IOSTAT = <scalar-int-variable>
                   | IOMSG = <iomsg-variable>
                   | ERR = <label>
                   | STATUS = <scalar-default-char-expr>
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Scalar_Default_Char_Expr', 'Label', 'Iomsg_Variable',
                 'Scalar_Int_Variable']

class Read_Stmt(StmtBase): # R910
    """
    <read-stmt> = READ ( <io-control-spec-list> ) [ <input-item-list> ]
                  | READ <format> [ , <input-item-list> ]
    """
    subclass_names = []
    use_names = ['Io_Control_Spec_List', 'Input_Item_List', 'Format']

class Write_Stmt(StmtBase): # R911
    """
    <write-stmt> = WRITE ( <io-control-spec-list> ) [ <output-item-list> ]
    """
    subclass_names = []
    use_names = ['Io_Control_Spec_List', 'Output_Item_List']
    def match(string):
        if string[:5].upper()!='WRITE': return
        line = string[5:].lstrip()
        if not line.startswith('('): return
        line, repmap = string_replace_map(line)
        i = line.find(')')
        if i==-1: return
        l = line[1:i].strip()
        if not l: return
        l = repmap(l)
        if i==len(line)-1:
            return Io_Control_Spec_List(l),None
        return Io_Control_Spec_List(l), Output_Item_List(repmap(line[i+1:].lstrip()))
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        if self.items[1] is None: return 'WRITE(%s)' % (self.items[0])
        return 'WRITE(%s) %s' % tuple(self.items)
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))
    
class Print_Stmt(StmtBase): # R912
    """
    <print-stmt> = PRINT <format> [ , <output-item-list> ]
    """
    subclass_names = []
    use_names = ['Format', 'Output_Item_List']
    def match(string):
        if string[:5].upper()!='PRINT': return
        line = string[5:]
        if not line: return
        c = line[0].upper()
        if 'A'<=c<='Z' or c=='_' or '0'<=c<='9': return
        line, repmap = string_replace_map(line.lstrip())
        i = line.find(',')
        if i==-1: return Format(repmap(line)), None
        l = repmap(line[i+1:].lstrip())
        if not l: return
        return Format(repmap(line[:i].rstrip())), Output_Item_List(l)
    match = staticmethod(match)
    def init(self, *args):
        self.items = args
        return
    def tostr(self):
        if self.items[1] is None: return 'PRINT %s' % (self.items[0])
        return 'PRINT %s, %s' % tuple(self.items)
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))
    
class Io_Control_Spec_List(SequenceBase): # R913-list
    """
    <io-control-spec-list> is a list taking into account C910, C917, C918
    """
    subclass_names = []
    use_names = ['Io_Control_Spec']
    def match(string):
        line, repmap = string_replace_map(string)
        splitted = line.split(',')
        if not splitted: return
        lst = []
        for i in range(len(splitted)):
            p = splitted[i].strip()
            if i==0:
                if '=' not in p: p = 'UNIT=%s' % (repmap(p))
                else: p = repmap(p)
            elif i==1:
                if '=' not in p:
                    p = repmap(p)
                    try:
                        f = Format(p)
                        # todo: make sure that f is char-expr, if not, raise NoMatchError
                        p = 'FMT=%s' % (Format(p))
                    except NoMatchError:
                        p = 'NML=%s' % (Namelist_Group_Name(p))
                else:
                    p = repmap(p)
            else:
                p = repmap(p)
            lst.append(Io_Control_Spec(p))
        return ',', tuple(lst)
    match = staticmethod(match)

class Io_Control_Spec(KeywordValueBase): # R913
    """
    <io-control-spec> = [ UNIT = ] <io-unit>
                        | [ FMT = ] <format>
                        | [ NML = ] <namelist-group-name>
                        | ADVANCE = <scalar-default-char-expr>
                        | ASYNCHRONOUS = <scalar-char-initialization-expr>
                        | BLANK = <scalar-default-char-expr>
                        | DECIMAL = <scalar-default-char-expr>
                        | DELIM = <scalar-default-char-expr>
                        | END = <label>
                        | EOR = <label>
                        | ERR = <label>
                        | ID = <scalar-int-variable>
                        | IOMSG = <iomsg-variable>
                        | IOSTAT = <scalar-int-variable>
                        | PAD = <scalar-default-char-expr>
                        | POS = <scalar-int-expr>
                        | REC = <scalar-int-expr>
                        | ROUND = <scalar-default-char-expr>
                        | SIGN = <scalar-default-char-expr>
                        | SIZE = <scalar-int-variable>
    """
    subclass_names = []
    use_names = ['Io_Unit', 'Format', 'Namelist_Group_Name', 'Scalar_Default_Char_Expr',
                 'Scalar_Char_Initialization_Expr', 'Label', 'Scalar_Int_Variable',
                 'Iomsg_Variable', 'Scalar_Int_Expr']
    def match(string):
        for (k,v) in [\
            (['ADVANCE', 'BLANK', 'DECIMAL', 'DELIM', 'PAD', 'ROUND', 'SIGN'], Scalar_Default_Char_Expr),
            ('ASYNCHRONOUS', Scalar_Char_Initialization_Expr),
            (['END','EOR','ERR'], Label),
            (['ID','IOSTAT','SIZE'], Scalar_Int_Variable),
            ('IOMSG', Iomsg_Variable),
            (['POS', 'REC'], Scalar_Int_Expr),
            ('UNIT', Io_Unit),
            ('FMT', Format),
            ('NML', Namelist_Group_Name)
            ]:
            try:
                obj = KeywordValueBase.match(k, v, string, upper_lhs = True)
            except NoMatchError:
                obj = None
            if obj is not None: return obj
        return
    match = staticmethod(match)

class Format(StringBase): # R914
    """
    <format> = <default-char-expr>
               | <label>
               | *
    """
    subclass_names = ['Label', 'Default_Char_Expr']
    def match(string): return StringBase.match('*', string)
    match = staticmethod(match)

class Input_Item(Base): # R915
    """
    <input-item> = <variable>
                   | <io-implied-do>
    """
    subclass_names = ['Variable', 'Io_Implied_Do']

class Output_Item(Base): # R916
    """
    <output-item> = <expr>
                    | <io-implied-do>
    """
    subclass_names = ['Expr', 'Io_Implied_Do']

class Io_Implied_Do(Base): # R917
    """
    <io-implied-do> = ( <io-implied-do-object-list> , <io-implied-do-control> )
    """
    subclass_names = []
    use_names = ['Io_Implied_Do_Object_List', 'Io_Implied_Do_Control']

class Io_Implied_Do_Object(Base): # R918
    """
    <io-implied-do-object> = <input-item>
                             | <output-item>
    """
    subclass_names = ['Input_Item', 'Output_Item']

class Io_Implied_Do_Control(Base): # R919
    """
    <io-implied-do-control> = <do-variable> = <scalar-int-expr> , <scalar-int-expr> [ , <scalar-int-expr> ]
    """
    subclass_names = []
    use_names = ['Do_Variable', 'Scalar_Int_Expr']

class Dtv_Type_Spec(Base): # R920
    """
    <dtv-type-spec> = TYPE ( <derived-type-spec> )
                      | CLASS ( <derived-type-spec> )
    """
    subclass_names = []
    use_names = ['Derived_Type_Spec']

class Wait_Stmt(StmtBase, CALLBase): # R921
    """
    <wait-stmt> = WAIT ( <wait-spec-list> )
    """
    subclass_names = []
    use_names = ['Wait_Spec_List']
    def match(string): return CALLBase.match('WAIT', Wait_Spec_List, string, require_rhs=True)
    match = staticmethod(match)

class Wait_Spec(KeywordValueBase): # R922
    """
    <wait-spec> = [ UNIT = ] <file-unit-number>
                  | END = <label>
                  | EOR = <label>
                  | ERR = <label>
                  | ID = <scalar-int-expr>
                  | IOMSG = <iomsg-variable>
                  | IOSTAT = <scalar-int-variable>
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Label', 'Scalar_Int_Expr', 'Iomsg_Variable', 'Scalar_Int_Variable']
    def match(string):
        for (k,v) in [\
            (['END','EOR','ERR'], Label),
            ('IOSTAT', Scalar_Int_Variable),
            ('IOMSG', Iomsg_Variable),
            ('ID', Scalar_Int_Expr),
            ('UNIT', File_Unit_Number),
            ]:
            try:
                obj = KeywordValueBase.match(k, v, string, upper_lhs = True)
            except NoMatchError:
                obj = None
            if obj is not None: return obj
        return 'UNIT', File_Unit_Number(string)

    match = staticmethod(match)

class Backspace_Stmt(StmtBase): # R923
    """
    <backspace-stmt> = BACKSPACE <file-unit-number>
                       | BACKSPACE ( <position-spec-list> )
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Position_Spec_List']

class Endfile_Stmt(StmtBase): # R924
    """
    <endfile-stmt> = ENDFILE <file-unit-number>
                     | ENDFILE ( <position-spec-list> )
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Position_Spec_List']

class Rewind_Stmt(StmtBase): # R925
    """
    <rewind-stmt> = REWIND <file-unit-number>
                    | REWIND ( <position-spec-list> )
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Position_Spec_List']

class Position_Spec(Base): # R926
    """
    <position-spec> = [ UNIT = ] <file-unit-number>
                      | IOMSG = <iomsg-variable>
                      | IOSTAT = <scalar-int-variable>
                      | ERR = <label>
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Iomsg_Variable', 'Scalar_Int_Variable', 'Label']

class Flush_Stmt(StmtBase): # R927
    """
    <flush-stmt> = FLUSH <file-unit-number>
                    | FLUSH ( <position-spec-list> )
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Position_Spec_List']

class Flush_Spec(Base): # R928
    """
    <flush-spec> = [ UNIT = ] <file-unit-number>
                   | IOMSG = <iomsg-variable>
                   | IOSTAT = <scalar-int-variable>
                   | ERR = <label>
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'Iomsg_Variable', 'Scalar_Int_Variable', 'Label']

class Inquire_Stmt(StmtBase): # R929
    """
    <inquire-stmt> = INQUIRE ( <inquire-spec-list> )
                     | INQUIRE ( IOLENGTH = <scalar-int-variable> ) <output-item-list>
    """
    subclass_names = []
    use_names = ['Inquire_Spec_List', 'Scalar_Int_Variable', 'Output_Item_List']

class Inquire_Spec(Base): # R930
    """
    <inquire-spec> = [ UNIT = ] <file-unit-number>
                     | FILE = <file-name-expr>
                     | ACCESS = <scalar-default-char-variable>
                     | ACTION = <scalar-default-char-variable>
                     | ASYNCHRONOUS = <scalar-default-char-variable>
                     | BLANK = <scalar-default-char-variable>
                     | DECIMAL = <scalar-default-char-variable>
                     | DELIM = <scalar-default-char-variable>
                     | DIRECT = <scalar-default-char-variable>
                     | ENCODING = <scalar-default-char-variable>
                     | ERR = <label>
                     | EXIST = <scalar-default-logical-variable>
                     | FORM = <scalar-default-char-variable>
                     | FORMATTED = <scalar-default-char-variable>
                     | ID = <scalar-int-expr>
                     | IOMSG = <iomsg-variable>
                     | IOSTAT = <scalar-int-variable>
                     | NAME = <scalar-default-char-variable>
                     | NAMED = <scalar-default-logical-variable>
                     | NEXTREC = <scalar-int-variable>
                     | NUMBER = <scalar-int-variable>
                     | OPENED = <scalar-default-logical-variable>
                     | PAD = <scalar-default-char-variable>
                     | PENDING = <scalar-default-logical-variable>
                     | POS = <scalar-int-variable>
                     | POSITION = <scalar-default-char-variable>
                     | READ = <scalar-default-char-variable>
                     | READWRITE = <scalar-default-char-variable>
                     | RECL = <scalar-int-variable>
                     | ROUND = <scalar-default-char-variable>
                     | SEQUENTIAL = <scalar-default-char-variable>
                     | SIGN = <scalar-default-char-variable>
                     | SIZE = <scalar-int-variable>
                     | STREAM = <scalar-default-char-variable>
                     | UNFORMATTED = <scalar-default-char-variable>
                     | WRITE = <scalar-default-char-variable>
    """
    subclass_names = []
    use_names = ['File_Unit_Number', 'File_Name_Expr', 'Scalar_Default_Char_Variable',
                 'Scalar_Default_Logical_Variable', 'Scalar_Int_Variable', 'Scalar_Int_Expr',
                 'Label', 'Iomsg_Variable']


###############################################################################
############################### SECTION 10 ####################################
###############################################################################

class Format_Stmt(StmtBase): # R1001
    """
    <format-stmt> = FORMAT <format-specification>
    """
    subclass_names = []
    use_names = ['Format_Specification']

class Format_Specification(Base): # R1002
    """
    <format-specification> = ( [ <format-item-list> ] )
    """
    subclass_names = []
    use_names = ['Format_Item_List']

class Format_Item(Base): # R1003
    """
    <format-item> = [ <r> ] <data-edit-desc>
                    | <control-edit-desc>
                    | <char-string-edit-desc>
                    | [ <r> ] ( <format-item-list> )
    """
    subclass_names = ['Control_Edit_Desc', 'Char_String_Edit_Desc']
    use_names = ['R', 'Format_Item_List']

class R(Base): # R1004
    """
    <r> = <int-literal-constant>
    <r> shall be positive and without kind parameter specified.
    """
    subclass_names = ['Int_Literal_Constant']

class Data_Edit_Desc(Base): # R1005
    """
    <data-edit-desc> = I <w> [ . <m> ]
                       | B <w> [ . <m> ]
                       | O <w> [ . <m> ]
                       | Z <w> [ . <m> ]
                       | F <w> . <d>
                       | E <w> . <d> [ E <e> ]
                       | EN <w> . <d> [ E <e> ]
                       | ES <w> . <d> [ E <e>]
                       | G <w> . <d> [ E <e> ]
                       | L <w>
                       | A [ <w> ]
                       | D <w> . <d>
                       | DT [ <char-literal-constant> ] [ ( <v-list> ) ]
    """
    subclass_names = []
    use_names = ['W', 'M', 'D', 'E', 'Char_Literal_Constant', 'V_List']

class W(Base): # R1006
    """
    <w> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']

class M(Base): # R1007
    """
    <m> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']

class D(Base): # R1008
    """
    <d> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']

class E(Base): # R1009
    """
    <e> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']

class V(Base): # R1010
    """
    <v> = <signed-int-literal-constant>
    """
    subclass_names = ['Signed_Int_Literal_Constant']

class Control_Edit_Desc(Base): # R1011
    """
    <control-edit-desc> = <position-edit-desc>
                          | [ <r> ] /
                          | :
                          | <sign-edit-desc>
                          | <k> P
                          | <blank-interp-edit-desc>
                          | <round-edit-desc>
                          | <decimal-edit-desc>
    """
    subclass_names = ['Position_Edit_Desc', 'Sign_Edit_Desc', 'Blank_Interp_Edit_Desc', 'Round_Edit_Desc',
                      'Decimal_Edit_Desc']
    use_names = ['R', 'K']

class K(Base): # R1012
    """
    <k> = <signed-int-literal-constant>
    """
    subclass_names = ['Signed_Int_Literal_Constant']

class Position_Edit_Desc(Base): # R1013
    """
    <position-edit-desc> = T <n>
                           | TL <n>
                           | TR <n>
                           | <n> X
    """
    subclass_names = []
    use_names = ['N']

class N(Base): # R1014
    """
    <n> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']

class Sign_Edit_Desc(STRINGBase): # R1015
    """
    <sign-edit-desc> = SS
                       | SP
                       | S
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['SS','SP','S'], string)
    match = staticmethod(match)
    
class Blank_Interp_Edit_Desc(STRINGBase): # R1016
    """
    <blank-interp-edit-desc> = BN
                               | BZ
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['BN','BZ',], string)
    match = staticmethod(match)

class Round_Edit_Desc(STRINGBase): # R1017
    """
    <round-edit-desc> = RU
                        | RD
                        | RZ
                        | RN
                        | RC
                        | RP
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['RU','RD','RZ','RN','RC','RP'], string)
    match = staticmethod(match)

class Decimal_Edit_Desc(STRINGBase): # R1018
    """
    <decimal-edit-desc> = DC
                          | DP
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['DC','DP'], string)
    match = staticmethod(match)

class Char_String_Edit_Desc(Base): # R1019
    """
    <char-string-edit-desc> = <char-literal-constant>
    """
    subclass_names = ['Char_Literal_Constant']

###############################################################################
############################### SECTION 11 ####################################
###############################################################################

class Main_Program(Base): # R1101
    """
    <main-program> = [ <program-stmt> ]
                         [ <specification-part> ]
                         [ <execution-part> ]
                         [ <internal-subprogram-part> ]
                         <end-program-stmt>
    """
    subclass_names = []
    use_names = ['Program_Stmt', 'Specification_Part', 'Execution_Part', 'Internal_Subprogram_Part',
                 'End_Program_Stmt']

class Program_Stmt(StmtBase): # R1102
    """
    <program-stmt> = PROGRAM <program-name>
    """
    subclass_names = []
    use_names = ['Program_Name']

class End_Program_Stmt(EndStmtBase): # R1103
    """
    <end-program-stmt> = END [ PROGRAM [ <program-name> ] ]
    """
    subclass_names = []
    use_names = ['Program_Name']
    def match(string): return EndStmtBase.match('PROGRAM',Program_Name, string)
    match = staticmethod(match)

class Module(Base): # R1104
    """
    <module> = <module-stmt>
                   [ <specification-part> ]
                   [ <module-subprogram-part> ]
                   <end-module-stmt>
    """
    subclass_names = []
    use_names = ['Module_Stmt', 'Specification_Part', 'Module_Subprogram_Part', 'End_Module_Stmt']

class Module_Stmt(StmtBase): # R1105
    """
    <module-stmt> = MODULE <module-name>
    """
    subclass_names = []
    use_names = ['Module_Name']

class End_Module_Stmt(EndStmtBase): # R1106
    """
    <end-module-stmt> = END [ MODULE [ <module-name> ] ]
    """
    subclass_names = []
    use_names = ['Module_Name']
    def match(string): return EndStmtBase.match('MODULE',Module_Name, string, require_stmt_type=True)
    match = staticmethod(match)

class Module_Subprogram_Part(Base): # R1107
    """
    <module-subprogram-part> = <contains-stmt>
                                   <module-subprogram>
                                   [ <module-subprogram> ]...
    """
    subclass_names = []
    use_names = ['Contains_Stmt', 'Module_Subprogram']

class Module_Subprogram(Base): # R1108
    """
    <module-subprogram> = <function-subprogram>
                          | <subroutine-subprogram>
    """
    subclass_names = ['Function_Subprogram', 'Subroutine_Subprogram']

class Use_Stmt(StmtBase): # R1109
    """
    <use-stmt> = USE [ [ , <module-nature> ] :: ] <module-name> [ , <rename-list> ]
                 | USE [ [ , <module-nature> ] :: ] <module-name> , ONLY: [ <only-list> ]
    """
    subclass_names = []
    use_names = ['Module_Nature', 'Module_Name', 'Rename_List', 'Only_List']

class Module_Nature(STRINGBase): # R1110
    """
    <module-nature> = INTRINSIC
                      | NON_INTRINSIC
    """
    subclass_names = []
    def match(string): return STRINGBase.match(['INTRINSIC','NON_INTRINSIC'], string)
    match = staticmethod(match)

class Rename(Base): # R1111
    """
    <rename> = <local-name> => <use-name>
               | OPERATOR(<local-defined-operator>) => OPERATOR(<use-defined-operator>)
    """
    subclass_names = []
    use_names = ['Local_Name', 'Use_Name', 'Local_Defined_Operator', 'Use_Defined_Operator']

class Only(Base): # R1112
    """
    <only> = <generic-spec>
             | <only-use-name>
             | <rename>
    """
    subclass_names = ['Generic_Spec', 'Only_Use_Name', 'Rename']

class Only_Use_Name(Base): # R1113
    """
    <only-use-name> = <name>
    """
    subclass_names = ['Name']

class Local_Defined_Operator(Base): # R1114
    """
    <local-defined-operator> = <defined-unary-op>
                               | <defined-binary-op>
    """
    subclass_names = ['Defined_Unary_Op', 'Defined_Binary_Op']

class Use_Defined_Operator(Base): # R1115
    """
    <use-defined-operator> = <defined-unary-op>
                             | <defined-binary-op>
    """
    subclass_names = ['Defined_Unary_Op', 'Defined_Binary_Op']

class Block_Data(Base): # R1116
    """
    <block-data> = <block-data-stmt>
                       [ <specification-part> ]
                       <end-block-data-stmt>
    """
    subclass_names = []
    use_names = ['Block_Data_Stmt', 'Specification_Part', 'End_Block_Data_Stmt']

class Block_Data_Stmt(StmtBase): # R1117
    """
    <block-data-stmt> = BLOCK DATA [ <block-data-name> ]
    """
    subclass_names = []
    use_names = ['Block_Data_Name']

class End_Block_Data_Stmt(EndStmtBase): # R1118
    """
    <end-block-data-stmt> = END [ BLOCK DATA [ <block-data-name> ] ]
    """
    subclass_names = []
    use_names = ['Block_Data_Name']
    def match(string): return EndStmtBase.match('BLOCK DATA',Block_Data_Name, string)
    match = staticmethod(match)

###############################################################################
############################### SECTION 12 ####################################
###############################################################################


class Interface_Block(Base): # R1201
    """
    <interface-block> = <interface-stmt>
                            [ <interface-specification> ]...
                            <end-interface-stmt>
    """
    subclass_names = []
    use_names = ['Interface_Stmt', 'Interface_Specification', 'End_Interface_Stmt']

class Interface_Specification(Base): # R1202
    """
    <interface-specification> = <interface-body>
                                | <procedure-stmt>
    """
    subclass_names = ['Interface_Body', 'Procedure_Stmt']

class Interface_Stmt(StmtBase): # R1203
    """
    <interface-stmt> = INTERFACE [ <generic-spec> ]
                       | ABSTRACT INTERFACE
    """
    subclass_names = []
    use_names = ['Generic_Spec']

class End_Interface_Stmt(EndStmtBase): # R1204
    """
    <end-interface-stmt> = END INTERFACE [ <generic-spec> ]
    """
    subclass_names = []
    use_names = ['Generic_Spec']
    def match(string): return EndStmtBase.match('INTERFACE',Generic_Spec, string, require_stmt_type=True)
    match = staticmethod(match)

class Interface_Body(Base): # R1205
    """
    <interface-body> = <function-stmt>
                           [ <specification-part> ]
                           <end-function-stmt>
                       | <subroutine-stmt>
                           [ <specification-part> ]
                           <end-subroutine-stmt>
    """
    subclass_names = []
    use_names = ['Function_Stmt', 'Specification_Part', 'Subroutine_Stmt', 'End_Function_Stmt', 'End_Subroutine_Stmt']

class Procedure_Stmt(StmtBase): # R1206
    """
    <procedure-stmt> = [ MODULE ] PROCEDURE <procedure-name-list>
    """
    subclass_names = []
    use_names = ['Procedure_Name_List']

class Generic_Spec(Base): # R1207
    """
    <generic-spec> = <generic-name>
                     | OPERATOR ( <defined-operator> )
                     | ASSIGNMENT ( = )
                     | <dtio-generic-spec>
    """
    subclass_names = ['Generic_Name', 'Dtio_Generic_Spec']
    use_names = ['Defined_Operator']

class Dtio_Generic_Spec(Base): # R1208
    """
    <dtio-generic-spec> = READ ( FORMATTED )
                          | READ ( UNFORMATTED )
                          | WRITE ( FORMATTED )
                          | WRITE ( UNFORMATTED )
    """
    subclass_names = []

class Import_Stmt(StmtBase): # R1209
    """
    <import-stmt> = IMPORT [ :: ] <import-name-list>
    """
    subclass_names = []
    use_names = ['Import_Name_List']
    def match(string):
        start = string[:6].upper()
        if start != 'IMPORT': return
        line = string[6:].lstrip()
        if line.startswith('::'):
            line = line[2:].lstrip()
        return Import_Name_List(line),
    match = staticmethod(match)
    def init(self, names):
        self.names = names
        return
    def tostr(self): return 'IMPORT :: %s' % (self.names)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.names)

class External_Stmt(StmtBase): # R1210
    """
    <external-stmt> = EXTERNAL [ :: ] <external-name-list>
    """
    subclass_names = []
    use_names = ['External_Name_List']
    def match(string):
        if string[:8].upper() != 'EXTERNAL': return
        line = string[8:].lstrip()
        if line.startswith('::'): line = line[2:].lstrip()
        return External_Name_List(line),
    match = staticmethod(match)
    def init(self, names):
        self.names = names
        return
    def tostr(self): return 'EXTERNAL :: %s' % (self.names)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.names)

class Procedure_Declaration_Stmt(StmtBase): # R1211
    """
    <procedure-declaration-stmt> = PROCEDURE ( [ <proc-interface> ] ) [ [ , <proc-attr-spec> ]... :: ] <proc-decl-list>
    """
    subclass_names = []
    use_names = ['Proc_Interface', 'Proc_Attr_Spec', 'Proc_Decl_List']

class Proc_Interface(Base): # R1212
    """
    <proc-interface> = <interface-name>
                       | <declaration-type-spec>
    """
    subclass_names = ['Interface_Name', 'Declaration_Type_Spec']

class Proc_Attr_Spec(Base): # R1213
    """
    <proc-attr-spec> = <access-spec>
                       | <proc-language-binding-spec>
                       | INTENT ( <intent-spec> )
                       | OPTIONAL
                       | SAVE
    """
    subclass_names = ['Access_Spec', 'Proc_Language_Binding_Spec']
    use_names = ['Intent_Spec']

class Proc_Decl(Base): # R1214
    """
    <proc-decl> = <procedure-entity-name> [ => <null-init> ]
    """
    subclass_names = ['Procedure_Entity_Name']
    use_names = ['Null_Init']

class Interface_Name(Base): # R1215
    """
    <interface-name> = <name>
    """
    subclass_names = ['Name']

class Intrinsic_Stmt(StmtBase): # R1216
    """
    <intrinsic-stmt> = INTRINSIC [ :: ] <intrinsic-procedure-name-list>
    """
    subclass_names = []
    use_names = ['Intrinsic_Procedure_Name_List']

class Function_Reference(CallBase): # R1217
    """
    <function-reference> = <procedure-designator> ( [ <actual-arg-spec-list> ] )
    """
    subclass_names = []
    use_names = ['Procedure_Designator','Actual_Arg_Spec_List']
    def match(string):
        return CallBase.match(Procedure_Designator, Actual_Arg_Spec_List, string)
    match = staticmethod(match)

class Call_Stmt(StmtBase): # R1218
    """
    <call-stmt> = CALL <procedure-designator> [ ( [ <actual-arg-spec-list> ] ) ]
    """
    subclass_names = []
    use_names = ['Procedure_Designator', 'Actual_Arg_Spec_List']
    def match(string):
        if string[:4].upper()!='CALL': return
        line, repmap = string_replace_map(string[4:].lstrip())
        if line.endswith(')'):
            i = line.rfind('(')
            if i==-1: return
            args = repmap(line[i+1:-1].strip())
            if args:
                return Procedure_Designator(repmap(line[:i].rstrip())),Actual_Arg_Spec_List(args)
            return Procedure_Designator(repmap(line[:i].rstrip())),None
        return Procedure_Designator(string[4:].lstrip()),None
    match = staticmethod(match)
    def init(self, *args):
        self.designator, self.args = args
    def tostr(self):
        if self.args is None: return 'CALL %s' % (self.designator)
        return 'CALL %s(%s)' % (self.designator, self.args)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.designator, self.args)

class Procedure_Designator(BinaryOpBase): # R1219
    """
    <procedure-designator> = <procedure-name>
                             | <proc-component-ref>
                             | <data-ref> % <binding-name>
    """
    subclass_names = ['Procedure_Name','Proc_Component_Ref']
    use_names = ['Data_Ref','Binding_Name']
    def match(string):
        return BinaryOpBase.match(\
            Data_Ref, pattern.percent_op.named(),  Binding_Name, string)
    match = staticmethod(match)

class Actual_Arg_Spec(KeywordValueBase): # R1220
    """
    <actual-arg-spec> = [ <keyword> = ] <actual-arg>
    """
    subclass_names = ['Actual_Arg']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Keyword, Actual_Arg, string)
    match = staticmethod(match)

class Actual_Arg(Base): # R1221
    """
    <actual-arg> = <expr>
                 | <variable>
                 | <procedure-name>
                 | <proc-component-ref>
                 | <alt-return-spec>
    """
    subclass_names = ['Procedure_Name','Proc_Component_Ref','Alt_Return_Spec', 'Variable', 'Expr']

class Alt_Return_Spec(Base): # R1222
    """
    <alt-return-spec> = * <label>
    """
    subclass_names = []
    def match(string):
        if not string.startswith('*'): return
        line = string[1:].lstrip()
        if pattern.abs_label.match(line):
            return line,
        return
    match = staticmethod(match)
    def init(self, label):
        self.label = label
        return
    def tostr(self): return '*%s' % (self.label)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.label)

class Function_Subprogram(BlockBase): # R1223
    """
    <function-subprogram> = <function-stmt>
                               [ <specification-part> ]
                               [ <execution-part> ]
                               [ <internal-subprogram-part> ]
                            <end-function-stmt>
    """
    subclass_names = []
    use_names = ['Function_Stmt', 'Specification_Part', 'Execution_Part',
                 'Internal_Subprogram_Part', 'End_Function_Stmt']
    def match(reader):
        return BlockBase.match(Function_Stmt, [Specification_Part, Execution_Part, Internal_Subprogram_Part], End_Function_Stmt, reader)
    match = staticmethod(match)

class Function_Stmt(StmtBase): # R1224
    """
    <function-stmt> = [ <prefix> ] FUNCTION <function-name> ( [ <dummy-arg-name-list> ] ) [ <suffix> ]
    """
    subclass_names = []
    use_names = ['Prefix','Function_Name','Dummy_Arg_Name_List', 'Suffix']

class Proc_Language_Binding_Spec(Base): #1225
    """
    <proc-language-binding-spec> = <language-binding-spec>
    """
    subclass_names = ['Language_Binding_Spec']

class Dummy_Arg_Name(Base): # R1226
    """
    <dummy-arg-name> = <name>
    """
    subclass_names = ['Name']

class Prefix(SequenceBase): # R1227
    """
    <prefix> = <prefix-spec> [ <prefix-spec> ]..
    """
    subclass_names = ['Prefix_Spec']
    _separator = (' ',re.compile(r'\s+(?=[a-z_])',re.I))
    def match(string): return SequenceBase.match(Prefix._separator, Prefix_Spec, string)
    match = staticmethod(match)

class Prefix_Spec(STRINGBase): # R1228
    """
    <prefix-spec> = <declaration-type-spec>
                    | RECURSIVE
                    | PURE
                    | ELEMENTAL
    """
    subclass_names = ['Declaration_Type_Spec']
    def match(string):
        return STRINGBase.match(['RECURSIVE', 'PURE', 'ELEMENTAL'], string)
    match = staticmethod(match)

class Suffix(Base): # R1229
    """
    <suffix> = <proc-language-binding-spec> [ RESULT ( <result-name> ) ]
               | RESULT ( <result-name> ) [ <proc-language-binding-spec> ]
    """
    subclass_names = ['Proc_Language_Binding_Spec']
    use_names = ['Result_Name']

class End_Function_Stmt(EndStmtBase): # R1230
    """
    <end-function-stmt> = END [ FUNCTION [ <function-name> ] ]
    """
    subclass_names = []
    use_names = ['Function_Name']
    def match(string): return EndStmtBase.match('FUNCTION',Function_Name, string)
    match = staticmethod(match)

class Subroutine_Subprogram(BlockBase): # R1231
    """
    <subroutine-subprogram> = <subroutine-stmt>
                                 [ <specification-part> ]
                                 [ <execution-part> ]
                                 [ <internal-subprogram-part> ]
                              <end-subroutine-stmt>
    """
    subclass_names = []
    use_names = ['Subroutine_Stmt', 'Specification_Part', 'Execution_Part',
                 'Internal_Subprogram_Part', 'End_Subroutine_Stmt']
    def match(reader):
        return BlockBase.match(Subroutine_Stmt, [Specification_Part, Execution_Part, Internal_Subprogram_Part], End_Subroutine_Stmt, reader)
    match = staticmethod(match)

class Subroutine_Stmt(StmtBase): # R1232
    """
    <subroutine-stmt> = [ <prefix> ] SUBROUTINE <subroutine-name> [ ( [ <dummy-arg-list> ] ) [ <proc-language-binding-spec> ] ]
    """
    subclass_names = []
    use_names = ['Prefix', 'Subroutine_Name', 'Dummy_Arg_List', 'Proc_Language_Binding_Spec']
    def match(string):
        line, repmap = string_replace_map(string)
        m = pattern.subroutine.search(line)
        if m is None: return
        prefix = line[:m.start()].rstrip() or None
        if prefix is not None:
            prefix = Prefix(repmap(prefix))
        line = line[m.end():].lstrip()
        m = pattern.name.match(line)
        if m is None: return
        name = Subroutine_Name(m.group())
        line = line[m.end():].lstrip()
        dummy_args = None
        if line.startswith('('):
            i = line.find(')')
            if i==-1: return
            dummy_args = line[1:i].strip() or None
            if dummy_args is not None:
                dummy_args = Dummy_Arg_List(repmap(dummy_args))
            line = line[i+1:].lstrip()
        binding_spec = None
        if line:
            binding_spec = Proc_Language_Binding_Spec(repmap(line))
        return prefix, name, dummy_args, binding_spec
    match = staticmethod(match)
    def init(self, *args):
        self.prefix, self.name, self.dummy_args, self.binding_spec = args
        return
    def tostr(self):
        if self.prefix is not None:
            s = '%s SUBROUTINE %s' % (self.prefix, self.name)
        else:
            s = 'SUBROUTINE %s' % (self.name)
        if self.dummy_args is not None:
            s += '(%s)' % (self.dummy_args)
        if self.binding_spec is not None:
            s += ' %s' % (self.binding_spec)
        return s
    def torepr(self):
        return '%s(%r, %r, %r, %r)'\
               % (self.__class__.__name__, self.prefix, self.name, self.dummy_args, self.binding_spec)

class Dummy_Arg(StringBase): # R1233
    """
    <dummy-arg> = <dummy-arg-name>
                  | *
    """
    subclass_names = ['Dummy_Arg_Name']
    def match(string): return StringBase.match('*', string)
    match = staticmethod(match)

class End_Subroutine_Stmt(EndStmtBase): # R1234
    """
    <end-subroutine-stmt> = END [ SUBROUTINE [ <subroutine-name> ] ]
    """
    subclass_names = []
    use_names = ['Subroutine_Name']
    def match(string): return EndStmtBase.match('SUBROUTINE', Subroutine_Name, string)
    match = staticmethod(match)

class Entry_Stmt(StmtBase): # R1235
    """
    <entry-stmt> = ENTRY <entry-name> [ ( [ <dummy-arg-list> ] ) [ <suffix> ] ]
    """
    subclass_names = []
    use_names = ['Entry_Name', 'Dummy_Arg_List', 'Suffix']

class Return_Stmt(StmtBase): # R1236
    """
    <return-stmt> = RETURN [ <scalar-int-expr> ]
    """
    subclass_names = []
    use_names = ['Scalar_Int_Expr']
    def match(string):
        start = string[:6].upper()
        if start!='RETURN': return
        if len(string)==6: return None,
        return Scalar_Int_Expr(string[6:].lstrip()),
    match = staticmethod(match)
    def init(self, value):
        self.value = value
        return
    def tostr(self): 
        if self.value is None: return 'RETURN'
        return 'RETURN %s' % (self.value)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.value)

class Contains_Stmt(StmtBase, STRINGBase): # R1237
    """
    <contains-stmt> = CONTAINS
    """
    subclass_names = []
    def match(string): return STRINGBase.match('CONTAINS',string)
    match = staticmethod(match)

class Stmt_Function_Stmt(StmtBase): # R1238
    """
    <stmt-function-stmt> = <function-name> ( [ <dummy-arg-name-list> ] ) = Scalar_Expr
    """
    subclass_names = []
    use_names = ['Function_Name', 'Dummy_Arg_Name_List', 'Scalar_Expr']

###############################################################################
################ GENERATE Scalar_, _List, _Name CLASSES #######################
###############################################################################

ClassType = type(Base)
_names = dir()
for clsname in _names:
    cls = eval(clsname)
    if not (isinstance(cls, ClassType) and issubclass(cls, Base) and not cls.__name__.endswith('Base')): continue
    names = getattr(cls, 'subclass_names', []) + getattr(cls, 'use_names', [])
    for n in names:
        if n in _names: continue
        if n.endswith('_List'):
            _names.append(n)
            n = n[:-5]
            #print 'Generating %s_List' % (n)
            exec '''\
class %s_List(SequenceBase):
    subclass_names = [\'%s\']
    use_names = []
    def match(string): return SequenceBase.match(r\',\', %s, string)
    match = staticmethod(match)
''' % (n, n, n)
        elif n.endswith('_Name'):
            _names.append(n)
            n = n[:-5]
            #print 'Generating %s_Name' % (n)
            exec '''\
class %s_Name(Base):
    subclass_names = [\'Name\']
''' % (n)
        elif n.startswith('Scalar_'):
            _names.append(n)
            n = n[7:]
            #print 'Generating Scalar_%s' % (n)
            exec '''\
class Scalar_%s(Base):
    subclass_names = [\'%s\']
''' % (n,n)


Base_classes = {}
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base) and not cls.__name__.endswith('Base'):
        Base_classes[cls.__name__] = cls


###############################################################################
##################### OPTIMIZE subclass_names tree ############################
###############################################################################

if 1: # Optimize subclass tree:

    def _rpl_list(clsname):
        if not Base_classes.has_key(clsname):
            print 'Not implemented:',clsname
            return [] # remove this code when all classes are implemented
        cls = Base_classes[clsname]
        if cls.__dict__.has_key('match'): return [clsname]
        l = []
        for n in getattr(cls,'subclass_names',[]):
            l1 = _rpl_list(n)
            for n1 in l1: 
                if n1 not in l:
                    l.append(n1)
        return l
    
    for cls in Base_classes.values():
        if not hasattr(cls, 'subclass_names'): continue
        opt_subclass_names = []
        for n in cls.subclass_names:
            for n1 in _rpl_list(n):
                if n1 not in opt_subclass_names:  opt_subclass_names.append(n1)
        if not opt_subclass_names==cls.subclass_names:
            #print cls.__name__,':',', '.join(cls.subclass_names),'->',', '.join(opt_subclass_names)
            cls.subclass_names[:] = opt_subclass_names
        #else:
        #    print cls.__name__,':',opt_subclass_names


# Initialize Base.subclasses dictionary:
for clsname, cls in Base_classes.items():
    subclass_names = getattr(cls, 'subclass_names', None)
    if subclass_names is None:
        print '%s class is missing subclass_names list' % (clsname)
        continue
    try:
        l = Base.subclasses[clsname]
    except KeyError:
        Base.subclasses[clsname] = l = []
    for n in subclass_names:
        if Base_classes.has_key(n):
            l.append(Base_classes[n])
        else:
            print '%s not implemented needed by %s' % (n,clsname)

if 1:
    for cls in Base_classes.values():
        subclasses = Base.subclasses.get(cls.__name__,[])
        subclasses_names = [c.__name__ for c in subclasses]
        subclass_names = getattr(cls,'subclass_names', [])
        use_names = getattr(cls,'use_names',[])
        for n in subclasses_names:
            break
            if n not in subclass_names:
                print '%s needs to be added to %s subclasses_name list' % (n,cls.__name__)
        for n in subclass_names:
            break
            if n not in subclasses_names:
                print '%s needs to be added to %s subclass_name list' % (n,cls.__name__)
        for n in use_names + subclass_names:            
            if not Base_classes.has_key(n):
                print '%s not defined used by %s' % (n, cls.__name__)


#EOF
