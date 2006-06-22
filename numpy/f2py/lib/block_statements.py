"""

"""

import re
import sys

from base_classes import BeginStatement, EndStatement, Statement
from readfortran import Line

# File block

class EndSource(EndStatement):
    """
    """
    match = staticmethod(lambda s: False)

class BeginSource(BeginStatement):
    """
    """
    match = staticmethod(lambda s: True)

    end_stmt_cls = EndSource

    def process_item(self):
        self.name = self.reader.name
        self.fill(end_flag = True)
        return

    def get_classes(self):
        return program_unit

    def process_subitem(self, item):
        # MAIN block does not define start/end line conditions,
        # so it should never end until all lines are read.
        # However, sometimes F77 programs lack the PROGRAM statement,
        # and here we fix that:
        if self.reader.isfix77:
            line = item.get_line()
            if line=='end':
                message = self.reader.format_message(\
                        'WARNING',
                        'assuming the end of undefined PROGRAM statement',
                        item.span[0],item.span[1])
                print >> sys.stderr, message
                p = Program(self)
                p.content.extend(self.content)
                p.content.append(EndProgram(p,item))
                self.content[:] = [p]
                return
        return BeginStatement.process_subitem(self, item)

# Module

class EndModule(EndStatement):
    match = re.compile(r'end(\s*module\s*\w*|)\Z', re.I).match

class Module(BeginStatement):
    """
    MODULE <name>
     ..
    END [MODULE [name]]
    """
    match = re.compile(r'module\s*\w+\Z', re.I).match
    end_stmt_cls = EndModule

    def get_classes(self):
        return access_spec + specification_part + module_subprogram_part

    def process_item(self):
        name = self.item.get_line().replace(' ','')[len(self.blocktype):].strip()
        self.name = name
        return BeginStatement.process_item(self)

    #def __str__(self):
    #    s = self.get_indent_tab(deindent=True)
    #    s += 'MODULE '+ self.name
    #    return s

# Python Module

class EndPythonModule(EndStatement):
    match = re.compile(r'end(\s*python\s*module\s*\w*|)\Z', re.I).match

class PythonModule(BeginStatement):
    """
    PYTHON MODULE <name>
     ..
    END [PYTHON MODULE [name]]
    """
    modes = ['pyf']
    match = re.compile(r'python\s*module\s*\w+\Z', re.I).match
    end_stmt_cls = EndPythonModule

    def get_classes(self):
        return [Interface, Function, Subroutine, Module]

    def process_item(self):
        name = self.item.get_line().replace(' ','')[len(self.blocktype):].strip()
        self.name = name
        return BeginStatement.process_item(self)

# Program

class EndProgram(EndStatement):
    """
    END [PROGRAM [name]]
    """
    match = re.compile(r'end(\s*program\s*\w*|)\Z', re.I).match

class Program(BeginStatement):
    """ PROGRAM [name]
    """
    match = re.compile(r'program\s*\w*\Z', re.I).match
    end_stmt_cls = EndProgram

    def get_classes(self):
        return specification_part + execution_part + internal_subprogram_part

    def process_item(self):
        name = self.item.get_line().replace(' ','')[len(self.blocktype):].strip()
        if name:
            self.name = name
        return BeginStatement.process_item(self)

# Interface

class EndInterface(EndStatement):
    match = re.compile(r'end\s*interface\s*\w*\Z', re.I).match
    blocktype = 'interface'

class Interface(BeginStatement):
    """
    INTERFACE [generic-spec] | ABSTRACT INTERFACE
    END INTERFACE [generic-spec]
    """
    modes = ['free90', 'fix90', 'pyf']
    match = re.compile(r'(interface\s*\w*|abstract\s*interface)\Z',re.I).match
    end_stmt_cls = EndInterface
    blocktype = 'interface'

    def get_classes(self):
        return interface_specification

    def process_item(self):
        line = self.item.get_line()
        self.isabstract = line.startswith('abstract')
        if self.isabstract:
            self.generic_spec = ''
        else:
            self.generic_spec = line[len(self.blocktype):].strip()
        self.name = self.generic_spec # XXX
        return BeginStatement.process_item(self)

    def tostr(self):
        if self.isabstract:
            return 'ABSTRACT INTERFACE'
        return 'INTERFACE '+ str(self.generic_spec)


# Subroutine

class EndSubroutine(EndStatement):
    """
    END [SUBROUTINE [name]]
    """
    match = re.compile(r'end(\s*subroutine\s*\w*|)\Z', re.I).match

class Subroutine(BeginStatement):
    """
    [prefix] SUBROUTINE <name> [ ( [<dummy-arg-list>] ) [<proc-language-binding-spec>]]
    """
    end_stmt_cls = EndSubroutine
    match = re.compile(r'[\w\s]*subroutine\s*\w+', re.I).match
    
    item_re = re.compile(r'(?P<prefix>[\w\s]*)\s*subroutine\s*(?P<name>\w+)', re.I).match
    def process_item(self):
        line = self.item.get_line()
        m = self.item_re(line)
        self.name = m.group('name')
        line = line[m.end():].strip()
        args = []
        if line.startswith('('):
            assert line.endswith(')'),`line`
            for a in line.split(','):
                args.append(a.strip())
        self.args = args
        return BeginStatement.process_item(self)

    def get_classes(self):
        return specification_part + execution_part + internal_subprogram_part

# Function

class EndFunction(EndStatement):
    """
    END [FUNCTION [name]]
    """
    match = re.compile(r'end(\s*function\s*\w*|)\Z', re.I).match

class Function(BeginStatement):
    """
    [prefix] SUBROUTINE <name> [ ( [<dummy-arg-list>] ) [suffix]
    """
    end_stmt_cls = EndFunction
    match = re.compile(r'([\w\s]+(\(\s*\w+\s*\)|)|)\s*function\s*\w+', re.I).match
    item_re = re.compile(r'(?P<prefix>([\w\s](\(\s*\w+\s*\)|))*)\s*function\s*(?P<name>\w+)\s*\((?P<args>.*)\)\s*(?P<suffix>.*)\Z', re.I).match    

    def process_item(self):
        line = self.item.get_line()
        m = self.item_re(line)
        if m is None:
            self.isvalid = False
            return
        self.name = m.group('name')
        self.prefix = m.group('prefix').strip()
        self.suffix = m.group('suffix').strip()
        args = []
        for a in m.group('args').split(','):
            args.append(a.strip())
        self.args = args
        return BeginStatement.process_item(self)

    def tostr(self):
        return '%s FUNCTION %s(%s) %s' % (self.prefix, self.name,
                                          ', '.join(self.args), self.suffix)

    def get_classes(self):
        return specification_part + execution_part + internal_subprogram_part

# SelectCase

class EndSelect(EndStatement):
    match = re.compile(r'end\s*select\s*\w*\Z', re.I).match
    blocktype = 'select'

class Select(BeginStatement):
    """
    [ <case-construct-name> : ] SELECT CASE ( <case-expr> )
    
    """
    match = re.compile(r'select\s*case\s*\(.*\)\Z',re.I).match
    end_stmt_cls = EndSelect
    name = ''
    def tostr(self):
        return 'SELECT CASE ( %s )' % (self.expr)
    def process_item(self):
        self.expr = self.item.get_line()[6:].lstrip()[4:].lstrip()[1:-1].strip()
        self.name = self.item.label                
        return BeginStatement.process_item(self)

    def get_classes(self):
        return [Case] + execution_part_construct

# IfThen

class EndIfThen(EndStatement):
    match = re.compile(r'end\s*if\s*\w*\Z', re.I).match
    blocktype = 'if'

class IfThen(BeginStatement):
    """
    [<if-construct-name> :] IF ( <scalar-logical-expr> ) THEN

    IfThen instance has the following attributes:
      expr
    """

    match = re.compile(r'if\s*\(.*\)\s*then\Z',re.I).match
    end_stmt_cls = EndIfThen
    name = ''

    def tostr(self):
        return 'IF (%s) THEN' % (self.expr)

    def process_item(self):
        item = self.item
        line = item.get_line()[2:-4].strip()
        assert line[0]=='(' and line[-1]==')',`line`
        self.expr = line[1:-1].strip()
        self.name = item.label
        return BeginStatement.process_item(self)
        
    def get_classes(self):
        return [Else, ElseIf] + execution_part_construct

class If(BeginStatement):
    """
    IF ( <scalar-logical-expr> ) action-stmt
    """

    match = re.compile(r'if\s*\(').match

    def process_item(self):
        item = self.item
        mode = self.reader.mode
        classes = self.get_classes()
        classes = [cls for cls in classes if mode in cls.modes]

        line = item.get_line()[2:]
        i = line.find(')')
        expr = line[1:i].strip()
        line = line[i+1:].strip()
        if line=='then':
            self.isvalid = False
            return
        self.expr = expr[1:-1]

        newitem = item.copy(line)
        for cls in classes:
            if cls.match(line):
                stmt = cls(self, newitem)
                if stmt.isvalid:
                    self.content.append(stmt)
                    return
        self.handle_unknown_item(newitem)
        return
        
    def tostr(self):
        assert len(self.content)==1,`self.content`
        return 'IF (%s) %s' % (self.expr, str(self.content[0]).lstrip())

    def __str__(self):
        return self.get_indent_tab(colon=':') + self.tostr()

    def get_classes(self):
        return action_stmt

# Do

class EndDo(EndStatement):
    """
    """
    match = re.compile(r'end\s*do\s*\w*\Z', re.I).match
    blocktype = 'do'

class Do(BeginStatement):
    """
    [ <do-construct-name> : ] DO label [loopcontrol]
    [ <do-construct-name> : ] DO [loopcontrol]

    """

    match = re.compile(r'do\b\s*\d*',re.I).match
    item_re = re.compile(r'do\b\s*(?P<label>\d*)\s*,?\s*(?P<loopcontrol>.*)\Z',re.I).match
    end_stmt_cls = EndDo
    name = ''

    def tostr(self):
        return 'DO %s %s' % (self.endlabel, self.loopcontrol)

    def process_item(self):
        item = self.item
        line = item.get_line()
        m = self.item_re(line)
        self.endlabel = m.group('label').strip()
        self.name = item.label
        self.loopcontrol = m.group('loopcontrol').strip()
        return BeginStatement.process_item(self)

    def process_subitem(self, item):
        r = False
        if self.endlabel:
            label = item.label
            if label == self.endlabel:
                r = True
                if isinstance(self.parent, Do) and label==self.parent.endlabel:
                    # the same item label may be used for different block ends
                    self.put_item(item)
        return BeginStatement.process_subitem(self, item) or r

    def get_classes(self):
        return execution_part_construct

class EndType(EndStatement):
    """
    END TYPE [<type-name>]
    """
    match = re.compile(r'end\s*type\s*\w*\Z', re.I).match
    blocktype = 'type'

class Type(BeginStatement):
    """
    TYPE [ [, <type-attr-spec-list>] ::] <type-name> [ ( <type-param-name-list> ) ]
    <type-attr-spec> = <access-spec> | EXTENDS ( <parent-type-name> )
                       | ABSTRACT | BIND(C)
    """
    match = re.compile(r'type\b\s*').match
    end_stmt_cls = EndType

    def process_item(self):
        line = self.item.get_line()[4:].lstrip()
        if line.startswith('('):
            self.isvalid = False
            return
        attr_specs = []
        i = line.find('::')
        if i!=-1:
            for s in line[:i].split(','):
                s = s.strip()
                if s: attr_specs.append(s)
            line = line[i+2:].lstrip()
        self.attr_specs = attr_specs
        i = line.find('(')
        if i!=-1:
            self.name = line[:i].rstrip()
            assert line[-1]==')',`line`
            self.params = line[i+1:-1].lstrip()
        else:
            self.name = line
            self.params = ''
        return BeginStatement.process_item(self)

    def tostr(self):
        s = 'TYPE'
        if self.attr_specs:
            s += ', '.join(['']+self.attr_specs) + ' ::'
        s += ' ' + self.name
        if self.params:
            s += ' ('+self.params+')'
        return s

    def get_classes(self):
        return [Integer] + private_or_sequence + component_part + type_bound_procedure_part

TypeDecl = Type

###################################################

from statements import *
from typedecl_statements import *

access_spec = [Public, Private]

interface_specification = [Function, Subroutine,
                           ModuleProcedure
                           ]

module_subprogram_part = [
    Contains,
    Function,
    Subroutine
    ]

specification_stmt = [
    # Access, Allocatable, Asynchronous, Bind, Common,
    Data, Dimension,
    Equivalence, #External, Intent
    # Intrinsic, Namelist, Optional, Pointer, Protected,
    Save, #Target, Volatile, Value
    ]
intrinsic_type_spec = [
    Integer , Real, DoublePrecision, Complex, Character, Logical
    ]
declaration_type_spec = intrinsic_type_spec + [
    TypeStmt,
    Class
    ]
type_declaration_stmt = declaration_type_spec

private_or_sequence = [
    Private, #Sequence
    ]

component_part = declaration_type_spec + [
    #Procedure
    ]

type_bound_procedure_part = [
    Contains, Private, #Procedure, Generic, Final
    ]

#R214
action_stmt = [
    Allocate,
    Assignment, #PointerAssignment,
    Backspace,
    Call,
    Close,
    Continue,
    Cycle,
    Deallocate,
    Endfile, #EndFunction, EndProgram, EndSubroutine,
    Exit,
    # Flush, Forall,
    Goto, If, #Inquire,
    Nullify,
    Open, 
    Print, Read,
    Return,
    Rewind,
    Stop, #Wait, Where,
    Write,
    # arithmetic-if-stmt, computed-goto-stmt
    ]

executable_construct = action_stmt + [
    # Associate, Case,
    Do,
    # Forall,
    IfThen,
    Select, #Where
    ]
execution_part_construct = executable_construct + [
    Format, #Entry, Data
    ]
execution_part = execution_part_construct[:]

#C201, R208
for cls in [EndFunction, EndProgram, EndSubroutine]:
    try: execution_part.remove(cls)
    except ValueError: pass

internal_subprogram = [Function, Subroutine]
internal_subprogram_part = [
    Contains,
    ] + internal_subprogram

declaration_construct = [
    TypeDecl, #Entry, Enum,
    Format,
    Interface,
    Parameter, #Procedure,
    ] + specification_stmt + type_declaration_stmt # stmt-function-stmt
implicit_part = [
    Implicit, Parameter, Format, #Entry
    ]
specification_part = [
    Use, #Import 
    ] + implicit_part + declaration_construct
external_subprogram = [Function, Subroutine]
main_program = [Program] + specification_part + execution_part + internal_subprogram_part
program_unit = main_program + external_subprogram + [Module,
                                                     #BlockData
                                                     ]
