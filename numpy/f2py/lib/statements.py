
import re
import sys

from base_classes import Statement

# Execution statements

class Assignment(Statement):
    """
    <variable> = <expr>
    <pointer variable> => <expr>
    """

    match = re.compile(r'\w(\s*\(\s*[^)]*\)|[\w%]*)*\s*=\>?',re.I).match
    item_re = re.compile(r'(?P<variable>\w(\s*\(\s*[^)]*\)|[\w%]*)*)\s*(?P<sign>=\>?)\s*(?P<expr>.*)\Z',re.I).match

    def process_item(self):
        m = self.item_re(self.item.get_line())
        self.variable = m.group('variable').replace(' ','')
        self.sign = m.group('sign')
        self.expr = m.group('expr')
        return

    def __str__(self):
        return self.get_indent_tab() + '%s %s %s' \
               % (self.variable, self.sign, self.expr)
    
class Call(Statement):
    """Call statement class
    CALL <proc-designator> [([arg-spec-list])]

    Call instance has attributes:
      designator
      arg_list
    """
    match = re.compile(r'call\b', re.I).match

    def process_item(self):
        item = self.item
        line = item.get_line()[4:].strip()
        i = line.find('(')
        self.arg_list = []
        if i==-1:
            self.designator = line.strip()
        else:
            self.designator = line[:i].strip()
            for n in line[i+1:-1].split(','):
                n = n.strip()
                if not n: continue
                self.arg_list.append(n)
        return

    def __str__(self):
        s = self.get_indent_tab() + 'CALL '+str(self.designator)
        if self.arg_list:
            s += '('+', '.join(map(str,self.arg_list))+ ')'
        return s

class Goto(Statement):
    """
    GO TO <label>
    
    """
    match = re.compile(r'go\s*to\b\s*\w*\Z', re.I).match

    def process_item(self):
        self.gotolabel = self.item.get_line()[2:].strip()[2:].strip()
        return

    def __str__(self):
        return self.get_indent_tab() + 'GO TO %s' % (self.gotolabel)
        
class Continue(Statement):
    """
    CONTINUE
    """
    match = re.compile(r'continue\Z',re.I).match

    def process_item(self):
        self.label = self.item.label
        return

    def __str__(self):
        return self.get_indent_tab(deindent=True) + 'CONTINUE'


class Return(Statement):
    """
    RETURN [scalar-int-expr]
    """
    match = re.compile(r'return\b',re.I).match

    def process_item(self):
        line = self.item.get_line()[6:].lstrip()
        self.expr = line
        return

    def __str__(self):
        return self.get_indent_tab() + 'RETURN %s' % (self.expr)

class Stop(Statement):
    """
    STOP [stop-code]
    """
    match = re.compile(r'stop\b\s*\w*\Z',re.I).match

    def process_item(self):
        self.stopcode = self.item.get_line()[4:].lstrip()
        return

    def __str__(self):
        return self.get_indent_tab() + 'STOP %s' % (self.stopcode)

class Print(Statement):
    """
    PRINT <format> [, <output-item-list>]
    <format> == <default-char-expr> | <label> | *
    """
    match = re.compile(r'print\b\s*[\w*]', re.I).match

    def process_item(self):
        item = self.item
        line = item.get_line()[5:].lstrip()
        items = line.split(',')
        self.format = items[0].strip()
        self.items = [s.strip() for s in items[1:]]
        return

    def __str__(self):
        return self.get_indent_tab() + 'PRINT %s' % (', '.join([self.format]+self.items))

class Read(Statement):
    """
Read0:    READ ( io-control-spec-list ) [<input-item-list>]
    
Read1:    READ <format> [, <input-item-list>]
    <format> == <default-char-expr> | <label> | *
    """
    match = re.compile(r'read\b\s*[\w(*]', re.I).match

    def process_item(self):
        item = self.item
        line = item.get_line()[4:].lstrip()
        if line.startswith('('):
            self.__class__ = Read0
        else:
            self.__class__ = Read1
        self.process_item()

class Read0(Read):

    def process_item(self):
        item = self.item
        line = item.get_line()[4:].lstrip()
        i = line.find(')')
        self.io_control_specs = line[1:i].strip()
        self.items = [s.strip() for s in line[i+1:].split(',')]

    def __str__(self):
        return self.get_indent_tab() + 'READ (%s) %s' \
               % (self.io_control_specs, ', '.join(self.items))

class Read1(Read):

    def process_item(self):
        item = self.item
        line = item.get_line()[4:].lstrip()
        items = line.split(',')
        self.format = items[0].strip()
        self.items = [s.strip() for s in items[1:]]
        return

    def __str__(self):
        return self.get_indent_tab() + 'READ %s' % (', '.join([self.format]+self.items))

class Write(Statement):
    """
    WRITE ( io-control-spec-list ) [<output-item-list>]
    """
    match = re.compile(r'write\s*\(', re.I).match
    def process_item(self):
        item = self.item
        line = item.get_line()[5:].lstrip()
        i = line.find(')')
        self.io_control_specs = line[1:i].strip()
        self.items = [s.strip() for s in line[i+1:].split(',')]

    def __str__(self):
        return self.get_indent_tab() + 'WRITE (%s) %s' \
               % (self.io_control_specs, ', '.join(self.items))        

class Contains(Statement):
    """
    CONTAINS
    """
    match = re.compile(r'contains\Z',re.I).match
    def process_item(self):
        return
    def __str__(self): return self.get_indent_tab() + 'CONTAINS'

class Allocate(Statement):
    """
    ALLOCATE ( [ <type-spec> :: ] <allocation-list> [ , <alloc-opt-list> ] )
    """
    match = re.compile(r'allocate\s*\(.*\)\Z',re.I).match
    def process_item(self):
        self.items = self.item.get_line()[8:].lstrip()[1:-1].strip()
    def __str__(self): return self.get_indent_tab() \
        + 'ALLOCATE ( %s )' % (self.items)

class Deallocate(Statement):
    """
    DEALLOCATE ( <allocate-object-list> [ , <dealloc-opt-list> ] )
    """
    match = re.compile(r'deallocate\s*\(.*\)\Z',re.I).match
    def process_item(self):
        self.items = self.item.get_line()[10:].lstrip()[1:-1].strip()
    def __str__(self): return self.get_indent_tab() \
        + 'DEALLOCATE ( %s )' % (self.items)

class ModuleProcedure(Statement):
    """
    [ MODULE ] PROCEDURE <procedure-name-list>
    """
    match = re.compile(r'(module\s*|)procedure\b',re.I).match
    def process_item(self):
        line = self.item.get_line()
        m = self.match(line)
        self.names = [s.strip() for s in line[m.end():].split(',')]
    def __str__(self):
        tab = self.get_indent_tab()
        return tab + 'MODULE PROCEDURE %s' % (', '.join(self.names))

class Access(Statement):
    """
    <access-spec> [ [::] <access-id-list>]
    <access-spec> = PUBLIC | PRIVATE
    """
    match = re.compile(r'(public|private)\b',re.I).match
    def process_item(self):
        clsname = self.__class__.__name__.lower()
        line = self.item.get_line()
        if not line.startswith(clsname):
            self.isvalid = False
            return
        line = line[len(clsname):].lstrip()
        if line.startswith('::'):
            line = line[2:].lstrip()
        self.items = [s.strip() for s in line.split(',')]
    def __str__(self):
        clsname = self.__class__.__name__.upper()
        tab = self.get_indent_tab()
        if self.items:
            return tab + clsname + ' :: ' + ', '.join(self.items)
        return tab + clsname

class Public(Access): pass
class Private(Access): pass

class Close(Statement):
    """
    CLOSE ( <close-spec-list> )
    <close-spec> = [ UNIT = ] <file-unit-number>
                   | IOSTAT = <scalar-int-variable>
                   | IOMSG = <iomsg-variable>
                   | ERR = <label>
                   | STATUS = <scalar-default-char-expr>
    """
    match = re.compile(r'close\s*\(.*\)\Z',re.I).match
    def process_item(self):
        self.close_specs = self.item.get_line()[5:].lstrip()[1:-1].strip()
        return
    def __str__(self):
        tab = self.get_indent_tab()
        return tab + 'CLOSE (%s)' % (self.close_specs)

class Cycle(Statement):
    """
    CYCLE [ <do-construct-name> ]
    """
    match = re.compile(r'cycle\b\s*\w*\Z',re.I).match
    def process_item(self):
        self.name = self.item.get_line()[5:].lstrip()
        return
    def __str__(self):
        return self.get_indent_tab() + 'CYCLE ' + self.name

class FilePositioningStatement(Statement):
    """
    REWIND <file-unit-number>
    REWIND ( <position-spec-list> )
    <position-spec-list> = [ UNIT = ] <file-unit-number>
                           | IOMSG = <iomsg-variable>
                           | IOSTAT = <scalar-int-variable>
                           | ERR = <label>
    The same for BACKSPACE, ENDFILE.
    """
    match = re.compile(r'(rewind|backspace|endfile)\b',re.I).match

    def process_item(self):
        clsname = self.__class__.__name__.lower()
        line = self.item.get_line()
        if not line.startswith(clsname):
            self.isvalid = False
            return
        line = line[len(clsname):].lstrip()
        if line.startswith('('):
            assert line[-1]==')',`line`
            self.fileunit = None
            self.position_specs = line[1:-1].strip()
        else:
            self.fileunit = line
            self.position_specs = None
        return

    def __str__(self):
        clsname = self.__class__.__name__.upper()
        if self.fileunit is None:
            return self.get_indent_tab() + clsname + ' (%s)' % (self.position_specs)
        return self.get_indent_tab() + clsname + ' %s' % (self.fileunit)

class Backspace(FilePositioningStatement): pass

class Endfile(FilePositioningStatement): pass

class Rewind(FilePositioningStatement): pass

class Open(Statement):
    """
    OPEN ( <connect-spec-list> )
    <connect-spec> = [ UNIT = ] <file-unit-number>
                     | ACCESS = <scalar-default-char-expr>
                     | ..
    """
    match = re.compile(r'open\s*\(.*\)\Z',re.I).match
    def process_item(self):
        self.connect_specs = self.item.get_line()[4:].lstrip()[1:-1].strip()
        return
    def __str__(self):
        return self.get_indent_tab() + 'OPEN (%s)' % (self.connect_specs)

class Format(Statement):
    """
    FORMAT <format-specification>
    <format-specification> = ( [ <format-item-list> ] )
    
    """
    match = re.compile(r'format\s*\(.*\)\Z', re.I).match
    def process_item(self):
        item = self.item
        if not item.label:
            # R1001:
            message = self.reader.format_message(\
                        'WARNING',
                        'R1001: FORMAT statement must be labeled but got %r.' \
                        % (item.label),
                        item.span[0],item.span[1])
            print >> sys.stderr, message
        line = item.get_line()[6:].lstrip()
        assert line[0]+line[-1]=='()',`line`
        self.specs = line[1:-1].strip()
        return
    def __str__(self):
        return self.get_indent_tab() + 'FORMAT (%s)' % (self.specs)

class Save(Statement):
    """
    SAVE [ [ :: ] <saved-entity-list> ]
    <saved-entity> = <object-name>
                     | <proc-pointer-name>
                     | / <common-block-name> /
    <proc-pointer-name> = <name>
    <object-name> = <name>
    """
    match = re.compile(r'save\b',re.I).match
    is_name = re.compile(r'\w+\Z').match
    def process_item(self):
        line = self.item.get_line()[4:].lstrip()
        if line.startswith('::'):
            line = line[2:].lstrip()
        items = []
        for s in line.split(','):
            s = s.strip()
            if not s: continue
            if s.startswith('/'):
                assert s.endswith('/'),`s`
                n = s[1:-1].strip()
                assert self.is_name(n)
                items.append('/%s/' % ())
            elif self.is_name(s):
                items.append(s)
            else:
                self.isvalid = False
                return
        self.items = items
        return
    def __str__(self):
        tab = self.get_indent_tab()
        if not self.items:
            return tab + 'SAVE'
        return tab + 'SAVE :: %s' % (', '.join(self.items))

class Data(Statement):
    """
    DATA <data-stmt-set> [ [ , ] <data-stmt-set> ]...
    <data-stmt-set> = <data-stmt-object-list> / <data-stmt-value-list> /
    <data-stmt-object> = <variable> | <data-implied-do>
    <data-implied-do> = ( <data-i-do-object-list> , <data-i-do-variable> = <scalar-int-expr> , <scalar-int-expr> [ , <scalar-int-expr> ] )
    <data-i-do-object> = <array-element> | <scalar-structure-component> | <data-implied-do>
    <data-i-do-variable> = <scalar-int-variable>
    <variable> = <designator>
    <designator> = <object-name>
                   | <array-element>
                   | <array-section>
                   | <structure-component>
                   | <substring>
    <array-element> = <data-ref>
    <array-section> = <data-ref> [ ( <substring-range> ) ]
    
    """
    match = re.compile(r'data\b',re.I).match

    def process_item(self):
        line = self.item.get_line()[4:].lstrip()
        stmts = []
        while line:
            i = line.find('/')
            assert i!=-1,`line`
            j = line.find('/',i+1)
            assert j!=-1,`line`
            stmts.append((line[:i].rstrip(),line[i+1:j].strip()))
            line = line[j+1:].lstrip()
            if line.startswith(','):
                line = line[1:].lstrip()
        self.stmts = stmts
        return

    def __str__(self):
        tab = self.get_indent_tab()
        l = []
        for o,v in self.stmts:
            l.append('%s / %s /' %(o,v))
        return tab + 'DATA ' + ' '.join(l)

class Nullify(Statement):
    """
    NULLIFY ( <pointer-object-list> )
    <pointer-object> = <variable-name>
    """
    match = re.compile(r'nullify\s*\(.*\)\Z',re.I).match
    def process_item(self):
        self.item_list = self.item.get_line()[7:].lstrip()[1:-1].strip()
        return
    def __str__(self):
        return self.get_indent_tab() + 'NULLIFY (%s)' % (self.item_list)

class Use(Statement):
    """
    USE [ [ , <module-nature> ] :: ] <module-name> [ , <rename-list> ]
    USE [ [ , <module-nature> ] :: ] <module-name> , ONLY : [ <only-list> ]
    <module-nature> = INTRINSIC | NON_INTRINSIC
    <rename> = <local-name> => <use-name>
               | OPERATOR ( <local-defined-operator> ) => OPERATOR ( <use-defined-operator> )
    <only> = <generic-spec> | <only-use-name> | <rename>
    <only-use-name> = <use-name>
    """
    match = re.compile(r'use\b',re.I).match
    def process_item(self):
        line = self.item.get_line()[3:].lstrip()
        nature = ''
        if line.startswith(','):
            i = line.find('::')
            nature = line[1:i].strip()
            line = line[i+2:].lstrip()
        if line.startswith('::'):
            line = line[2:].lstrip()
        self.nature = nature
        i = line.find(',')
        self.isonly = False
        if i==-1:
            self.module = line
            self.items = []
        else:
            self.module = line[:i].rstrip()
            line = line[i+1:].lstrip()
            if line.startswith('only') and line[4:].lstrip().startswith(':'):
                self.isonly = True
                line = line[4:].lstrip()[1:].lstrip()
            self.items = [s.strip() for s in line.split(',')]

    def __str__(self):
        tab = self.get_indent_tab()
        s = 'USE'
        if self.nature:
            s += ' ' + self.nature + ' ::'
        s += ' ' + self.module
        if self.isonly:
            s += ' ONLY:'
        elif self.items:
            s += ','
        if self.items:
            s += ' ' + ', '.join(self.items)
        return s

class Exit(Statement):
    """
    EXIT [ <do-construct-name> ]
    """
    match = re.compile(r'exit\b\s*\w*\Z',re.I).match
    def process_item(self):
        self.exitname = self.item.get_line()[4:].lstrip()
        return
    def __str__(self):
        return self.get_indent_tab() + 'EXIT ' + self.exitname

class Parameter(Statement):
    """
    PARAMETER ( <named-constant-def-list> )
    <named-constant-def> = <named-constant> = <initialization-expr>
    """
    match = re.compile(r'parameter\s*\(.*\)\Z', re.I).match
    def process_item(self):
        self.params = self.item.get_line()[9:].lstrip()[1:-1].strip()
        return
    def __str__(self):
        return self.get_indent_tab() + 'PARAMETER (%s)' % (self.params)

class Equivalence(Statement):
    """
    EQUIVALENCE <equivalence-set-list>
    <equivalence-set> = ( <equivalence-object> , <equivalence-object-list> )
    <equivalence-object> = <variable-name> | <array-element> | <substring>
    """
    match = re.compile(r'equivalence\s*\(.*\)\Z', re.I).match
    def process_item(self):
        items = []
        for s in self.item.get_line()[12:].lstrip().split(','):
            s = s.strip()
            assert s[0]+s[-1]=='()',`s`
            items.append(s)
        self.items = items
    def __str__(self):
        return self.get_indent_tab() + 'EQUIVALENCE %s' % (', '.join(self.items))

class Dimension(Statement):
    """
    DIMENSION [ :: ] <array-name> ( <array-spec> ) [ , <array-name> ( <array-spec> ) ]...
    
    """
    match = re.compile(r'dimension\b').match
    def process_item(self):
        line = self.item.get_line()[9:].lstrip()
        if line.startswith('::'):
            line = line[2:].lstrip()
        self.items = [s.split() for s in line.split(',')]
        return
    def __str__(self):
        return self.get_indent_tab() + 'DIMENSION %s' % (', '.join(self.items))

# IF construct statements

class Else(Statement):
    """
    ELSE [<if-construct-name>]
    """
    match = re.compile(r'else\s*\w*\Z',re.I).match

    def process_item(self):
        item = self.item
        self.name = item.get_line()[4:].strip()
        if self.name and not self.name==self.parent.name:
            message = self.reader.format_message(\
                        'WARNING',
                        'expected if-construct-name %r but got %r, skipping.'\
                        % (self.parent.name, self.name),
                        item.span[0],item.span[1])
            print >> sys.stderr, message
            self.isvalid = False        
        return

    def __str__(self):
        return self.get_indent_tab(deindent=True) + 'ELSE ' + self.name

class ElseIf(Statement):
    """
    ELSE IF ( <scalar-logical-expr> ) THEN [<if-construct-name>]
    """
    match = re.compile(r'else\s*if\s*\(.*\)\s*then\s*\w*\Z',re.I).match

    def process_item(self):
        item = self.item
        line = item.get_line()[4:].lstrip()[2:].lstrip()
        i = line.find(')')
        assert line[0]=='('
        self.expr = line[1:i]
        self.name = line[i+1:].lstrip()[4:].strip()
        if self.name and not self.name==self.parent.name:
            message = self.reader.format_message(\
                        'WARNING',
                        'expected if-construct-name %r but got %r, skipping.'\
                        % (self.parent.name, self.name),
                        item.span[0],item.span[1])
            print >> sys.stderr, message
            self.isvalid = False        
        return
        
    def __str__(self):
        return self.get_indent_tab(deindent=True) + 'ELSE IF (%s) THEN %s' \
               % (self.expr, self.name)

# SelectCase construct statements

class Case(Statement):
    """
    CASE <case-selector> [ <case-constract-name> ]
    <case-selector> = ( <case-value-range-list> ) | DEFAULT
    <case-value-range> = <case-value>
                         | <case-value> :
                         | : <case-value>
                         | <case-value> : <case-value>
    <case-value> = <scalar-(int|char|logical)-initialization-expr>
    """
    match = re.compile(r'case\b\s*(\(.*\)|DEFAULT)\s*\w*\Z',re.I).match
    def process_item(self):
        assert self.parent.__class__.__name__=='Select',`self.parent.__class__`
        line = self.item.get_line()[4:].lstrip()
        if line.startswith('('):
            i = line.find(')')
            self.ranges = line[1:i].strip()
            line = line[i+1:].lstrip()
        else:
            assert line.startswith('default'),`line`
            self.ranges = ''
            line = line[7:].lstrip()
        self.name = line
        if self.name and not self.name==self.parent.name:
            message = self.reader.format_message(\
                        'WARNING',
                        'expected case-construct-name %r but got %r, skipping.'\
                        % (self.parent.name, self.name),
                        self.item.span[0],self.item.span[1])
            print >> sys.stderr, message
            self.isvalid = False        
        return
