
import re
from base_classes import Statement, BeginStatement, EndStatement

# Intrinsic type specification statements

class TypeDeclarationStatement(Statement):
    """
    <declaration-type-spec> [ [, <attr-spec>] :: ] <entity-decl-list>
    <declaration-type-spec> = <intrinsic-type-spec>
                              | TYPE ( <derived-type-spec> )
                              | CLASS ( <derived-type-spec> )
                              | CLASS ( * )

    <derived-type-spec> = <type-name> [ ( <type-param-spec-list> ) ]
    <type-param-spec> = [ <keyword> = ] <type-param-value>
    <type-param-value> = <scalar-int-expr> | * | :

    <intrinsic-type-spec> = INTEGER [<kind-selector>]
                            | REAL [<kind-selector>]
                            | DOUBLE PRECISION
                            | COMPLEX [<kind-selector>]
                            | CHARACTER [<char-selector>]
                            | LOGICAL [<kind-selector>]

    <kind-selector> = ( [ KIND = ] <scalar-int-initialization-expr> )
    <char-selector> = <length-selector>
                      | ( LEN = <type-param-value>, KIND = <scalar-int-initialization-expr> )
                      | ( <type-param-value>, [ KIND = ] <scalar-int-initialization-expr> )
                      | ( KIND = <scalar-int-initialization-expr> [, LEN = <type-param-value>] )
    <length-selector> = ( [ LEN = ] <type-param-value> )
                        | * <char-length> [ , ]
    <char-length> = ( <type-param-value> ) | <scalar-int-literal-expr>

    <attr-spec> = <access-spec> | ALLOCATABLE | ASYNCHRONOUS
                  | DIMENSION ( <array-spec> ) | EXTERNAL
                  | INTENT ( <intent-spec> ) | INTRINSIC
                  | <language-binding-spec> | OPTIONAL
                  | PARAMETER | POINTER | PROTECTED | SAVE
                  | TARGET | VALUE | VOLATILE
    <entity-decl> = <object-name> [ ( <array-spec> ) ] [ * <char-length> ] [ <initialization> ]
                  | <function-name> [ * <char-length> ]
    <initialization> =  = <initialization-expr>
                        | => NULL
    <access-spec> = PUBLIC | PRIVATE
    <language-binding-spec> = BIND ( C [ , NAME = <scalar-char-initialization-expr>] )
    <array-spec> =   <explicit-shape-spec-list>
                   | <assumed-shape-spec-list>
                   | <deferred-shape-spec-list>
                   | <assumed-size-spec>
    <explicit-shape-spec> = [ <lower-bound> : ] <upper-bound>
    <assumed-shape-spec> = [ <lower-bound> ] :
    <deferred-shape-spec> = :
    <assumed-size-spec> = [ <explicit-shape-spec-list> , ] [ <lower-bound> : ] *
    <bound> = <specification-expr>

    """

    def process_item(self):
        item = self.item
        clsname = self.__class__.__name__.lower()
        line = item.get_line()
        from block_statements import Function
        if Function.match(line):
            self.isvalid = False
            return
        if not line.startswith(clsname):
            line = line[:len(clsname)].replace(' ','') + line[len(clsname):]

        assert line.startswith(clsname),`line,clsname`
        line = line[len(clsname):].lstrip()

        if line.startswith('('):
            i = line.find(')')
            selector = line[:i+1].strip()
            line = line[i+1:].lstrip()
        elif line.startswith('*'):
            selector = '*'
            line = line[1:].lstrip()
            if line.startswith('('):
                i = line.find(')')
                selector += line[:i+1].rstrip()
                line = line[i+1:].lstrip()
            else:
                i = len(line)
                ci = ''
                for c in [',','::',' ']:
                    j = line.find(c)
                    if j!=-1 and j<i:
                        i = j
                        ci = c
                assert i!=len(line),`i,line`
                selector += line[:i].rstrip()
                line = line[i+len(ci):].lstrip()
        else:
            selector = ''
        if line.startswith(','):
            line = line[1:].lstrip()

        self.raw_selector = selector
        i = line.find('::')
        if i==-1:
            self.attrspec = ''
            self.entity_decls = line
        else:
            self.attrspec = line[:i].rstrip()
            self.entity_decls = line[i+2:].lstrip()
        return

    def __str__(self):
        clsname = self.__class__.__name__.upper()
        tab = self.get_indent_tab()
        return tab + clsname + '%s, %s :: %s' \
               % (self.raw_selector, self.attrspec, self.entity_decls)
        
class Integer(TypeDeclarationStatement):
    match = re.compile(r'integer\b',re.I).match

class Real(TypeDeclarationStatement):
    match = re.compile(r'real\b',re.I).match

class DoublePrecision(TypeDeclarationStatement):
    match = re.compile(r'double\s*precision\b',re.I).match

class Complex(TypeDeclarationStatement):
    match = re.compile(r'complex\b',re.I).match

class DoubleComplex(TypeDeclarationStatement):
    match = re.compile(r'double\s*complex\b',re.I).match
    modes = ['pyf','fix77']

class Logical(TypeDeclarationStatement):
    match = re.compile(r'logical\b',re.I).match

class Character(TypeDeclarationStatement):
    match = re.compile(r'character\b',re.I).match

class Type(TypeDeclarationStatement):
    match = re.compile(r'type\s*\(', re.I).match
TypeStmt = Type

class Class(TypeDeclarationStatement):
    match = re.compile(r'class\s*\(', re.I).match


class Implicit(Statement):
    """
    IMPLICIT <implicit-spec-list>
    IMPLICIT NONE
    <implicit-spec> = <declaration-type-spec> ( <letter-spec-list> )
    <letter-spec> = <letter> [ - <letter> ]
    """
    match = re.compile(r'implicit\b').match
    def process_item(self):
        line = self.item.get_line()[8:].lstrip()
        if line=='none':
            self.items = []
            return
        self.items = [s.strip() for s in line.split()]
        assert self.items

    def __str__(self):
        tab = self.get_indent_tab()
        if not self.items:
            return tab + 'IMPLICIT NONE'
        return tab + 'IMPLICIT ' + ', '.join(self.items)
