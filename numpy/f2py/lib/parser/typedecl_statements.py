"""
Fortran type declaration statements.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['Integer', 'Real', 'DoublePrecision', 'Complex', 'DoubleComplex',
           'Character', 'Logical', 'Byte', 'TypeStmt','Class',
           'intrinsic_type_spec', 'declaration_type_spec',
           'Implicit']

import re
import string
from base_classes import Statement, BeginStatement, EndStatement,\
     AttributeHolder, Variable
from utils import split_comma, AnalyzeError, name_re, is_entity_decl, is_name, CHAR_BIT, parse_array_spec

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
    EXTENSION:
      <kind-selector> = ( [ KIND = ] <scalar-int-initialization-expr> )
                        | * <length>

    <char-selector> = <length-selector>
                      | ( LEN = <type-param-value>, KIND = <scalar-int-initialization-expr> )
                      | ( <type-param-value>, [ KIND = ] <scalar-int-initialization-expr> )
                      | ( KIND = <scalar-int-initialization-expr> [, LEN = <type-param-value>] )
    <length-selector> = ( [ LEN = ] <type-param-value> )
                        | * <char-length> [ , ]
    <char-length> = ( <type-param-value> ) | <scalar-int-literal-constant>

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

    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    <digit-string> = <digit> [ <digit> ]..
    <kind-param> = <digit-string> | <scalar-int-constant-name>
    """
    _repr_attr_names = ['selector','attrspec','entity_decls'] + Statement._repr_attr_names

    def process_item(self):
        item = self.item
        apply_map = item.apply_map
        clsname = self.__class__.__name__.lower()
        line = item.get_line()
        from block_statements import Function

        if not line.lower().startswith(clsname):
            i = 0
            j = 0
            for c in line:
                i += 1
                if c==' ': continue
                j += 1
                if j==len(clsname):
                    break
            line = line[:i].replace(' ','') + line[i:]

        assert line.lower().startswith(clsname),`line,clsname`
        line = line[len(clsname):].lstrip()

        if line.startswith('('):
            i = line.find(')')
            selector = apply_map(line[:i+1].strip())
            line = line[i+1:].lstrip()
        elif line.startswith('*'):
            selector = '*'
            line = line[1:].lstrip()
            if line.startswith('('):
                i = line.find(')')
                selector += apply_map(line[:i+1].rstrip())
                line = line[i+1:].lstrip()
            else:
                m = re.match(r'\d+(_\w+|)|[*]',line)
                if not m:
                    self.isvalid = False
                    return
                i = m.end()
                selector += line[:i].rstrip()
                line = line[i:].lstrip()
        else:
            selector = ''

        fm = Function.match(line)
        if fm:
            l2 = line[:fm.end()]
            m2 = re.match(r'.*?\b(?P<name>\w+)\Z',l2)
            if not m2:
                self.isvalid = False
                return
            fname = m2.group('name')
            fitem = item.copy(clsname+selector+' :: '+fname,
                              apply_map=True)
            self.parent.put_item(fitem)
            item.clone(line)
            self.isvalid = False
            return

        if line.startswith(','):
            line = line[1:].lstrip()

        self.raw_selector = selector
        if isinstance(self, Character):
            self.selector = self._parse_char_selector(selector)
        else:
            self.selector = self._parse_kind_selector(selector)

        i = line.find('::')
        if i==-1:
            self.attrspec = []
            self.entity_decls = split_comma(line, self.item)
        else:
            self.attrspec = split_comma(line[:i].rstrip(), self.item)
            self.entity_decls = split_comma(line[i+2:].lstrip(), self.item)
        for entity in self.entity_decls:
            if not is_entity_decl(entity):
                self.isvalid = False
                return

        if isinstance(self.parent, Function) \
               and self.parent.name in self.entity_decls:
            assert self.parent.typedecl is None,`self.parent.typedecl`
            self.parent.typedecl = self
            self.ignore = True
        if isinstance(self, Type):
            self.name = self.selector[1].lower()
            assert is_name(self.name),`self.name`
        else:
            self.name = clsname
        return

    def _parse_kind_selector(self, selector):
        if not selector:
            return '',''
        length,kind = '',''
        if selector.startswith('*'):
            length = selector[1:].lstrip()
        else:
            assert selector[0]+selector[-1]=='()',`selector`
            l = selector[1:-1].strip()
            if l.lower().startswith('kind'):
                l = l[4:].lstrip()
                assert l.startswith('='),`l`
                kind = l[1:].lstrip()
            else:
                kind = l
        return length,kind

    def _parse_char_selector(self, selector):
        if not selector:
            return '',''
        if selector.startswith('*'):
            l = selector[1:].lstrip()
            if l.startswith('('):
                if l.endswith(','): l = l[:-1].rstrip()
                assert l.endswith(')'),`l`
                l = l[1:-1].strip()
                if l.lower().startswith('len'):
                    l = l[3:].lstrip()[1:].lstrip()
            kind=''
        else:
            assert selector[0]+selector[-1]=='()',`selector`
            l = split_comma(selector[1:-1].strip(), self.item)
            if len(l)==1:
                l = l[0]
                if l.lower().startswith('len'):
                    l=l[3:].lstrip()
                    assert l.startswith('='),`l`
                    l=l[1:].lstrip()
                    kind = ''
                elif l.lower().startswith('kind'):
                    kind = l[4:].lstrip()[1:].lstrip()
                    l = ''
                else:
                    kind = ''
            else:
                assert len(l)==2
                if l[0].lower().startswith('len'):
                    assert l[1].lower().startswith('kind'),`l`
                    kind = l[1][4:].lstrip()[1:].lstrip()
                    l = l[0][3:].lstrip()[1:].lstrip()
                elif l[0].lower().startswith('kind'):
                    assert l[1].lower().startswith('len'),`l`
                    kind = l[0][4:].lstrip()[1:].lstrip()
                    l = l[1][3:].lstrip()[1:].lstrip()
                else:
                    if l[1].lower().startswith('kind'):
                        kind = l[1][4:].lstrip()[1:].lstrip()
                        l = l[0]
                    else:
                        kind = l[1]
                        l = l[0]
        return l,kind

    def tostr(self):
        clsname = self.__class__.__name__.upper()
        s = ''
        length, kind = self.selector
        if isinstance(self, Character):
            if length and kind:
                s += '(LEN=%s, KIND=%s)' % (length,kind)
            elif length:
                s += '(LEN=%s)' % (length)
            elif kind:
                s += '(KIND=%s)' % (kind)
        else:
            if isinstance(self, Type):
                s += '(%s)' % (kind)
            else:
                if length:
                    s += '*%s' % (length)
                if kind:
                    s += '(KIND=%s)' % (kind)

        return clsname + s

    def tofortran(self,isfix=None):
        tab = self.get_indent_tab(isfix=isfix)
        s = self.tostr()
        if self.attrspec:
            s += ', ' + ', '.join(self.attrspec)
            if self.entity_decls:
                s += ' ::'
        if self.entity_decls:
            s += ' ' + ', '.join(self.entity_decls)
        return tab + s

    def __str__(self):
        return self.tofortran()

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self.selector==other.selector

    def astypedecl(self):
        if self.entity_decls or self.attrspec:
            return self.__class__(self.parent, self.item.copy(self.tostr()))
        return self

    def analyze(self):
        if not self.entity_decls:
            return
        variables = self.parent.a.variables
        typedecl = self.astypedecl()
        attrspec = self.attrspec[:]
        try:
            access_spec = [a for a in attrspec if a.lower() in ['private','public']][0]
            attrspec.remove(access_spec)
        except IndexError:
            access_spec = None
        for item in self.entity_decls:
            name, array_spec, char_length, value = self._parse_entity(item)
            var = self.parent.get_variable(name)
            var.add_parent(self)
            if char_length:
                var.set_length(char_length)
            else:
                var.set_type(typedecl)
            var.update(self.attrspec)
            if array_spec:
                var.set_bounds(array_spec)
            if value:
                var.set_init(value)
            if access_spec is not None:
                l = getattr(self.parent.a,access_spec.lower() + '_id_list')
                l.append(name)
            var.analyze()
        return

    def _parse_entity(self, line):
        m = name_re(line)
        assert m,`line,self.item,self.__class__.__name__`
        name = line[:m.end()]
        line = line[m.end():].lstrip()
        array_spec = None
        item = self.item.copy(line)
        line = item.get_line()
        if line.startswith('('):
            i = line.find(')')
            assert i!=-1,`line`
            array_spec = parse_array_spec(line[1:i].strip(), item)
            line = line[i+1:].lstrip()
        char_length = None
        if line.startswith('*'):
            i = line.find('=')
            if i==-1:
                char_length = item.apply_map(line[1:].lstrip())
                line = ''
            else:
                char_length = item.apply_map(line[1:i].strip())
                line = line[i:]
        value = None
        if line.startswith('='):
            value = item.apply_map(line[1:].lstrip())
        return name, array_spec, char_length, value

    def get_zero_value(self):
        raise NotImplementedError,`self.__class__.__name__`

    def assign_expression(self, name, value):
        return '%s = %s' % (name, value)

    def get_kind(self):
        return self.selector[1] or self.default_kind

    def get_length(self):
        return self.selector[0] or 1

    def get_byte_size(self):
        length, kind = self.selector
        if length: return int(length)
        if kind: return int(kind)
        return self.default_kind

    def get_bit_size(self):
        return CHAR_BIT * int(self.get_byte_size())

    def is_intrinsic(self): return not isinstance(self,(Type,Class))
    def is_derived(self): return isinstance(self,Type)

    def is_numeric(self): return isinstance(self,(Integer,Real, DoublePrecision,Complex,DoubleComplex,Byte))
    def is_nonnumeric(self): return isinstance(self,(Character,Logical))


class Integer(TypeDeclarationStatement):
    match = re.compile(r'integer\b',re.I).match
    default_kind = 4

    def get_zero_value(self):
        kind = self.get_kind()
        if kind==self.default_kind: return '0'
        return '0_%s' % (kind)

class Real(TypeDeclarationStatement):
    match = re.compile(r'real\b',re.I).match
    default_kind = 4

    def get_zero_value(self):
        kind = self.get_kind()
        if kind==self.default_kind: return '0.0'
        return '0_%s' % (kind)

class DoublePrecision(TypeDeclarationStatement):
    match = re.compile(r'double\s*precision\b',re.I).match
    default_kind = 8

    def get_byte_size(self):
        return self.default_kind

    def get_zero_value(self):
        return '0.0D0'

class Complex(TypeDeclarationStatement):
    match = re.compile(r'complex\b',re.I).match
    default_kind = 4

    def get_byte_size(self):
        length, kind = self.selector
        if length: return int(length)
        if kind: return 2*int(kind)
        return 2*self.default_kind

    def get_zero_value(self):
        kind = self.get_kind()
        if kind==self.default_kind: return '(0.0, 0.0)'
        return '(0.0_%s, 0.0_%s)' % (kind, kind)

    def get_part_typedecl(self):
        bz = self.get_byte_size()/2
        return Real(self.parent, self.item.copy('REAL*%s' % (bz)))

class DoubleComplex(TypeDeclarationStatement):
    # not in standard
    match = re.compile(r'double\s*complex\b',re.I).match
    default_kind = 8

    def get_byte_size(self):
        return 2*self.default_kind

    def get_zero_value(self):
        return '(0.0D0,0.0D0)'

class Logical(TypeDeclarationStatement):
    match = re.compile(r'logical\b',re.I).match
    default_kind = 4

    def get_zero_value(self):
        return ".FALSE."

class Character(TypeDeclarationStatement):
    match = re.compile(r'character\b',re.I).match
    default_kind = 1

    def get_bit_size(self):
        length = self.get_length()
        if length=='*':
            return 0  # model for character*(*)
        return CHAR_BIT * int(length) * int(self.get_kind())

    def get_zero_value(self):
        return "''"

class Byte(TypeDeclarationStatement):
    # not in standard
    match = re.compile(r'byte\b',re.I).match
    default_kind = 1

    def get_zero_value(self):
        return '0'

class Type(TypeDeclarationStatement):
    match = re.compile(r'type\s*\(', re.I).match

    def get_zero_value(self):
        type_decl = self.get_type_decl(self.name)
        component_names = type_decl.a.component_names
        components = type_decl.a.components
        l = []
        for name in component_names:
            var = components[name]
            l.append(var.typedecl.get_zero_value())
        return '%s(%s)' % (type_decl.name, ', '.join(l))

    def get_kind(self):
        # See 4.5.2, page 48
        raise NotImplementedError,`self.__class__.__name__`

    def get_bit_size(self):
        return self.get_type_decl(self.name).get_bit_size()

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
    match = re.compile(r'implicit\b',re.I).match

    letters = string.lowercase

    def process_item(self):
        line = self.item.get_line()[8:].lstrip()
        if line.lower()=='none':
            self.items = []
            return
        items = []
        for item in split_comma(line, self.item):
            i = item.find('(')
            assert i!=-1 and item.endswith(')'),`item`
            specs = []
            for spec in split_comma(item[i+1:-1].strip(), self.item):
                if '-' in spec:
                    s,e = spec.lower().split('-')
                    s = s.strip()
                    e = e.strip()
                    assert s in self.letters and e in self.letters,`s,e`
                else:
                    e = s = spec.lower().strip()
                    assert s in self.letters,`s,e`
                specs.append((s,e))
            tspec = item[:i].rstrip()
            stmt = None
            for cls in declaration_type_spec:
                if cls.match(tspec):
                    stmt = cls(self, self.item.copy(tspec))
                    if stmt.isvalid:
                        break
            assert stmt is not None,`item,line`
            items.append((stmt,specs))
        self.items = items
        return

    def tofortran(self, isfix=None):
        tab = self.get_indent_tab(isfix=isfix)
        if not self.items:
            return tab + 'IMPLICIT NONE'
        l = []
        for stmt,specs in self.items:
            l1 = []
            for s,e in specs:
                if s==e:
                    l1.append(s)
                else:
                    l1.append(s + '-' + e)
            l.append('%s ( %s )' % (stmt.tostr(), ', '.join(l1)))
        return tab + 'IMPLICIT ' + ', '.join(l)

    def analyze(self):
        implicit_rules = self.parent.a.implicit_rules
        if not self.items:
            if implicit_rules:
                self.warning('overriding previously set implicit rule mapping'\
                      ' %r.' % (implicit_rules))
            self.parent.a.implicit_rules = None
            return
        if implicit_rules is None:
            self.warning('overriding previously set IMPLICIT NONE')
            self.parent.a.implicit_rules = implicit_rules = {}
        for stmt,specs in self.items:
            for s,e in specs:
                for l in string.lowercase[string.lowercase.index(s.lower()):\
                                          string.lowercase.index(e.lower())+1]:
                    implicit_rules[l] = stmt
        return

intrinsic_type_spec = [ \
    Integer , Real,
    DoublePrecision, Complex, DoubleComplex, Character, Logical, Byte
    ]
declaration_type_spec = intrinsic_type_spec + [ TypeStmt, Class ]
