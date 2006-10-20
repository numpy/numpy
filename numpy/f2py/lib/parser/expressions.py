#!/usr/bin/env python
"""

Copyright 2006 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

$Date: $
Pearu Peterson
"""

import re

class DefinedOp:
    def __new__(cls, letters):
        if not letters: return None
        obj = object.__new__(cls)
        obj._init(letters)
        return obj

    def _init(self, letters):
        self.letters = letters.upper()
        return

    def __str__(self): return '.%s.' % (self.letters)
    def __repr__(self): return '%s(%r)' % (self.__class__.__name__, self.letters)

class NoChildAllowed:
    pass
class NoChildAllowedError(Exception):
    pass
class NoMatchError(Exception):
    pass

is_name = re.compile(r'\A[a-z]\w*\Z',re.I).match

class Base(object):

    subclasses = {}

    def __new__(cls, string):
        match = getattr(cls,'match',None)
        if match is not None:
            if match(string):
                obj = object.__new__(cls)
                init = cls.__dict__.get('init', Base.init)
                init(obj, string)
                return obj
        for c in Base.subclasses.get(cls.__name__,[]):
            try:
                return c(string)
            except NoMatchError:
                pass
        raise NoMatchError,'%s: %r' % (cls.__name__, string)

    def init(self, string):
        self.string = string
        return
    
    def __str__(self):
        str_func = self.__class__.__dict__.get('tostr', None)
        if str_func is not None:
            return str_func(self)
        return self.string
    def __repr__(self): return '%s(%r)' % (self.__class__.__name__, self.string)



class Primary(Base):
    """
    <primary> = <constant>
                | <designator>
                | <array-constructor>
                | <structure-constructor>
                | <function-reference>
                | <type-param-inquiry>
                | <type-param-name>
                | ( <expr> )
    <type-param-inquiry> = <designator> % <type-param-name>
    """

class Constant(Primary):
    """
    <constant> = <literal-constant>
                 | <named-constant>
    """

class Designator(Primary):
    """
    <designator> = <object-name>
                   | <array-element>
                   | <array-section>
                   | <structure-component>
                   | <substring>
    <array-element> = <data-ref>
    <array-section> = <data-ref> [ ( <substring-range> ) ]
    <data-ref> = <part-ref> [ % <part-ref> ]...
    <part-ref> = <part-name> [ ( <section-subscript-list> ) ]
    <substring> = <parent-string> ( <substring-range> )
    <parent-string> = <scalar-variable-name>
                      | <array-element>
                      | <scalar-structure-component>
                      | <scalar-constant>
    <substring-range> = [ <scalar-int-expr> ] : [ <scalar-int-expr> ]
    <structure-component> = <data-ref>
    """



class LiteralConstant(Constant):
    """
    <constant> = <int-literal-constant>
                 | <real-literal-constant>
                 | <complex-literal-constant>
                 | <logical-literal-constant>
                 | <char-literal-constant>
                 | <boz-literal-constant>
    """

class SignedIntLiteralConstant(LiteralConstant):
    """
    <signed-int-literal-constant> = [ <sign> ] <int-literal-constant>
    <sign> = + | -
    """
    match = re.compile(r'\A[+-]\s*\d+\Z').match

    def init(self, string):
        Base.init(self, string)
        self.content = [string[0], IntLiteralConstant(string[1:].lstrip())]
        return
    def tostr(self):
        return '%s%s' % tuple(self.content)

class NamedConstant(Constant):
    """
    <named-constant> = <name>
    """

def compose_patterns(pattern_list, names join=''):
    return join.join(pattern_list)

def add_pattern(pattern_name, *pat_list):
    p = ''
    for pat in pat_list:
        if isinstance(pat, PatternOptional):
            p += '(%s|)' % (add_pattern(None, pat.args))
        elif isinstance(pat, PatternOr):
            p += '(%s)' % ('|'.join([add_pattern(None, p1) for p1 in par.args]))
        else:
            subpat = pattern_map.get(pat,None)
            if subpat is None:
                p += pat
            else:
                p += '(?P<%s>%s)' % (pat, subpat)
    if pattern_map is not None:
        pattern_map[pattern_name] = p
    return p



class PatternBase:
    def __init__(self,*args):
        self.args = args
        return

class PatternOptional(PatternBase):
    pass
class PatternOr(PatternBase):
    pass
class PatternJoin(PatternBase):
    join = ''

pattern_map = {
    'name': r'[a-zA-Z]\w+'
    'digit-string': r'\d+'
    }
add_pattern('kind-param',
            PatternOr('digit-string','name'))
add_pattern('int-literal-constant',
            'digit-string',PatternOptional('_','kind-param'))

name_pat = r'[a-z]\w*'
digit_pat = r'\d'
digit_string_pat = r'\d+'
kind_param_pat = '(%s|%s)' % (digit_string_pat, name_pat)

class Name(Designator, NamedConstant, NoChildAllowed):
    """
    <name> = <letter> [ <alpha-numeric-character> ]...
    """
    match = re.compile(r'\A'+name_pat+r'\Z',re.I).match

class IntLiteralConstant(SignedIntLiteralConstant, NoChildAllowed):
    """
    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    <kind-param> = <digit-string>
                 | <scalar-int-constant-name>
    <digit-string> = <digit> [ <digit> ]...
    """
    match = compose_pattern([digit_string_pat, '_', kind_param_pat],r'\s*')

    compose_pattern('int-literal-constant','digit-string','_','kind-param')

class DigitString(IntLiteralConstant, NoChildAllowed):
    """
    <digit-string> = <digit> [ <digit> ]...
    """
    match = re.compile(r'\A\d+\Z').match

################# Setting up Base.subclasses #####################

def set_subclasses(cls):
    """
    Append cls to cls base classes attribute lists `_subclasses`
    so that all classes derived from Base know their subclasses
    one level down.
    """
    for basecls in cls.__bases__:
        if issubclass(basecls, Base):
            if issubclass(basecls, NoChildAllowed):
                raise NoChildAllowedError,'%s while adding %s' % (basecls.__name__,cls.__name__)
            try:
                Base.subclasses[basecls.__name__].append(cls)
            except KeyError:
                Base.subclasses[basecls.__name__] = [cls]
    return
ClassType = type(Base)
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base):
        set_subclasses(cls)

####################################################################

class Level1Expression:#(Primary):
    """
    <level-1-expr> = [ <defined-unary-op> ] <primary>
    <defined-unary-op> = . <letter> [ <letter> ]... .
    """
    def __new__(cls, primary, defined_unary_op = None):
        obj = object.__new__(cls)
        
        return obj

class Level2Expression:
    """
    <level-2-expr> = [ [ <level-2-expr> ] <add-op> ] <add-operand>
    <add-operand> = [ <add-operand> <mult-op> ] <mult-operand>
    <mult-operand> = <level-1-expr> [ <power-op> <mult-operand> ]
    <power-op> = **
    <mult-op>  = *
                 | /
    <add-op>   = +
                 | -
    """

class Level3Expression:
    """
    <level-3-expr> = [ <level-3-expr> <concat-op> ] <level-2-expr>
    <concat-op>    = //
    """

class Level4Expression:
    """
    <level-4-expr> = [ <level-3-expr> <rel-op> ] <level-3-expr>
    <rel-op> = .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. | == | /= | < | <= | > | >=
    """

class Level5Expression:
    """
    <level-5-expr> = [ <level-5-expr> <equiv-op> ] <equiv-operand>
    <equiv-operand> = [ <equiv-operand> <or-op> ] <or-operand>
    <or-operand> = [ <or-operand> <and-op> ] <and-operand>
    <and-operand> = [ <not-op> ] <level-4-expr>
    <not-op> = .NOT.
    <and-op> = .AND.
    <or-op>  = .OR.
    <equiv-op> = .EQV.
               | .NEQV.
    """

class Expression:
    """
    <expr> = [ <expr> <defined-binary-op> ] <level-5-expr>
    <defined-binary-op> = . <letter> [ <letter> ]... .
    """

from splitline import string_replace_map

def parse_expr(line, lower=False):
    newline, repmap = string_replace_map(line, lower=lower)
    if repmap:
        raise NotImplementedError,`newline,repmap`
