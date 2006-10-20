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

def set_subclasses(cls):
    for basecls in cls.__bases__:
        if issubclass(basecls, Base):
            try:
                subclasses = basecls.__dict__['_subclasses']
            except KeyError:
                subclasses = basecls._subclasses = []
            subclasses.append(cls)
    return

is_name = re.compile(r'\A[a-z]\w*\Z',re.I).match

class Expression:
    def __new__(cls, string):
        if is_name(string):
            obj = object.__new__(Name)
            obj._init(string)
            return obj

class NoMatch(Exception):
    pass

class Base(object):
    def __new__(cls, string):
        match = getattr(cls,'match',None)
        if match is not None:
            if match(string):
                obj = object.__new__(cls)
                obj._init(string)
                return obj
        else:
            assert cls._subclasses,`cls`
            for c in cls._subclasses:
                try:
                    return c(string)
                except NoMatch, msg:
                    pass
        raise NoMatch,'%s: %r' % (cls.__name__, string)
    def _init(self, string):
        self.string = string
        return
    def __str__(self): return self.string
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

class Name(Designator):
    """
    <name> = <letter> [ <alpha-numeric-character> ]...
    """
    match = is_name

class LiteralConstant(Constant):
    """
    <constant> = <int-literal-constant>
                 | <real-literal-constant>
                 | <complex-literal-constant>
                 | <logical-literal-constant>
                 | <char-literal-constant>
                 | <boz-literal-constant>
    """

class IntLiteralConstant(LiteralConstant):
    """
    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    <kind-param> = <digit-string>
                 | <scalar-int-constant-name>
    <digit-string> = <digit> [ <digit> ]...
    """
    match = re.compile(r'\A\d+\Z').match

class NamedConstant(Constant, Name):
    """
    <named-constant> = <name>
    """

ClassType = type(Base)
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base):
        set_subclasses(cls)

class Level1Expression(Primary):
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
