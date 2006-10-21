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
from splitline import string_replace_map
import pattern_tools as pattern

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


## class Designator(Primary):
##     """
##     <designator> = <object-name>
##                    | <array-element>
##                    | <array-section>
##                    | <structure-component>
##                    | <substring>
##     <array-element> = <data-ref>
##     <array-section> = <data-ref> [ ( <substring-range> ) ]
##     <data-ref> = <part-ref> [ % <part-ref> ]...
##     <part-ref> = <part-name> [ ( <section-subscript-list> ) ]
##     <substring> = <parent-string> ( <substring-range> )
##     <parent-string> = <scalar-variable-name>
##                       | <array-element>
##                       | <scalar-structure-component>
##                       | <scalar-constant>
##     <substring-range> = [ <scalar-int-expr> ] : [ <scalar-int-expr> ]
##     <structure-component> = <data-ref>
##     """



## class LiteralConstant(Constant):
##     """
##     <constant> = <int-literal-constant>
##                  | <real-literal-constant>
##                  | <complex-literal-constant>
##                  | <logical-literal-constant>
##                  | <char-literal-constant>
##                  | <boz-literal-constant>
##     """

## class SignedIntLiteralConstant(LiteralConstant):
##     """
##     <signed-int-literal-constant> = [ <sign> ] <int-literal-constant>
##     <sign> = + | -
##     """
##     match = re.compile(r'\A[+-]\s*\d+\Z').match

##     def init(self, string):
##         Base.init(self, string)
##         self.content = [string[0], IntLiteralConstant(string[1:].lstrip())]
##         return
##     def tostr(self):
##         return '%s%s' % tuple(self.content)

## class NamedConstant(Constant):
##     """
##     <named-constant> = <name>
##     """

## class Name(Designator, NamedConstant, NoChildAllowed):
##     """
##     <name> = <letter> [ <alpha-numeric-character> ]...
##     """
##     match = re.compile(r'\A'+name_pat+r'\Z',re.I).match

## class IntLiteralConstant(SignedIntLiteralConstant, NoChildAllowed):
##     """
##     <int-literal-constant> = <digit-string> [ _ <kind-param> ]
##     <kind-param> = <digit-string>
##                  | <scalar-int-constant-name>
##     <digit-string> = <digit> [ <digit> ]...
##     """
##     match = compose_pattern([digit_string_pat, '_', kind_param_pat],r'\s*')

##     compose_pattern('int-literal-constant','digit-string','_','kind-param')

## class DigitString(IntLiteralConstant, NoChildAllowed):
##     """
##     <digit-string> = <digit> [ <digit> ]...
##     """
##     match = re.compile(r'\A\d+\Z').match


class Base(object):

    subclasses = {}

    def __new__(cls, string):
        if hasattr(cls,'match'):
            match = cls.match
            result = match(string)
        else:
            result = None
        if isinstance(result, tuple):
            obj = object.__new__(cls)
            obj.string = string
            obj.init(*result)
            return obj
        elif isinstance(result, Base):
            return result
        elif result is None:
            for subcls in Base.subclasses.get(cls.__name__,[]):
                try:
                    return subcls(string)
                except NoMatchError:
                    pass
        else:
            raise AssertionError,`result`
        raise NoMatchError,'%s: %r' % (cls.__name__, string)

    findall = staticmethod(re.compile(r'(_F2PY_STRING_CONSTANT_\d+_|F2PY_EXPR_TUPLE_\d+)').findall)
    
    def match_binary_operand_right(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.rsplit(line)
        if t is None: return
        lhs, op, rhs = t
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repman[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repman[k])
        lhs_obj = lhs_cls(lhs)
        rhs_obj = rhs_cls(rhs)
        return lhs_obj, t[1], rhs_obj
    match_binary_operand_right = staticmethod(match_binary_operand_right)

    def match_binary_unary_operand_right(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.rsplit(line)
        if t is None: return
        lhs, op, rhs = t
        if lhs: 
            for k in Base.findall(lhs):
                lhs = lhs.replace(k, repman[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repman[k])
        rhs_obj = rhs_cls(rhs)
        if lhs:
            lhs_obj = lhs_cls(lhs)
            return lhs_obj, t[1], rhs_obj
        else:
            return None, t[1], rhs_obj
    match_binary_unary_operand_right = staticmethod(match_binary_unary_operand_right)

    def match_binary_operand_left(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.lsplit(line)
        if t is None: return
        lhs, op, rhs = t
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repman[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repman[k])
        lhs_obj = lhs_cls(lhs)
        rhs_obj = rhs_cls(rhs)
        return lhs_obj, t[1], rhs_obj
    match_binary_operand_left = staticmethod(match_binary_operand_left)

    def match_unary_operand(op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.lsplit(line)
        if t is None: return
        lhs, op, rhs = t
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repman[k])
        assert not lhs,`lhs`
        rhs_obj = rhs_cls(rhs)
        return t[1], rhs_obj
    match_unary_operand = staticmethod(match_unary_operand)

    def init_binary_operand(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        return

    def init_unary_operand(self, op, rhs):
        self.op = op
        self.rhs = rhs
        return

    def init_primary(self, primary):
        self.primary = primary
        return

    def tostr_binary_operand(self):
        return '%s %s %s' % (self.lhs, self.op, self.rhs)

    def tostr_binary_unary_operand(self):
        if self.lhs is None:
            return '%s %s' % (self.op, self.rhs)
        return '%s %s %s' % (self.lhs, self.op, self.rhs)

    def tostr_unary_operand(self):
        return '%s %s' % (self.op, self.rhs)

    def tostr_primary(self):
        return str(self.primary)

    def torepr_binary_operand(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,self.lhs, self.op, self.rhs)

    def torepr_binary_unary_operand(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,self.lhs, self.op, self.rhs)

    def torepr_unary_operand(self):
        return '%s(%r, %r)' % (self.__class__.__name__,self.op, self.rhs)

    def torepr_primary(self):
        return '%s(%r)' % (self.__class__.__name__,self.primary)

    def __str__(self):
        if self.__class__.__dict__.has_key('tostr'):
            return self.tostr()
        return repr(self)

    def __repr__(self):
        if self.__class__.__dict__.has_key('torepr'):
            return self.torepr()
        return '%s(%r)' % (self.__class__.__name__, self.string)

class Expr(Base):
    """
    <expr> = [ <expr> <defined-binary-op> ] <level-5-expr>
    <defined-binary-op> = . <letter> [ <letter> ]... .
    """
    def match(string):
        return Base.match_binary_operand_right(\
            Expr,pattern.defined_binary_op.named(),Level_5_Expr,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand

class Level_5_Expr(Expr):
    """
    <level-5-expr> = [ <level-5-expr> <equiv-op> ] <equiv-operand>
    <equiv-op> = .EQV.
               | .NEQV.
    """
    def match(string):
        return Base.match_binary_operand_right(\
            Level_5_Expr,pattern.equiv_op.named(),Equiv_Operand,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class Equiv_Operand(Level_5_Expr):
    """
    <equiv-operand> = [ <equiv-operand> <or-op> ] <or-operand>
    <or-op>  = .OR.
    """
    def match(string):
        return Base.match_binary_operand_right(\
            Equiv_Operand,pattern.or_op.named(),Or_Operand,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class Or_Operand(Equiv_Operand):
    """
    <or-operand> = [ <or-operand> <and-op> ] <and-operand>    
    <and-op> = .AND.

    """
    def match(string):
        return Base.match_binary_operand_right(\
            Or_Operand,pattern.and_op.named(),And_Operand,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class And_Operand(Or_Operand):
    """
    <and-operand> = [ <not-op> ] <level-4-expr>
    <not-op> = .NOT.
    """
    def match(string):
        return Base.match_unary_operand(\
            pattern.not_op.named(),Level_4_Expr,string)
    match = staticmethod(match)
    init = Base.init_unary_operand
    tostr = Base.tostr_unary_operand
    torepr = Base.torepr_unary_operand
    
class Level_4_Expr(And_Operand):
    """
    <level-4-expr> = [ <level-3-expr> <rel-op> ] <level-3-expr>
    <rel-op> = .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. | == | /= | < | <= | > | >=
    """
    def match(string):
        return Base.match_binary_operand_right(\
            Level_3_Expr,pattern.rel_op.named(),Level_3_Expr,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class Level_3_Expr(Level_4_Expr):
    """
    <level-3-expr> = [ <level-3-expr> <concat-op> ] <level-2-expr>
    <concat-op>    = //
    """
    def match(string):
        return Base.match_binary_operand_right(\
            Level_3_Expr,pattern.concat_op.named(),Level_2_Expr,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class Level_2_Expr(Level_3_Expr):
    """
    <level-2-expr> = [ [ <level-2-expr> ] <add-op> ] <add-operand>
    <add-op>   = +
                 | -
    """

    def match(string):
        return Base.match_binary_unary_operand_right(\
            Level_2_Expr,pattern.add_op.named(),Add_Operand,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_unary_operand
    torepr = Base.torepr_binary_operand
    
class Add_Operand(Level_2_Expr):
    """
    <add-operand> = [ <add-operand> <mult-op> ] <mult-operand>
    <mult-op>  = *
                 | /
    """

    def match(string):
        return Base.match_binary_operand_right(\
            Add_Operand,pattern.mult_op.named(),Mult_Operand,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class Mult_Operand(Add_Operand):
    """
    <mult-operand> = <level-1-expr> [ <power-op> <mult-operand> ]
    <power-op> = **
    """

    def match(string):
        return Base.match_binary_operand_left(\
            Level_1_Expr,pattern.power_op.named(),Mult_Operand,string)
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand
    
class Level_1_Expr(Mult_Operand):
    """
    <level-1-expr> = [ <defined-unary-op> ] <primary>
    <defined-unary-op> = . <letter> [ <letter> ]... .
    """
    def match(string):
        return Base.match_unary_operand(\
            pattern.defined_unary_op.named(),Primary,string)
    match = staticmethod(match)
    init = Base.init_unary_operand
    tostr = Base.tostr_unary_operand
    torepr = Base.torepr_unary_operand
    
class Primary(Level_1_Expr):
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
    def match(string):
        if pattern.abs_constant.match(string):
            return (string,)
        return
    match = staticmethod(match)
    init = Base.init_primary
    tostr = Base.tostr_primary
    torepr = Base.torepr_primary

ClassType = type(Base)
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base):
        for basecls in cls.__bases__:
            if issubclass(basecls, Base):
                try:
                    Base.subclasses[basecls.__name__].append(cls)
                except KeyError:
                    Base.subclasses[basecls.__name__] = [cls]


print Constant('a')
print `Constant('1')`
print `Base('+1')`
print `Base('c-1*a/b')`
