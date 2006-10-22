#!/usr/bin/env python
"""
Fortran expressions.

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

class NoMatchError(Exception):
    pass

class NoSubClasses:
    pass

class Base(object):

    subclasses = {}

    def __new__(cls, string):
        match = cls.__dict__.get('match', None)
        if match is not None:
            result = cls.match(string)
        else:
            result = None
        if isinstance(result, tuple):
            obj = object.__new__(cls)
            obj.string = string
            if cls.__dict__.has_key('init'):
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

    default_match = staticmethod(lambda string: None)

    findall = staticmethod(re.compile(r'(_F2PY_STRING_CONSTANT_\d+_|F2PY_EXPR_TUPLE_\d+)').findall)
    
    def match_binary_operand_right(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.rsplit(line)
        if t is None: return
        lhs, op, rhs = t
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
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
                lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
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
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
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
            rhs = rhs.replace(k, repmap[k])
        assert not lhs,`lhs`
        rhs_obj = rhs_cls(rhs)
        return t[1], rhs_obj
    match_unary_operand = staticmethod(match_unary_operand)

    def match_list_of(subcls, string):
        line, repmap = string_replace_map(string)
        lst = []
        for p in line.split(','):
            p = p.strip()
            for k in Base.findall(p):
                p = p.replace(k,repmap[k])
            lst.append(subcls(p))
        return tuple(lst)
    match_list_of = staticmethod(match_list_of)

    def init_list(self, *items):
        self.items = items
        return

    def tostr_list(self):
        return ', '.join(map(str,self.items))

    def torepr_list(self):
        return '%s(%s)' % (self.__class__.__name__,', '.join(map(repr,self.items)))

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

    def tostr_string(self):
        return str(self.string)

    def torepr_binary_operand(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,self.lhs, self.op, self.rhs)

    def torepr_binary_unary_operand(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,self.lhs, self.op, self.rhs)

    def torepr_unary_operand(self):
        return '%s(%r, %r)' % (self.__class__.__name__,self.op, self.rhs)

    def torepr_primary(self):
        return '%s(%r)' % (self.__class__.__name__,self.primary)

    def torepr_string(self):
        return '%s(%r)' % (self.__class__.__name__,self.string)

    def tostr_number(self):
        if self.items[1] is None: return str(self.items[0])
        return '%s_%s' % (self.items[0],self.items[1])
    def torepr_number(self):
        return '%s(%r,%r)' % (self.__class__.__name__, self.items[0],self.items[1])

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



class Array_Constructor(Primary):
    """
    <array-constructor> = (/ <ac-spec> /)
                          | <left-square-bracket> <ac-spec> <right-square-bracket>

    """
    def match(string):
        if string[:2]+string[-2:]=='(//)':
            return '(/',Ac_Spec(string[2:-2].strip()),'/)'
        if string[:1]+string[-1:]=='[]':
            return '[',Ac_Spec(string[1:-1].strip()),']'
        return
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return ''.join(map(str,self.items))
    def torepr(self): return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,self.items)))

class Ac_Spec(Base):
    """
    <ac-spec> = <type-spec> ::
                | [ <type-spec> :: ] <ac-value-list>
    """
    def match(string):
        if string.endswith('::'):
            return Type_Spec(string[:-2].rstrip()),None
        line, repmap = string_replace_map(string)
        i = line.find('::')
        if i==-1:
            return None, Ac_Value_List(string)
        ts = line[:i].rstrip()
        line = line[i+2:].lstrip()
        for k in Base.findall(ts):
            ts = ts.replace(k,repmap[k])
        for k in Base.findall(line):
            line = line.replace(k,repmap[k])
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

class Ac_Value_List(Base):
    """
    <ac-value-list> = <ac-value> [ , <ac-value> ]...
    """
    def match(string):
        return Base.match_list_of(Ac_Value, string)
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_list
    torepr = Base.torepr_list

class Ac_Value(Base):
    """
    <ac-value> = <expr>
                 | <ac-implied-do>
    """
    extra_subclasses = [Expr]

class Ac_Implied_Do(Ac_Value):
    """
    <ac-implied-do> = ( <ac-value-list> , <ac-implied-do-control> )
    """
    def match(string):
        if string[0]+string[-1] != '()': return
        line, repmap = string_replace_map(string[1:-1].strip())
        i = line.rfind('=')
        if i==-1: return
        j = line[:i].rfind(',')
        assert j!=-1
        s1 = line[:j].rstrip()
        s2 = line[j+1:].lstrip()
        for k in Base.findall(s1):
            s1 = s1.replace(k,repmap[k])
        for k in Base.findall(s2):
            s2 = s2.replace(k,repmap[k])
        return Ac_Value_List(s1),Ac_Implied_Do_Control(s2)
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return '(%s, %s)' % tuple(self.items)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Ac_Implied_Do_Control(Base):
    """
    <ac-implied-do-control> = <ac-do-variable> = <scalar-int-expr> , <scalar-int-expr> [ , <scalar-int-expr> ]    
    """
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

class Scalar_Int_Expr(Base):
    """
    <scalar-int-expr> = <expr>
    """
    extra_subclasses = [Expr]

class Ac_Do_Variable(Base):
    """
    <ac-do-variable> = <scalar-int-variable>
    <ac-do-variable> shall be a named variable    
    """

class Type_Spec(Base):
    """
    <type-spec> = <intrinsic-type-spec>
                  | <derived-type-spec>
    """

class Intrinsic_Type_Spec(Type_Spec):
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
    def match(string):
        if string[:7].upper()=='INTEGER':
            t = string[:7].upper()
            line = string[7:].lstrip()
            if line: return t,Kind_Selector(line)
            return t,None
        elif string[:4].upper()=='REAL':
            t = string[:4].upper()
            line = string[4:].lstrip()
            if line: return t,Kind_Selector(line)
            return t,None
        elif string[:7].upper()=='COMPLEX':
            t = string[:7].upper()
            line = string[7:].lstrip()
            if line: return t,Kind_Selector(line)
            return t,None
        elif string[:7].upper()=='LOGICAL':
            t = string[:7].upper()
            line = string[7:].lstrip()
            if line: return t,Kind_Selector(line)
            return t,None
        elif string[:9].upper()=='CHARACTER':
            t = string[:9].upper()
            line = string[9:].lstrip()
            if line: return t,Char_Selector(line)
        elif string[:6].upper()=='DOUBLE':
            line = string[6:].lstrip().upper()
            if line=='COMPLEX':
                return 'DOUBLE COMPLEX',None
            if line=='PRECISION':
                return 'DOUBLE PRECISION',None
        elif string.upper()=='BYTE':
            return 'BYTE',None
        return
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[1] is None: return str(self.items[0])
        return '%s%s' % tuple(self.items)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Kind_Selector(Base):
    """
    <kind-selector> = ( [ KIND = ] <scalar-int-initialization-expr> )
    Extensions:
                      | * <char-length>
    """
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

class Char_Selector(Base):
    """
    <char-selector> = <length-selector>
                      | ( LEN = <type-param-value> , KIND = <scalar-int-initialization-expr> )
                      | ( <type-param-value> , [ KIND = ] <scalar-int-initialization-expr> )
                      | ( KIND = <scalar-int-initialization-expr> [ , LEN = <type-param-value> ] )
    """
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
            for k in Base.findall(v): v = v.replace(k,repmap[k])
            for k in Base.findall(line): line = line.replace(k,repmap[k])
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
    
class Lenght_Selector(Char_Selector):
    """
    <length-selector> = ( [ LEN = ] <type-param-value> )
                        | * <char-length> [ , ]
    """
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

class Char_Length(Base):
    """
    <char-length> = ( <type-param-value> )
                    | scalar-int-literal-constant
    """
    def match(string):
        if string[0]+string[-1] == '()':
            return '(',Type_Param_Value(string[1:-1].strip()),')'
        return Scalar_Int_Literal_Constant(string),
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if len(self.items)==1: return str(self.items[0])
        return '%s%s%s' % tuple(self.items)
    def torepr(self):
        if len(self.items)==1:
            return '%s(%r)' % (self.__class__.__name__, self.items[0])
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.items[0],self.items[1],self.items[2])


Scalar_Int_Expr = Expr
Scalar_Int_Initialization_Expr = Expr

class Type_Param_Value(Base):
    """
    <type-param-value> = <scalar-int-expr>
                       | *
                       | :
    """
    extra_subclasses = [Scalar_Int_Expr]
    def match(string):
        if string in ['*',':']: return string,
        return
    match = staticmethod(match)
    def init(self, value): self.value = value
    def tostr(self): return str(self.value)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.value)

class Derived_Type_Spec(Type_Spec):
    """
    <derived-type-spec> = <type-name> [ ( <type-param-spec-list> ) ]
    <type-param-spec> = [ <keyword> = ] <type-param-value>
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


class Literal_Constant(Constant):
    """
    <literal-constant> = <int-literal-constant>
                         | <real-literal-constant>
                         | <complex-literal-constant>
                         | <logical-literal-constant>
                         | <char-literal-constant>
                         | <boz-literal-constant>
    """

class Int_Literal_Constant(Literal_Constant):
    """
    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    """
    def match(string):
        m = pattern.abs_int_literal_constant_named.match(string)
        if m is None: return
        return m.group('value'),m.group('kind_param')
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_number
    torepr = Base.torepr_number

Scalar_Int_Literal_Constant = Int_Literal_Constant

class Real_Literal_Constant(Literal_Constant):
    """
    """
    def match(string):
        m = pattern.abs_real_literal_constant_named.match(string)
        if m is None: return
        return m.group('value'),m.group('kind_param')
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_number
    torepr = Base.torepr_number

class Complex_Literal_Constant(Literal_Constant):
    """
    <complex-literal-constant> = ( <real-part>, <imag-part> )
    <real-part> = <imag-part> = <signed-int-literal-constant>
                                | <signed-real-literal-constant>
                                | <named-constant>
    """
    def match(string):
        if string[0]+string[-1]!='()': return
        if not pattern.abs_complex_literal_constant.match(string):
            return
        r,i = string[1:-1].split(',')
        return Real_Part(r.strip()), Imag_Part(i.strip())
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return '(%s, %s)' % (self.items[0], self.items[1])
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Real_Part(Base):
    """
    <real-part> = <signed-int-literal-constant>
                  | <signed-real-literal-constant>
                  | <named-constant>
    """
    def match(string):
        clses = [Int_Literal_Constant, Real_Literal_Constant]
        if string[0] in '+-':
            sign = string[0]
            string = string[1:].lstrip()

        else:
            sign = None
            clses.append(Named_Constant)
        obj = None
        for cls in clses:
            try:
                obj = cls(string)
            except NoMatchError:
                pass
        if obj is None:
            return
        return sign, obj
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[0] is None: return str(self.items[1])
        return '%s%s' % (self.items[0], self.items[1])
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Imag_Part(Real_Part):
    """
    <imag-part> = <real-part>
    """
    match = staticmethod(Real_Part.match)
    init = Real_Part.init
    tostr = Real_Part.tostr
    torepr = Real_Part.torepr

class Char_Literal_Constant(Literal_Constant):
    """
    <char-literal-constant> = [ <kind-param> _ ] ' <rep-char> '
                              | [ <kind-param> _ ] \" <rep-char> \"
    """
    def match(string):
        if string[-1] not in '"\'': return
        if string[-1]=='"':
            abs_a_n_char_literal_constant_named = pattern.abs_a_n_char_literal_constant_named2
        else:
            abs_a_n_char_literal_constant_named = pattern.abs_a_n_char_literal_constant_named1
        line, repmap = string_replace_map(string)
        m = abs_a_n_char_literal_constant_named.match(line)
        if m is None: return
        kind_param = m.group('kind_param')
        line = m.group('value')
        for k in Base.findall(line):
            line = line.replace(k,repmap[k])
        return line, kind_param
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[1] is None: return self.items[0]
        return '%s_%s' % (self.items[1], self.items[0])
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Logical_Literal_Constant(Literal_Constant):
    """
    <logical-literal-constant> = .TRUE. [ _ <kind-param> ]
                                 | .FALSE. [ _ <kind-param> ]
    """
    def match(string):
        m = pattern.abs_logical_literal_constant_named.match(string)
        if m is None: return
        return m.group('value'), m.group('kind_param')
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[1] is None: return self.items[0]
        return '%s_%s' % (self.items[0], self.items[1])
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Boz_Literal_Constant(Literal_Constant):
    """
    <boz-literal-constant> = <binary-constant>
                             | <octal-constant>
                             | <hex-constant>
    """


class Binary_Constant(Boz_Literal_Constant):
    """
    <binary-constant> = B ' <digit> [ <digit> ]... '
                        | B \" <digit> [ <digit> ]... \"
    """
    def match(string):
        if pattern.abs_binary_constant.match(string): return (string,)
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string

class Octal_Constant(Boz_Literal_Constant):
    """
    <octal-constant> = O ' <digit> [ <digit> ]... '
                       | O \" <digit> [ <digit> ]... \"
    """
    def match(string):
        if pattern.abs_octal_constant.match(string): return (string,)
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string

class Hex_Constant(Boz_Literal_Constant):
    """
    <hex-constant> = Z ' <digit> [ <digit> ]... '
                     | Z \" <digit> [ <digit> ]... \"
    """
    def match(string):
        if pattern.abs_hex_constant.match(string): return (string,)
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string

class Named_Constant(Constant):
    """
    <named-constant> = <name>
    """


class Object_Name(Designator):
    """
    <object-name> = <name>
    """

class Function_Reference(Primary):
    """
    <function-reference> = <procedure-designator> ( [ <actual-arg-spec-list> ] )
    """
    def match(string):
        if string[-1] != ')': return None
        line, repmap = string_replace_map(string)
        i = line.rfind('(')
        if i == -1: return
        lhs = line[:i].lstrip()
        if not lhs: return
        rhs = line[i+1:-1].strip()
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        lhs_obj = Procedure_Designator(lhs)
        if rhs:
            rhs_obj = Actual_Arg_Spec_List(rhs)
        else:
            rhs_obj = None
        return lhs_obj,'(',rhs_obj,')'
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[2] is None: return '%s()' % (self.items[0])
        return '%s(%s)' % (self.items[0],self.items[2])
    torepr = Base.torepr_list
        
    
class Procedure_Designator(Base):
    """
    <procedure-designator> = <procedure-name>
                           | <proc-component-ref>
                           | <data-ref> % <binding-name>
    """
        
class Actual_Arg_Spec_List(Base):
    """
    <actual-arg-spec-list> = <actual-arg-spec> [ , <actual-arg-spec> ]...
    """
    def match(string):
        return Base.match_list_of(Actual_Arg_Spec, string)
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_list
    torepr = Base.torepr_list

def try_cls(cls, string):
    try:
        return cls(string)
    except NoMatchError:
        pass
    return

class Actual_Arg_Spec(Base):
    """
    <actual-arg-spec> = [ <keyword> = ] <actual-arg>
    """    
    def match(string):
        if pattern.keyword_equal.match(string):
            i = string.find('=')
            assert i!=-1,`string`
            kw = Keyword(string[:i].rstrip())
            string = string[i+1:].lstrip()
        else:
            kw = None
        return kw, Actual_Arg(string)

    match = staticmethod(match)
    def init(self,kw,arg):
        self.keyword = kw
        self.actual_arg =arg
        return
    def tostr(self):
        if self.keyword is not None:
            return '%s = %s' % (self.keyword, self.actual_arg)
        return str(self.actual_arg)
    def torepr(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.keyword, self.actual_arg)

class Keyword(Base):
    """
    <keyword> = <name>
    """

class Actual_Arg(Actual_Arg_Spec):
    """
    <actual-arg> = <expr>
                 | <variable>
                 | <procedure-name>
                 | <proc-component-ref>
                 | <alt-return-spec>
    <alt-return-spec> = * <label>
    <variable> = <designator>
    <proc-component-ref> = <variable> % <procedure-component-name>
    """
    extra_subclasses = [Expr]
    
class Procedure_Name(Procedure_Designator, Actual_Arg):
    """
    <procedure-name> = <name>
    """
    
class Parenthesis(Primary):
    """
    ( <expr> )
    """
    def match(string):
        if string[0]+string[-1]=='()':
            return '(',Expr(string[1:-1].strip()),')'
        return None
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand

class Name(Object_Name, Named_Constant, Procedure_Name, Keyword, Ac_Do_Variable):
    """
    <name> = <letter> [ <alphanumeric_character> ]...
    """
    def match(string):
        if pattern.abs_name.match(string):
            return string,
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string
    
ClassType = type(Base)
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base):
        extra_subclasses = cls.__dict__.get('extra_subclasses',[])
        if extra_subclasses:
            try:
                l = Base.subclasses[cls.__name__]
            except KeyError:
                Base.subclasses[cls.__name__] = l = []
            l.extend(extra_subclasses)
        for basecls in cls.__bases__:
            if basecls is Base: continue
            if issubclass(basecls, Base):
                try:
                    Base.subclasses[basecls.__name__].append(cls)
                except KeyError:
                    Base.subclasses[basecls.__name__] = [cls]
#import pprint
#pprint.pprint(Base.subclasses)
