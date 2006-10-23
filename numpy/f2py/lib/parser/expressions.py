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
            if hasattr(cls, 'init'):
                obj.init(*result)
            #if cls.__dict__.has_key('init'):
            #    obj.init(*result)
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
        return self.tostr()
        if self.__class__.__dict__.has_key('tostr'):
            return self.tostr()
        return repr(self)

    def __repr__(self):
        return self.torepr()
        if self.__class__.__dict__.has_key('torepr'):
            return self.torepr()
        return '%s(%r)' % (self.__class__.__name__, self.string)

class SequenceBase(Base):

    def match(separator, subcls, string):
        line, repmap = string_replace_map(string)
        lst = []
        for p in line.split(separator):
            p = p.strip()
            for k in Base.findall(p):
                p = p.replace(k,repmap[k])
            lst.append(subcls(p))
        return separator, tuple(lst)
    match = staticmethod(match)
    def init(self, separator, items):
        self.separator = separator
        self.items = items
    def tostr(self): return (self.separator+' ').join(map(str, self.items))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.separator, self.items)
    
class UnaryOpBase(Base):
    def init(self, op, rhs):
        self.op = op
        self.rhs = rhs
        return    
    def tostr(self):
        return '%s %s' % (self.op, self.rhs)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__,self.op, self.rhs)
    def match(op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.lsplit(line)
        if t is None:
            return rhs_cls(string)
        lhs, op, rhs = t
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        assert not lhs,`lhs`
        rhs_obj = rhs_cls(rhs)
        return t[1], rhs_obj
    match = staticmethod(match)

class BinaryOpBase(Base):
    """
    <..> = <lhs> <op> <rhs>
    """
    def match(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.rsplit(line)
        if t is None or len(t)!=3: return
        lhs, op, rhs = t
        if lhs is None: return
        if rhs is None: return
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        lhs_obj = lhs_cls(lhs)
        rhs_obj = rhs_cls(rhs)
        return lhs_obj, t[1], rhs_obj
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

class RBinaryOpBase(BinaryOpBase):
    """
    <..> = [ <lhs> <op> ] <rhs>
    """
    def match(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.rsplit(line)
        if t is None:
            return rhs_cls(string)
        lhs, op, rhs = t
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        lhs_obj = lhs_cls(lhs)
        rhs_obj = rhs_cls(rhs)
        return lhs_obj, t[1], rhs_obj
    match = staticmethod(match)

class LBinaryOpBase(BinaryOpBase):
    """
    <..> = <lhs> [ <op> <rhs> ]
    """
    def match(lhs_cls, op_pattern, rhs_cls, string):
        line, repmap = string_replace_map(string)
        t = op_pattern.lsplit(line)
        if t is None:
            return lhs_cls(string)
        lhs, op, rhs = t
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        lhs_obj = lhs_cls(lhs)
        rhs_obj = rhs_cls(rhs)
        return lhs_obj, t[1], rhs_obj
    match = staticmethod(match)

class Expr(RBinaryOpBase):
    """
    <expr> = [ <expr> <defined-binary-op> ] <level-5-expr>
    <defined-binary-op> = . <letter> [ <letter> ]... .
    TODO: defined_binary_op must not be intrinsic_binary_op!!
    """
    #subclass_names = ['Level_5_Expr']
    def match(string):
        return RBinaryOpBase.match(Expr, pattern.defined_binary_op.named(), Level_5_Expr,
                                 string)
    match = staticmethod(match)

class Level_5_Expr(RBinaryOpBase):
    """
    <level-5-expr> = [ <level-5-expr> <equiv-op> ] <equiv-operand>
    <equiv-op> = .EQV.
               | .NEQV.
    """
    #subclass_names = ['Equiv_Operand']
    def match(string):
        return RBinaryOpBase.match(\
            Level_5_Expr,pattern.equiv_op.named(),Equiv_Operand,string)
    match = staticmethod(match)
    
class Equiv_Operand(RBinaryOpBase):
    """
    <equiv-operand> = [ <equiv-operand> <or-op> ] <or-operand>
    <or-op>  = .OR.
    """
    #subclass_names = ['Or_Operand']
    def match(string):
        return RBinaryOpBase.match(\
            Equiv_Operand,pattern.or_op.named(),Or_Operand,string)
    match = staticmethod(match)
    
class Or_Operand(RBinaryOpBase):
    """
    <or-operand> = [ <or-operand> <and-op> ] <and-operand>    
    <and-op> = .AND.
    """
    #subclass_names = ['And_Operand']
    def match(string):
        return RBinaryOpBase.match(\
            Or_Operand,pattern.and_op.named(),And_Operand,string)
    match = staticmethod(match)
    
class And_Operand(UnaryOpBase):
    """
    <and-operand> = [ <not-op> ] <level-4-expr>
    <not-op> = .NOT.
    """
    #subclass_names = ['Level_4_Expr']
    def match(string):
        return UnaryOpBase.match(\
            pattern.not_op.named(),Level_4_Expr,string)
    match = staticmethod(match)
    
class Level_4_Expr(RBinaryOpBase):
    """
    <level-4-expr> = [ <level-3-expr> <rel-op> ] <level-3-expr>
    <rel-op> = .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. | == | /= | < | <= | > | >=
    """
    #subclass_names = ['Level_3_Expr']
    def match(string):
        return RBinaryOpBase.match(\
            Level_3_Expr,pattern.rel_op.named(),Level_3_Expr,string)
    match = staticmethod(match)
    
class Level_3_Expr(RBinaryOpBase):
    """
    <level-3-expr> = [ <level-3-expr> <concat-op> ] <level-2-expr>
    <concat-op>    = //
    """
    #subclass_names = ['Level_2_Expr']
    def match(string):
        return RBinaryOpBase.match(\
            Level_3_Expr,pattern.concat_op.named(),Level_2_Expr,string)
    match = staticmethod(match)
    
class Level_2_Expr(BinaryOpBase):
    """
    <level-2-expr> = [ [ <level-2-expr> ] <add-op> ] <add-operand>
    <add-op>   = +
                 | -
    """
    subclass_names = ['Add_Operand']
    def match(string):
        lhs_cls, op_pattern, rhs_cls = Level_2_Expr,pattern.add_op.named(),Add_Operand
        line, repmap = string_replace_map(string)
        t = op_pattern.rsplit(line)
        if t is None:
            return rhs_cls(string)
        lhs, op, rhs = t
        if lhs is not None:
            for k in Base.findall(lhs):
                lhs = lhs.replace(k, repmap[k])
            lhs_obj = lhs_cls(lhs)
        else:
            lhs_obj = None
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        rhs_obj = rhs_cls(rhs)
        return lhs_obj, t[1], rhs_obj
    match = staticmethod(match)
    
class Add_Operand(RBinaryOpBase):
    """
    <add-operand> = [ <add-operand> <mult-op> ] <mult-operand>
    <mult-op>  = *
                 | /
    """
    #subclass_names = ['Mult_Operand']
    def match(string):
        return RBinaryOpBase.match(\
            Add_Operand,pattern.mult_op.named(),Mult_Operand,string)
    match = staticmethod(match)
    
class Mult_Operand(LBinaryOpBase):
    """
    <mult-operand> = <level-1-expr> [ <power-op> <mult-operand> ]
    <power-op> = **
    """
    #subclass_names = ['Level_1_Expr']
    def match(string):
        return LBinaryOpBase.match(\
            Level_1_Expr,pattern.power_op.named(),Mult_Operand,string)
    match = staticmethod(match)
    
class Level_1_Expr(UnaryOpBase):
    """
    <level-1-expr> = [ <defined-unary-op> ] <primary>
    <defined-unary-op> = . <letter> [ <letter> ]... .
    """
    #subclass_names = ['Primary']
    def match(string):
        return UnaryOpBase.match(\
            pattern.defined_unary_op.named(),Primary,string)
    match = staticmethod(match)
    
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
    """
    subclass_names = ['Constant', 'Designator','Array_Constructor','Structure_Constructor',
                      'Function_Reference', 'Type_Param_Inquiry', 'Type_Param_Name', 'Parenthesis']

class Type_Param_Name(Base):
    """
    <type-param-name> = <name>
    """
    subclass_names = ['Name']

class Type_Param_Inquiry(BinaryOpBase):
    """
    <type-param-inquiry> = <designator> % <type-param-name>
    """
    subclass_names = []
    def match(string):
        return BinaryOpBase.match(\
            Designator, pattern.percent_op.named(), Type_Param_Name, string)
    match = staticmethod(match)

class Structure_Constructor(Base):
    """
    <structure-constructor> = <derived-type-spec> ( [ <component-spec-list> ] )
                            | [ <keyword> = ] <component-data-source>
    """
    subclass_names = []
    def match(string):
        line, repmap = string_replace_map(string)
        if line[-1]==')':
            i = line.rfind('(')
            if i==-1: return
            specline = line[:i].rstrip()
            for k in Base.findall(specline):
                specline = specline.replace(k,repmap[k])
            try:
                spec = Derived_Type_Spec(specline)
            except NoMatchError:
                spec = None
            if spec is not None:
                l = line[i+1:-1].strip()
                for k in Base.findall(l):
                    l = l.replace(k,repmap[k])
                if not l: return spec,'(',None,')'
                return spec,'(',Component_Spec_List(l),')'
        if '=' not in string: return None, Component_Data_Source(string)
        lhs,rhs = string.split('=',1)
        return Keyword(lhs.rstrip()), Component_Data_Source(rhs.lstrip())        
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if len(self.items)==4:
            if self.items[2] is None: return '%s()' % (self.items[0])
            return '%s(%s)' % (self.items[0],self.items[2])
        if self.items[0] is None:
            return str(self.items[1])
        return '%s = %s' % (self.items[0], self.items[1])
    torepr = Base.torepr_list

class Component_Data_Source(Base):
    """
    <component-data-source> = <expr>
                              | <data-target>
                              | <proc-target>
    """
    subclass_names = ['Expr','Data-Target','Proc-Target']

class Data_Target(Base):
    """
    <data-target> = <variable>
                    | <expr>
    """
    subclass_names = ['Variable','Expr']

class Proc_Target(Base):
    """
    <proc-target> = <expr>
                    | <procedure-name>
                    | <proc-component-ref>
    """
    subclass_names = ['Expr','Procedure_Name','Proc_Component_Ref']

class Proc_Component_Ref(BinaryOpBase):
    """
    <proc-component-ref> = <variable> % <procedure-component-name>
    """
    subclass_names = []
    def match(string):
        return BinaryOpBase.match(\
            Variable, pattern.percent_op.named(), Procedure_Component_Name, string)            
    match = staticmethod(match)

class Component_Spec(Base):
    """
    <component-spec> = [ <keyword> = ] <component-data-source>
    """
    subclass_names = []
    def match(string):
        if '=' not in string: return None, Component_Data_Source(string)
        lhs,rhs = string.split('=',1)
        return Keyword(lhs.rstrip()), Component_Data_Source(rhs.lstrip())
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[0] is None: return str(self.items[1])
        return '%s = %s' % tuple(self.items)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Component_Spec_List(Base):
    """
    <component-spec-list> = <component-spec> [ , <component-spec> ]...
    """
    subclass_names = []
    def match(string):
        return Base.match_list_of(Component_Spec, string)
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_list
    torepr = Base.torepr_list

class Array_Constructor(Base):
    """
    <array-constructor> = (/ <ac-spec> /)
                          | <left-square-bracket> <ac-spec> <right-square-bracket>

    """
    subclass_names = []
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
    subclass_names = []
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
    subclass_names = []
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
    subclass_names = ['Ac_Implied_Do','Expr']
    extra_subclasses = [Expr]

class Ac_Implied_Do(Base):
    """
    <ac-implied-do> = ( <ac-value-list> , <ac-implied-do-control> )
    """
    subclass_names = []
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
    subclass_names = []
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
    subclass_names = ['Expr']
    extra_subclasses = [Expr]

class Ac_Do_Variable(Base):
    """
    <ac-do-variable> = <scalar-int-variable>
    <ac-do-variable> shall be a named variable    
    """
    subclass_names = ['Name']

class Type_Spec(Base):
    """
    <type-spec> = <intrinsic-type-spec>
                  | <derived-type-spec>
    """
    subclass_names = ['Intrinsic_Type_Spec', 'Derived_Type_Spec']

class Intrinsic_Type_Spec(Base):
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
    subclass_names = []
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
    subclass_names = ['Lenght_Selector']
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
    
class Lenght_Selector(Base):
    """
    <length-selector> = ( [ LEN = ] <type-param-value> )
                        | * <char-length> [ , ]
    """
    subclass_names = []
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
                    | <scalar-int-literal-constant>
    """
    subclass_names = []
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


class Scalar_Int_Expr(Base):
    """
    <scalar-int-expr> = <expr>
    """
    subclass_names = ['Expr']

class Scalar_Int_Initialization_Expr(Base):
    """
    <scalar-int-initialization-expr> = <expr>
    """
    subclass_names = ['Expr']

class Type_Param_Value(Base):
    """
    <type-param-value> = <scalar-int-expr>
                       | *
                       | :
    """
    subclass_names = ['Scalar_Int_Expr']
    extra_subclasses = [Scalar_Int_Expr]
    def match(string):
        if string in ['*',':']: return string,
        return
    match = staticmethod(match)
    def init(self, value): self.value = value
    def tostr(self): return str(self.value)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.value)

class Derived_Type_Spec(Base):
    """
    <derived-type-spec> = <type-name> [ ( <type-param-spec-list> ) ]
    """
    subclass_names = []
    def match(string):
        i = string.find('(')
        if i==-1: return Type_Name(string),None
        if string[-1] != ')': return
        return Type_Name(string[:i].rstrip()), Type_Param_Spec_List(string[i+1:-1].strip())
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[1] is None: return str(self.items[0])
        return '%s(%s)' % tuple(self.items)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Type_Param_Spec(Base):
    """
    <type-param-spec> = [ <keyword> = ] <type-param-value>
    """
    subclass_names = []
    def match(string):
        if '=' not in string: return None, Type_Param_Value(string)
        lhs,rhs = string.split('=',1)
        return Keyword(lhs.rstrip()), Type_Param_Value(rhs.lstrip())
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self):
        if self.items[0] is None: return str(self.items[1])
        return '%s = %s' % tuple(self.items)
    def torepr(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.items[0],self.items[1])

class Type_Param_Spec_List(Base):
    """
    <type-param-spec> = <type-param> [ , <type-param> ]...
    """
    subclass_names = []
    def match(string):
        return Base.match_list_of(Type_Param_Spec, string)
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_list
    torepr = Base.torepr_list


class Constant(Base):
    """
    <constant> = <literal-constant>
                 | <named-constant>
    """
    subclass_names = ['Literal_Constant','Named_Constant']

class Designator(Base):
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
    subclass_names = ['Object_Name','Array_Element','Array_Section','Structure_Component',
                      'Substring'
                      ]

class Literal_Constant(Base):
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

class Int_Literal_Constant(Base):
    """
    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    """
    subclass_names = []
    def match(string):
        m = pattern.abs_int_literal_constant_named.match(string)
        if m is None: return
        return m.group('value'),m.group('kind_param')
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_number
    torepr = Base.torepr_number

class Scalar_Int_Literal_Constant(Base):
    """
    <scalar-int-literal-constant> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']

class Real_Literal_Constant(Base):
    """
    """
    subclass_names = []
    def match(string):
        m = pattern.abs_real_literal_constant_named.match(string)
        if m is None: return
        return m.group('value'),m.group('kind_param')
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_number
    torepr = Base.torepr_number

class Complex_Literal_Constant(Base):
    """
    <complex-literal-constant> = ( <real-part>, <imag-part> )
    <real-part> = <imag-part> = <signed-int-literal-constant>
                                | <signed-real-literal-constant>
                                | <named-constant>
    """
    subclass_names = []
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
    subclass_names = []
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
    subclass_names = []
    match = staticmethod(Real_Part.match)
    init = Real_Part.init
    tostr = Real_Part.tostr
    torepr = Real_Part.torepr

class Char_Literal_Constant(Base):
    """
    <char-literal-constant> = [ <kind-param> _ ] ' <rep-char> '
                              | [ <kind-param> _ ] \" <rep-char> \"
    """
    subclass_names = []
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

class Logical_Literal_Constant(Base):
    """
    <logical-literal-constant> = .TRUE. [ _ <kind-param> ]
                                 | .FALSE. [ _ <kind-param> ]
    """
    subclass_names = []
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

class Boz_Literal_Constant(Base):
    """
    <boz-literal-constant> = <binary-constant>
                             | <octal-constant>
                             | <hex-constant>
    """
    subclass_names = ['Binary_Constant','Octal_Constant','Hex_Constant']

class Binary_Constant(Base):
    """
    <binary-constant> = B ' <digit> [ <digit> ]... '
                        | B \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string):
        if pattern.abs_binary_constant.match(string): return (string,)
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string

class Octal_Constant(Base):
    """
    <octal-constant> = O ' <digit> [ <digit> ]... '
                       | O \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string):
        if pattern.abs_octal_constant.match(string): return (string,)
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string

class Hex_Constant(Base):
    """
    <hex-constant> = Z ' <digit> [ <digit> ]... '
                     | Z \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string):
        if pattern.abs_hex_constant.match(string): return (string,)
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string

class Named_Constant(Base):
    """
    <named-constant> = <name>
    """
    subclass_names = ['Name']

class Object_Name(Base):
    """
    <object-name> = <name>
    """
    subclass_names = ['Name']
    
class Function_Reference(Base):
    """
    <function-reference> = <procedure-designator> ( [ <actual-arg-spec-list> ] )
    """
    subclass_names = []
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
    subclass_names = ['Procedure_Name','Proc_Component_Ref']
    def match(string):
        return BinaryOpBase.match(\
            Data_Ref, pattern.percent_op.named(),  Binding_Name, string)
    match = staticmethod(match)

class Part_Ref(Base):
    """
    <part-ref> = <part-name> [ ( <section-subscript-list> ) ]
    """
    subclass_names = ['Part_Name']
    def match(string):
        if string.endswith(')'):
            i = string.find('(')
            if i==-1: return
            return Part_Name(string[:i].rstrip()),'(',Section_Subscript_List(string[i+1:-1].strip()),')'
        return
    match = staticmethod(match)
    init = Base.init_list

class Part_Name(Base):
    """
    <part-name> = <name>
    """
    subclass_names = ['Name']
    
class Binding_Name(Base):
    """
    <binding-name> = <name>
    """
    subclass_names = ['Name']

class Data_Ref(SequenceBase):
    """
    <data-ref> = <part-ref> [ % <part-ref> ]...
    """
    subclass_names = []
    def match(string):
        return SequenceBase.match(r'%', Part_Ref, string)
    match = staticmethod(match)

class Actual_Arg_Spec_List(Base):
    """
    <actual-arg-spec-list> = <actual-arg-spec> [ , <actual-arg-spec> ]...
    """
    subclass_names = []
    def match(string):
        return Base.match_list_of(Actual_Arg_Spec, string)
    match = staticmethod(match)
    init = Base.init_list
    tostr = Base.tostr_list
    torepr = Base.torepr_list

class Actual_Arg_Spec(Base):
    """
    <actual-arg-spec> = [ <keyword> = ] <actual-arg>
    """
    subclass_names = []
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
    subclass_names = ['Name']

class Type_Name(Base):
    """
    <type-name> = <name>
    <type-name> shall not be DOUBLEPRECISION or the name of intrinsic type
    """
    subclass_names = []
    def match(string):
        if pattern.abs_intrinsic_type_name.match(string): return
        return Name(string)
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string        
    
class Actual_Arg(Base):
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
    subclass_names = ['Expr','Variable','Procedure_Name','Alt_Return_Spec']
    extra_subclasses = [Expr]

class Variable(Base):
    """
    <variable> = <designator>
    """
    subclass_names = ['Designator']
    extra_subclasses = [Designator]

class Procedure_Component_Name(Base):
    """
    <procedure-component-name> = <name>
    """
    subclass_names = ['Name']

class Procedure_Name(Base):
    """
    <procedure-name> = <name>
    """
    subclass_names = ['Name']
    
class Parenthesis(Base):
    """
    <parenthesis> = ( <expr> )
    """
    subclass_names = []
    def match(string):
        if string[0]+string[-1]=='()':
            return '(',Expr(string[1:-1].strip()),')'
        return None
    match = staticmethod(match)
    init = Base.init_binary_operand
    tostr = Base.tostr_binary_operand
    torepr = Base.torepr_binary_operand

class Name(Base):
    """
    <name> = <letter> [ <alphanumeric_character> ]...
    """
    subclass_names = []
    def match(string):
        if pattern.abs_name.match(string):
            return string,
        return
    match = staticmethod(match)
    tostr = Base.tostr_string
    torepr = Base.torepr_string
    
ClassType = type(Base)
Base_classes = {}
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base):
        Base_classes[cls.__name__] = cls

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

print 
for clsname in dir():
    break
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

for cls in Base_classes.values():
    subclasses = Base.subclasses.get(cls.__name__,[])
    subclasses_names = [c.__name__ for c in subclasses]
    subclass_names = getattr(cls,'subclass_names', [])
    for n in subclasses_names:
        if n not in subclass_names:
            print '%s needs to be added to %s subclasses_name list' % (n,cls.__name__)
    for n in subclass_names:
        if n not in subclasses_names:
            print '%s needs to be added to %s subclass_name list' % (n,cls.__name__)
#import pprint
#pprint.pprint(Base.subclasses)
