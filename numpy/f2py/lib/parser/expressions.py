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

############################# Base classes ################################

class NoMatchError(Exception):
    pass

class Base(object):

    subclasses = {}

    def __new__(cls, string):
        #print '__new__:',cls.__name__,`string`
        match = cls.__dict__.get('match', None)
        if match is not None:
            result = cls.match(string)
        else:
            result = None
        #print '__new__:result:',cls.__name__,`string,result`
        if isinstance(result, tuple):
            obj = object.__new__(cls)
            obj.string = string
            if hasattr(cls, 'init'):
                obj.init(*result)
            return obj
        elif isinstance(result, Base):
            return result
        elif result is None:
            for subcls in Base.subclasses.get(cls.__name__,[]):
                #print cls.__name__,subcls.__name__,`string`
                try:
                    return subcls(string)
                except NoMatchError:
                    pass
        else:
            raise AssertionError,`result`
        raise NoMatchError,'%s: %r' % (cls.__name__, string)

    findall = staticmethod(re.compile(r'(_F2PY_STRING_CONSTANT_\d+_|F2PY_EXPR_TUPLE_\d+)').findall)
    
    def init_list(self, *items):
        self.items = items
        return

    def tostr_list(self):
        return ', '.join(map(str,self.items))

    def torepr_list(self):
        return '%s(%s)' % (self.__class__.__name__,', '.join(map(repr,self.items)))

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
    """
    <sequence-base> = <obj>, <obj> [ , <obj> ]...
    """
    def match(separator, subcls, string):
        line, repmap = string_replace_map(string)
        if separator not in line:
            return
        lst = []
        for p in line.split(separator):
            p = p.strip()
            for k in Base.findall(p):
                p = p.replace(k,repmap[k])
            lst.append(subcls(p))
        #if len(lst)==1: return lst[0]
        return separator, tuple(lst)
    match = staticmethod(match)
    def init(self, separator, items):
        self.separator = separator
        self.items = items
        return
    def tostr(self):
        s = self.separator
        if s==',': s = s + ' '
        else: s = ' ' + s + ' '
        return s.join(map(str, self.items))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.separator, self.items)
    
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

class BinaryOpBase(Base):
    """
    <binary-op-base> = <lhs> <op> <rhs>
    <op> is searched from right by default.
    """
    def match(lhs_cls, op_pattern, rhs_cls, string, right=True):
        line, repmap = string_replace_map(string)
        if right:
            t = op_pattern.rsplit(line)
        else:
            t = op_pattern.lsplit(line)
        if t is None or len(t)!=3: return
        lhs, op, rhs = t
        if not lhs: return
        if not rhs: return
        op = op.upper()
        for k in Base.findall(lhs):
            lhs = lhs.replace(k, repmap[k])
        for k in Base.findall(rhs):
            rhs = rhs.replace(k, repmap[k])
        lhs_obj = lhs_cls(lhs)
        rhs_obj = rhs_cls(rhs)
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

class KeywordValueBase(BinaryOpBase):
    """
    <keyword-value-base> = <keyword> = <rhs>
    """
    def match(cls, string):
        if '=' not in string: return
        lhs,rhs = string.split('=',1)
        return Keyword(lhs.rstrip()),'=',cls(rhs.lstrip())
    match = staticmethod(match)
    
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

class CallBase(Base):
    """
    <call-base> = <lhs> ( [ <rhs> ] )
    """
    def match(lhs_cls, rhs_cls, string):
        if not string.endswith(')'): return
        line, repmap = string_replace_map(string)
        i = line.find('(')
        if i==-1: return
        lhs = line[:i].rstrip()
        if not lhs: return
        rhs = line[i+1:-1].strip()
        for k in Base.findall(lhs):
            lhs = lhs.replace(k,repmap[k])
        if rhs:
            for k in Base.findall(rhs):
                rhs = rhs.replace(k,repmap[k])
            return lhs_cls(lhs), rhs_cls(rhs)
        return lhs_cls(lhs), None
    match = staticmethod(match)
    def init(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        return
    def tostr(self):
        if self.rhs is None: return '%s()' % (self.lhs)
        return '%s(%s)' % (self.lhs, self.rhs)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.lhs, self.rhs)

class StringBase(Base):
    """
    <string-base> = <xyz>
    """
    def match(pattern, string):
        if pattern.match(string): return string,
        return
    match = staticmethod(match)
    def init(self, string):
        self.string = string
        return
    def tostr(self): return str(self.string)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.string)


##################################################

class Name(StringBase):
    """
    <name> = <letter> [ <alphanumeric_character> ]...
    """
    subclass_names = []
    def match(string): return StringBase.match(pattern.abs_name, string)
    match = staticmethod(match)

class Binary_Constant(StringBase):
    """
    <binary-constant> = B ' <digit> [ <digit> ]... '
                        | B \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string): return StringBase.match(pattern.abs_binary_constant, string)
    match = staticmethod(match)
    def init(self, string):
        self.string = string.upper()
        return

class Octal_Constant(StringBase):
    """
    <octal-constant> = O ' <digit> [ <digit> ]... '
                       | O \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string): return StringBase.match(pattern.abs_octal_constant, string)
    match = staticmethod(match)
    def init(self, string):
        self.string = string.upper()
        return

class Hex_Constant(StringBase):
    """
    <hex-constant> = Z ' <digit> [ <digit> ]... '
                     | Z \" <digit> [ <digit> ]... \"
    """
    subclass_names = []
    def match(string): return StringBase.match(pattern.abs_hex_constant, string)
    match = staticmethod(match)
    def init(self, string):
        self.string = string.upper()
        return

class Int_Literal_Constant(NumberBase):
    """
    <int-literal-constant> = <digit-string> [ _ <kind-param> ]
    """
    subclass_names = []
    def match(string):
        return NumberBase.match(pattern.abs_int_literal_constant_named, string)
    match = staticmethod(match)

class Signed_Int_Literal_Constant(NumberBase):
    """
    <signed-int-literal-constant> = [ <sign> ] <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant'] # never used because sign is included in pattern
    def match(string):
        return NumberBase.match(pattern.abs_signed_int_literal_constant_named, string)
    match = staticmethod(match)

class Real_Literal_Constant(NumberBase):
    """
    """
    subclass_names = []
    def match(string):
        return NumberBase.match(pattern.abs_real_literal_constant_named, string)
    match = staticmethod(match)

class Signed_Real_Literal_Constant(NumberBase):
    """
    <signed-real-literal-constant> = [ <sign> ] <real-literal-constant>
    """
    subclass_names = ['Real_Literal_Constant'] # never used
    def match(string):
        return NumberBase.match(pattern.abs_signed_real_literal_constant_named, string)
    match = staticmethod(match)

class Logical_Literal_Constant(NumberBase):
    """
    <logical-literal-constant> = .TRUE. [ _ <kind-param> ]
                                 | .FALSE. [ _ <kind-param> ]
    """
    subclass_names = []
    def match(string):
        return NumberBase.match(pattern.abs_logical_literal_constant_named, string)    
    match = staticmethod(match)

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
        if not m: return
        kind_param = m.group('kind_param')
        line = m.group('value')
        for k in Base.findall(line):
            line = line.replace(k,repmap[k])
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

#####

    
class Expr(BinaryOpBase):
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

class Level_5_Expr(BinaryOpBase):
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
    
class Equiv_Operand(BinaryOpBase):
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
    
class Or_Operand(BinaryOpBase):
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
    
class And_Operand(UnaryOpBase):
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
    
class Level_4_Expr(BinaryOpBase):
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
    
class Level_3_Expr(BinaryOpBase):
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

class Level_2_Expr(BinaryOpBase):
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

class Level_2_Unary_Expr(UnaryOpBase):
    """
    <level-2-unary-expr> = [ <add-op> ] <add-operand>
    """
    subclass_names = ['Add_Operand']
    use_names = []
    def match(string): return UnaryOpBase.match(pattern.add_op.named(),Add_Operand,string)
    match = staticmethod(match)

class Add_Operand(BinaryOpBase):
    """
    <add-operand> = [ <add-operand> <mult-op> ] <mult-operand>
    <mult-op>  = *
                 | /
    """
    subclass_names = ['Mult_Operand']
    use_names = ['Add_Operand','Mult_Operand']
    def match(string):
        return BinaryOpBase.match(\
            Add_Operand,pattern.mult_op.named(),Mult_Operand,string)
    match = staticmethod(match)
    
class Mult_Operand(BinaryOpBase):
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
    
class Level_1_Expr(UnaryOpBase):
    """
    <level-1-expr> = [ <defined-unary-op> ] <primary>
    <defined-unary-op> = . <letter> [ <letter> ]... .
    """
    subclass_names = ['Primary']
    use_names = []
    def match(string):
        if pattern.non_defined_binary_op.match(string):
            raise NoMatchError,`string`
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
    subclass_names = ['Constant', 'Parenthesis', 'Designator','Array_Constructor','Structure_Constructor',
                      'Function_Reference', 'Type_Param_Inquiry', 'Type_Param_Name', 
                       ]

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
    use_names = ['Designator','Type_Param_Name']
    def match(string):
        return BinaryOpBase.match(\
            Designator, pattern.percent_op.named(), Type_Param_Name, string)
    match = staticmethod(match)

class Structure_Constructor_2(KeywordValueBase):
    """
    <structure-constructor-2> = [ <keyword> = ] <component-data-source>
    """
    subclass_names = ['Component_Data_Source']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Component_Data_Source, string)
    match = staticmethod(match)

class Structure_Constructor(CallBase):
    """
    <structure-constructor> = <derived-type-spec> ( [ <component-spec-list> ] )
                            | <structure-constructor-2>
    """
    subclass_names = ['Structure_Constructor_2']
    use_names = ['Derived_Type_Spec', 'Component_Spec_List']
    def match(string): return CallBase.match(Derived_Type_Spec, Component_Spec_List, string)
    match = staticmethod(match)

class Component_Data_Source(Base):
    """
    <component-data-source> = <expr>
                              | <data-target>
                              | <proc-target>
    """
    subclass_names = ['Proc_Target', 'Data_Target', 'Expr']

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
    subclass_names = ['Proc_Component_Ref', 'Procedure_Name', 'Expr']

class Proc_Component_Ref(BinaryOpBase):
    """
    <proc-component-ref> = <variable> % <procedure-component-name>
    """
    subclass_names = []
    use_names = ['Variable','Procedure_Component_Name']
    def match(string):
        return BinaryOpBase.match(\
            Variable, pattern.percent_op.named(), Procedure_Component_Name, string)            
    match = staticmethod(match)

class Component_Spec(KeywordValueBase):
    """
    <component-spec> = [ <keyword> = ] <component-data-source>
    """
    subclass_names = ['Component_Data_Source']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Component_Data_Source, string)
    match = staticmethod(match)

class Component_Spec_List(SequenceBase):
    """
    <component-spec-list> = <component-spec> [ , <component-spec> ]...
    """
    subclass_names = ['Component_Spec']
    use_names = []
    def match(string): return SequenceBase.match(r',', Component_Spec, string)
    match = staticmethod(match)

class Array_Constructor(BracketBase):
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

class Ac_Spec(Base):
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

class Ac_Value_List(SequenceBase):
    """
    <ac-value-list> = <ac-value> [ , <ac-value> ]...
    """
    subclass_names = ['Ac_Value']
    use_names = []
    def match(string): return SequenceBase.match(r',', Ac_Value, string)
    match = staticmethod(match)

class Ac_Value(Base):
    """
    <ac-value> = <expr>
                 | <ac-implied-do>
    """
    subclass_names = ['Ac_Implied_Do','Expr']

class Ac_Implied_Do(Base):
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

class Scalar_Int_Expr(Base):
    """
    <scalar-int-expr> = <expr>
    """
    subclass_names = ['Expr']

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
    use_names = ['Kind_Selector','Char_Selector']
        
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

class Char_Selector(Base):
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
    
class Length_Selector(Base):
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

class Char_Length(BracketBase):
    """
    <char-length> = ( <type-param-value> )
                    | <scalar-int-literal-constant>
    """
    subclass_names = ['Scalar_Int_Literal_Constant']
    use_names = ['Type_Param_Value']
    def match(string): return BracketBase.match('()',Type_Param_Value, string)
    match = staticmethod(match)

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
    use_names = []
    def match(string):
        if string in ['*',':']: return string,
        return
    match = staticmethod(match)
    def init(self, value): self.value = value
    def tostr(self): return str(self.value)
    def torepr(self): return '%s(%r)' % (self.__class__.__name__, self.value)

class Derived_Type_Spec(CallBase):
    """
    <derived-type-spec> = <type-name> [ ( <type-param-spec-list> ) ]
    """
    subclass_names = ['Type_Name']
    use_names = ['Type_Param_Spec_List']
    def match(string): return CallBase.match(Type_Name, Type_Param_Spec_List, string)
    match = staticmethod(match)

class Type_Param_Spec(KeywordValueBase):
    """
    <type-param-spec> = [ <keyword> = ] <type-param-value>
    """
    subclass_names = ['Type_Param_Value']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Type_Param_Value, string)
    match = staticmethod(match)

class Type_Param_Spec_List(SequenceBase):
    """
    <type-param-spec-list> = <type-param-spec> [ , <type-param-spec> ]...
    """
    subclass_names = ['Type_Param_Spec']
    use_names = []
    def match(string): return SequenceBase.match(',', Type_Param_Spec, string)
    match = staticmethod(match)


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
    <substring-range> = [ <scalar-int-expr> ] : [ <scalar-int-expr> ]
    <structure-component> = <data-ref>
    """
    subclass_names = ['Object_Name','Array_Section','Array_Element','Structure_Component',
                      'Substring'
                      ]

class Substring(CallBase):
    """
    <substring> = <parent-string> ( <substring-range> )    
    """
    subclass_names = []
    use_names = ['Parent_String','Substring_Range']
    def match(string): return CallBase.match(Parent_String, Substring_Range, string)
    match = staticmethod(match)

class Parent_String(Base):
    """
    <parent-string> = <scalar-variable-name>
                      | <array-element>
                      | <scalar-structure-component>
                      | <scalar-constant>    
    """
    subclass_names = ['Scalar_Variable_Name', 'Array_Element', 'Scalar_Structure_Component', 'Scalar_Constant']

class Scalar_Variable_Name(Base):
    """
    <scalar-variable-name> = <name>
    """
    subclass_names = ['Name']

class Scalar_Structure_Component(Base):
    """
    <scalar-structure-component> = <structure-component>
    """
    subclass_names = ['Structure_Component']

class Scalar_Constant(Base):
    """
    <scalar-constant> = <constant>
    """
    subclass_names = ['Constant']

class Structure_Component(Base):
    """
    <structure-component> = <data-ref>
    """
    subclass_names = ['Data_Ref']

class Array_Section(CallBase):
    """
    <array-section> = <data-ref> [ ( <substring-range> ) ]
    """
    subclass_names = ['Data_Ref']
    use_names = ['Substring_Range']
    def match(string): return CallBase.match(Data_Ref, Substring_Range, string)
    match = staticmethod(match)

class Substring_Range(Base):
    """
    <substring-range> = [ <scalar-int-expr> ] : [ <scalar-int-expr> ]
    """
    subclass_names = []
    use_names = ['Scalar_Int_Expr']
    def match(string):
        line, repmap = string_replace_map(string)
        if ':' not in line: return
        lhs,rhs = line.split(':',1)
        lhs = lhs.rstrip()
        rhs = rhs.lstrip()
        lhs_obj, rhs_obj = None, None
        if lhs:
            for k in Base.findall(lhs):
                lhs = lhs.replace(k,repmap[k])
            lhs_obj = Scalar_Int_Expr(lhs)
        if rhs:
            for k in Base.findall(rhs):
                rhs = rhs.replace(k,repmap[k])
            rhs_obj = Scalar_Int_Expr(rhs)
        return lhs_obj, rhs_obj
    match = staticmethod(match)
    def init(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs
        return
    def tostr(self):
        if self.lhs is None:
            if self.rhs is None: return ':'
            return ': '+ str(self.rhs)
        else:
            if self.rhs is None: return str(self.lhs)+' :'
            return str(self.lhs)+' : '+ str(self.rhs)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.lhs, self.rhs)
        
class Array_Element(Base):
    """
    <array-element> = <data-ref>
    """
    subclass_names = ['Data_Ref']

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

class Scalar_Int_Literal_Constant(Base):
    """
    <scalar-int-literal-constant> = <int-literal-constant>
    """
    subclass_names = ['Int_Literal_Constant']


class Complex_Literal_Constant(Base):
    """
    <complex-literal-constant> = ( <real-part>, <imag-part> )
    """
    subclass_names = []
    use_names = ['Real_Part','Imag_Part']
    def match(string):
        if string[0]+string[-1]!='()': return
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

class Real_Part(Base):
    """
    <real-part> = <signed-int-literal-constant>
                  | <signed-real-literal-constant>
                  | <named-constant>
    """
    subclass_names = ['Signed_Int_Literal_Constant','Signed_Real_Literal_Constant','Named_Constant']

class Imag_Part(Base):
    """
    <imag-part> = <real-part>
    """
    subclass_names = ['Signed_Int_Literal_Constant','Signed_Real_Literal_Constant','Named_Constant']


class Boz_Literal_Constant(Base):
    """
    <boz-literal-constant> = <binary-constant>
                             | <octal-constant>
                             | <hex-constant>
    """
    subclass_names = ['Binary_Constant','Octal_Constant','Hex_Constant']



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
    
class Function_Reference(CallBase):
    """
    <function-reference> = <procedure-designator> ( [ <actual-arg-spec-list> ] )
    """
    subclass_names = []
    use_names = ['Procedure_Designator','Actual_Arg_Spec_List']
    def match(string):
        return CallBase.match(Procedure_Designator, Actual_Arg_Spec_List, string)
    match = staticmethod(match)


class Procedure_Designator(BinaryOpBase):
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

class Part_Ref(CallBase):
    """
    <part-ref> = <part-name> [ ( <section-subscript-list> ) ]
    """
    subclass_names = ['Part_Name']
    use_names = ['Section_Subscript_List']
    def match(string): return CallBase.match(Part_Name, Section_Subscript_List, string)
    match = staticmethod(match)

class Section_Subscript_List(SequenceBase):
    """
    <section-subscript-list> = <section-subscript> [ , <section-subscript> ]...
    """
    subclass_names = ['Section_Subscript']
    use_names = []
    def match(string): return SequenceBase.match(',', Section_Subscript, string)
    match = staticmethod(match)

class Section_Subscript(Base):
    """
    <section-subscript> = <subscript>
                          | <subscript-triplet>
                          | <vector-subscript>
    """
    subclass_names = ['Subscript_Triplet', 'Vector_Subscript', 'Subscript']

class Subscript_Triplet(Base):
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
                for k in Base.findall(stride):
                    stride = stride.replace(k, repmap[k])
                stride_obj = Stride(stride)
        if lhs:
            for k in Base.findall(lhs):
                lhs = lhs.replace(k, repmap[k])
            lhs_obj = Subscript(lhs)
        if rhs:
            for k in Base.findall(rhs):
                rhs = rhs.replace(k, repmap[k])
            rhs_obj = Subscript(rhs)
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

class Stride(Base):
    """
    <stride> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Vector_Subscript(Base):
    """
    <vector-subscript> = <int-expr>
    """
    subclass_names = ['Int_Expr']

class Int_Expr(Base):
    """
    <int-expr> = <expr>
    """
    subclass_names = ['Expr']

class Subscript(Base):
    """
    <subscript> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

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
    subclass_names = ['Part_Ref']
    use_names = []
    def match(string): return SequenceBase.match('%', Part_Ref, string)
    match = staticmethod(match)

class Actual_Arg_Spec_List(SequenceBase):
    """
    <actual-arg-spec-list> = <actual-arg-spec> [ , <actual-arg-spec> ]...
    """
    subclass_names = ['Actual_Arg_Spec']
    use_names = []
    def match(string): return SequenceBase.match(r',', Actual_Arg_Spec, string)
    match = staticmethod(match)

class Actual_Arg_Spec(KeywordValueBase):
    """
    <actual-arg-spec> = [ <keyword> = ] <actual-arg>
    """
    subclass_names = ['Actual_Arg']
    use_names = ['Keyword']
    def match(string): return KeywordValueBase.match(Actual_Arg, string)
    match = staticmethod(match)

class Keyword(Base):
    """
    <keyword> = <name>
    """
    subclass_names = ['Name']

class Type_Name(Name):
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
    
class Actual_Arg(Base):
    """
    <actual-arg> = <expr>
                 | <variable>
                 | <procedure-name>
                 | <proc-component-ref>
                 | <alt-return-spec>
    """
    subclass_names = ['Procedure_Name','Proc_Component_Ref','Alt_Return_Spec', 'Variable', 'Expr']

class Alt_Return_Spec(Base):
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

class Variable(Base):
    """
    <variable> = <designator>
    """
    subclass_names = ['Designator']

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
    
class Parenthesis(BracketBase):
    """
    <parenthesis> = ( <expr> )
    """
    subclass_names = []
    use_names = ['Expr']
    def match(string): return BracketBase.match('()', Expr, string)
    match = staticmethod(match)


##############################################################################
    
ClassType = type(Base)
Base_classes = {}
for clsname in dir():
    cls = eval(clsname)
    if isinstance(cls, ClassType) and issubclass(cls, Base) and not cls.__name__.endswith('Base'):
        Base_classes[cls.__name__] = cls


# Optimize subclass_names tree:
def _rpl_list(clsname):
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

if 0:

    def _deep_subclass_names(clsname, l=[]):
        cls = Base_classes[clsname]
        subclass_names = getattr(cls,'subclass_names',[])
        if clsname not in l:
            l.append(clsname)
        for n in subclass_names:
            if n in l: continue
            l.append(n)
            _deep_subclass_names(n,l)

        return l

    Base_deps = {}
    for cls in Base_classes.values():
        l = []
        _deep_subclass_names(cls.__name__,l)
        Base_deps[cls.__name__] = l

    def _sort_deps(lhs,rhs):
        if lhs==rhs: return 0
        rhs_deps = Base_deps[rhs]
        lhs_deps = Base_deps[lhs]
        if lhs in rhs_deps and rhs in lhs_deps: return 0
        if lhs in rhs_deps: return -1
        if rhs in lhs_deps: return 1
        return 0
        return cmp(len(lhs_deps),len(rhs_deps))

    for clsname,deps in Base_deps.items():
        deps.sort(_sort_deps)
        print clsname,deps

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

if 0:
    for cls in Base_classes.values():
        subclasses = Base.subclasses.get(cls.__name__,[])
        subclasses_names = [c.__name__ for c in subclasses]
        subclass_names = getattr(cls,'subclass_names', [])
        use_names = getattr(cls,'use_names',[])
        for n in subclasses_names:
            if n not in subclass_names:
                print '%s needs to be added to %s subclasses_name list' % (n,cls.__name__)
        for n in subclass_names:
            if n not in subclasses_names:
                print '%s needs to be added to %s subclass_name list' % (n,cls.__name__)
        for n in use_names:
            if not Base_classes.has_key(n):
                print '%s not defined used by %s' % (n, cls.__name__)
#import pprint
#pprint.pprint(Base.subclasses)
