
from expressions import *

class Program(Base):
    """
    <program> = <program-unit>
                  [ <program-unit> ] ...
    """
    subclass_names = []
    use_names = ['Program_Unit']

class Program_Unit(Base):
    """
    <program-unit> = <main-program>
                     | <external-subprogram>
                     | <module>
                     | <block-data>
    """
    subclass_names = ['Main_Program', 'External_Subprogram', 'Module', 'Block_Data']

class External_Subprogram(Base):
    """
    <external-subprogram> = <function-subprogram>
                            | <subroutine-subprogram>
    """
    subclass_names = ['Function_Subprogram', 'Subroutine_Subprogram']

class Function_Subprogram(Base):
    """
    <function-subprogram> = <function-stmt>
                               [ <specification-part> ]
                               [ <execution-part> ]
                               [ <internal-subprogram-part> ]
                            <end-function-stmt>
    """
    subclass_names = []
    use_names = ['Function_Stmt', 'Specification_Part', 'Execution_Part',
                 'Internal_Subprogram_Part', 'End_Function_Stmt']

class Subroutine_Subprogram(BlockBase):
    """
    <subroutine-subprogram> = <subroutine-stmt>
                                 [ <specification-part> ]
                                 [ <execution-part> ]
                                 [ <internal-subprogram-part> ]
                              <end-subroutine-stmt>
    """
    subclass_names = []
    use_names = ['Subroutine_Stmt', 'Specification_Part', 'Execution_Part',
                 'Internal_Subprogram_Part', 'End_Subroutine_Stmt']
    def match(reader):
        item = get_item(reader)
        line = item.line
        content = []
        try:
            start_stmt = Subroutine_Stmt(line)
        except NoMatchError:
            start_stmt = None
        if start_stmt is None:
            put_item(reader, item)
            return
        content.append(start_stmt)
        content.extend(get_content(reader, Specification_Part, End_Subroutine_Stmt))
        if isinstance(content[-1], End_Subroutine_Stmt):
            return content,
        content.extend(get_content(reader, Execution_Part, End_Subroutine_Stmt))
        if isinstance(content[-1], End_Subroutine_Stmt):
            return content,
        content.extend(get_content(reader, Internal_Subprogram_Part, End_Subroutine_Stmt))
        assert isinstance(content[-1], End_Subroutine_Stmt),`content[-1]`
        return content,            
    match = staticmethod(match)




class Prefix(SequenceBase):
    """
    <prefix> = <prefix-spec> [ <prefix-spec> ]..
    """
    subclass_names = ['Prefix_Spec']
    _separator = (' ',re.compile(r'\s+(?=[a-z_])',re.I))
    def match(string): return SequenceBase.match(Prefix._separator, Prefix_Spec, string)
    match = staticmethod(match)

class Prefix_Spec(StringBase):
    """
    <prefix-spec> = <declaration-type-spec>
                    | RECURSIVE
                    | PURE
                    | ELEMENTAL
    """
    subclass_names = ['Declaration_Type_Spec']
    def match(string):
        if len(string)==9:
            upper = string.upper()
            if upper in ['RECURSIVE', 'ELEMENTAL']: return upper,
        elif len(string)==4:
            upper = string.upper()
            if upper=='PURE': return upper,
        return None
    match = staticmethod(match)


class Specification_Part(BlockBase):
    """
    <specification-part> = [ <use-stmt> ]...
                             [ <import-stmt> ]...
                             [ <implicit-part> ]
                             [ <declaration-construct> ]...
    """
    subclass_names = []
    use_names = ['Use_Stmt', 'Import_Stmt', 'Implicit_Part', 'Declaration_Construct']
    def match(reader):
        content = []
        for cls in [Declaration_Construct]:
            item = reader.get_item()
            while item is not None:
                obj = None
                line = item.line
                try:
                    obj = cls(line)
                except NoMatchError:
                    pass
                if obj is None:
                    reader.put_item(item)
                    break
                content.append(obj)
                item = reader.get_item()
        return content,
    match = staticmethod(match)

class Implicit_Part(Base):
    """
    <implicit-part> = [ <implicit-part-stmt> ]...
                        <implicit-stmt>
    """
    subclass_names = []
    use_names = ['Implicit_Part_Stmt', 'Implicit_Stmt']

class Implicit_Part_Stmt(Base):
    """
    <implicit-part-stmt> = <implicit-stmt>
                           | <parameter-stmt>
                           | <format-stmt>
                           | <entry-stmt>
    """
    subclass_names = ['Implicit_Stmt', 'Parameter_Stmt', 'Format_Stmt', 'Entry_Stmt']

class Declaration_Construct(Base):
    """
    <declaration-construct> = <derived-type-def>
                              | <entry-stmt>
                              | <enum-def>
                              | <format-stmt>
                              | <interface-block>
                              | <parameter-stmt>
                              | <procedure-declaration-stmt>
                              | <specification-stmt>
                              | <type-declaration-stmt>
                              | <stmt-function-stmt>
    """
    subclass_names = ['Derived_Type_Def', 'Entry_Stmt', 'Enum_Def', 'Format_Stmt',
                      'Interface_Block', 'Parameter_Stmt', 'Procedure_Declaration_Stmt',
                      'Specification_Stmt', 'Type_Declaration_Stmt', 'Stmt_Function_Stmt']

class Execution_Part(Base):
    """
    <execution-part> = <executable-construct>
                       | [ <execution-part-construct> ]...

    <execution-part> shall not contain <end-function-stmt>, <end-program-stmt>, <end-subroutine-stmt>
    """
    subclass_names = []
    use_names = ['Executable_Construct', 'Execution_Part_Construct']

class Execution_Part_Construct(Base):
    """
    <execution-part-construct> = <executable-construct>
                                 | <format-stmt>
                                 | <entry-stmt>
                                 | <data-stmt>
    """
    subclass_names = ['Executable_Construct', 'Format_Stmt', 'Entry_Stmt', 'Data_Stmt']

class Internal_Subprogram_Part(Base):
    """
    <internal-subprogram-part> = <contains-stmt>
                                   <internal-subprogram>
                                   [ <internal-subprogram> ]...
    """
    subclass_names = []
    use_names = ['Contains_Stmt', 'Internal_Subprogram']

class Internal_Subprogram(Base):
    """
    <internal-subprogram> = <function-subprogram>
                            | <subroutine-subprogram>
    """
    subclass_names = ['Function_Subprogram', 'Subroutine_Subprogram']

class Specification_Stmt(Base):
    """
    <specification-stmt> = <access-stmt>
                           | <allocatable-stmt>
                           | <asynchronous-stmt>
                           | <bind-stmt>
                           | <common-stmt>
                           | <data-stmt>
                           | <dimension-stmt>
                           | <equivalence-stmt>
                           | <external-stmt>
                           | <intent-stmt>
                           | <intrinsic-stmt>
                           | <namelist-stmt>
                           | <optional-stmt>
                           | <pointer-stmt>
                           | <protected-stmt>
                           | <save-stmt>
                           | <target-stmt>
                           | <volatile-stmt>
                           | <value-stmt>
    """
    subclass_names = ['Access_Stmt', 'Allocatable_Stmt', 'Asynchronous_Stmt','Bind_Stmt',
                      'Common_Stmt', 'Data_Stmt', 'Dimension_Stmt', 'Equivalence_Stmt',
                      'External_Stmt', 'Intent_Stmt', 'Intrinsic_Stmt', 'Namelist_Stmt',
                      'Optional_Stmt','Pointer_Stmt','Protected_Stmt','Save_Stmt',
                      'Target_Stmt','Volatile_Stmt', 'Value_Stmt']

class Executable_Construct(Base):
    """
    <executable-construct> = <action-stmt>
                             | <associate-stmt>
                             | <case-construct>
                             | <do-construct>
                             | <forall-construct>
                             | <if-construct>
                             | <select-type-construct>
                             | <where-construct>
    """
    subclass_names = ['Action_Stmt', 'Associate_Stmt', 'Case_Construct', 'Do_Construct',
                      'Forall_Construct', 'If_Construct', 'Select_Type_Construct', 'Where_Construct']

class Action_Stmt(Base):
    """
    <action-stmt> = <allocate-stmt>
                    | <assignment-stmt>
                    | <backspace-stmt>
                    | <call-stmt>
                    | <close-stmt>
                    | <continue-stmt>
                    | <cycle-stmt>
                    | <deallocate-stmt>
                    | <endfile-stmt>
                    | <end-function-stmt>
                    | <end-program-stmt>
                    | <end-subroutine-stmt>
                    | <exit-stmt>
                    | <flush-stmt>
                    | <forall-stmt>
                    | <goto-stmt>
                    | <if-stmt>
                    | <inquire-stmt>
                    | <nullify-stmt>
                    | <open-stmt>
                    | <pointer-assignment-stmt>
                    | <print-stmt>
                    | <read-stmt>
                    | <return-stmt>
                    | <rewind-stmt>
                    | <stop-stmt>
                    | <wait-stmt>
                    | <where-stmt>
                    | <write-stmt>
                    | <arithmetic-if-stmt>
                    | <computed-goto-stmt>
    """
    subclass_names = ['Allocate_Stmt', 'Assignment_Stmt', 'Backspace_Stmt', 'Call_Stmt',
                      'Close_Stmt', 'Continue_Stmt', 'Cycle_Stmt', 'Deallocate_Stmt',
                      'Endfile_Stmt', 'End_Function_Stmt', 'End_Subroutine_Stmt', 'Exit_Stmt',
                      'Flush_Stmt', 'Forall_Stmt', 'Goto_Stmt', 'If_Stmt', 'Inquire_Stmt',
                      'Nullify_Stmt', 'Open_Stmt', 'Pointer_Assignment_Stmt', 'Print_Stmt',
                      'Read_Stmt', 'Return_Stmt', 'Rewind_Stmt', 'Stop_Stmt', 'Wait_Stmt',
                      'Where_Stmt', 'Write_Stmt', 'Arithmetic_If_Stmt', 'Computed_Goto_Stmt']

class Int_Constant(Base):
    subclass_names = ['Constant']

class Scalar_Int_Constant(Base):
    subclass_names = ['Int_Constant']

class Derived_Type_Def(Base):
    """
    <derived-type-def> = <derived-type-stmt>
                           [ <type-param-def-stmt> ]...
                           [ <private-or-sequence> ]...
                           [ <component-part> ]
                           [ <type-bound-procedure-part> ]
                           <end-type-stmt>
    """
    subclass_names = []
    use_names = ['Derived_Type_Stmt', 'Type_Param_Def_Stmt', 'Private_Or_Sequence',
                 'Component_Part', 'Type_Bound_Procedure_Part', 'End_Type_Stmt']

class Derived_Type_Stmt(Base):
    """
    <derived-type-stmt> = TYPE [ [ , <type-attr-spec-list> ] :: ] <type-name> [ ( <type-param-name-list> ) ]
    """
    subclass_names = []
    use_names = ['Type_Attr_Spec_List', 'Type_Name', 'Type_Param_Name_List']

class Type_Attr_Spec(Base):
    """
    <type-attr-spec> = <access-spec>
                       | EXTENDS ( <parent-type-name> )
                       | ABSTRACT
                       | BIND (C)
    """
    subclass_names = ['Access_Spec']
    use_names = ['Parent_Type_Name']

class Private_Or_Sequence(Base):
    """
    <private-or-sequence> = <private-components-stmt>
                            | <sequence-stmt>
    """
    subclass_names = ['Private_Components_Stmt', 'Sequence_Stmt']

class End_Type_Stmt(Base):
    """
    <end-type-stmt> = END TYPE [ <type-name> ]
    """
    subclass_names = []
    use_names = ['Type_Name']
    
class Sequence_Stmt(Base):
    """
    <sequence-stmt> = SEQUENCE
    """
    subclass_names = []
    def match(string):
        if len(string) != 8: return
        start = string.upper()
        if start=='SEQUENCE': return start,
        return
    match = staticmethod(match)

class Type_Param_Def_Stmt(Base):
    """
    <type-param-def-stmt> = INTEGER [ <kind-selector> ] , <type-param-attr-spec> :: <type-param-decl-list>
    """
    subclass_names = []
    use_names = ['Kind_Selector', 'Type_Param_Attr_Spec', 'Type_Param_Decl_List']

class Type_Param_Decl(Base):
    """
    <type-param-decl> = <type-param-name> [ = <scalar-int-initialization-expr> ]
    """
    subclass_names = []
    use_names = ['Type_Param_Name', 'Scalar_Int_Initialization_Expr']

class Type_Param_Attr_Spec(Base):
    """
    <type-param-attr-spec> = KIND
                             | LEN
    """
    subclass_names = []

class Component_Part(Base):
    """
    <component-part> = [ <component-def-stmt> ]...
    """
    subclass_names = []
    use_names = ['Component_Def_Stmt']

class Component_Def_Stmt(Base):
    """
    <component-def-stmt> = <data-component-def-stmt>
                           | <proc-component-def-stmt>
    """
    subclass_names = ['Data_Component_Def_Stmt', 'Proc_Component_Def_Stmt']

class Data_Component_Def_Stmt(Base):
    """
    <data-component-def-stmt> = <declaration-type-spec> [ [ , <component-attr-spec-list> ] :: ] <component-decl-list>
    """
    subclass_names = []
    use_names = ['Declaration_Type_Spec', 'Component_Attr_Spec_List', 'Component_Decl_List']

class Component_Attr_Spec(Base):
    """
    <component-attr-spec> = POINTER
                            | DIMENSION ( <component-array-spec> )
                            | ALLOCATABLE
                            | <access-spec>
    """
    subclass_names = ['Access_Spec']
    use_names = ['Component_Array_Spec']

class Component_Decl(Base):
    """
    <component-decl> = <component-name> [ ( <component-array-spec> ) ] [ * <char-length> ] [ <component-initialization> ]
    """
    subclass_names = []
    use_names = ['Component_Name', 'Component_Array_Spec', 'Char_Length', 'Component_Initialization']

class Component_Array_Spec(Base):
    """
    <component-array-spec> = <explicit-shape-spec-list>
                             | <deferred-shape-spec-list>
    """
    subclass_names = ['Explicit_Shape_Spec_List', 'Deferred_Shape_Spec_List']

class Component_Initialization(Base):
    """
    <component-initialization> =  = <initialization-expr>
                                 | => <null-init>
    """
    subclass_names = []
    use_names = ['Initialization-expr', 'Null_Init']

class Proc_Component_Def_Stmt(Base):
    """
    <proc-component-def-stmt> = PROCEDURE ( [ <proc-interface> ] ) , <proc-component-attr-spec-list> :: <proc-decl-list>
    """
    subclass_names = []
    use_names = ['Proc_Interface', 'Proc_Component_Attr_Spec_List', 'Proc_Decl_List']

class Proc_Component_Attr_Spec(Base):
    """
    <proc-component-attr-spec> = POINTER
                                 | PASS [ ( <arg-name> ) ]
                                 | NOPASS
                                 | <access-spec>
    """
    subclass_names = []
    use_names = ['Arg_Name', 'Access_Spec']

class Private_Components_Stmt(Base):
    """
    <private-components-stmt> = PRIVATE
    """
    subclass_names = []

class Type_Bound_Procedure_Part(Base):
    """
    <type-bound-procedure-part> = <contains-stmt>
                                      [ <binding-private-stmt> ]
                                      <proc-binding-stmt>
                                      [ <proc-binding-stmt> ]...
    """
    subclass_names = []
    use_names = ['Contains_Stmt', 'Binding_Private_Stmt', 'Proc_Binding_Stmt']

class Binding_Private_Stmt(Base):
    """
    <binding-private-stmt> = PRIVATE
    """
    subclass_names = []

class Proc_Binding_Stmt(Base):
    """
    <proc-binding-stmt> = <specific-binding>
                          | <generic-binding>
                          | <final-binding>
    """
    subclass_names = ['Specific_Binding', 'Generic_Binding', 'Final_Binding']

class Specific_Binding(Base):
    """
    <specific-binding> = PROCEDURE [ ( <interface-name> ) ] [ [ , <binding-attr-list> ] :: ] <binding-name> [ => <procedure-name> ]
    """
    subclass_names = []
    use_names = ['Interface_Name', 'Binding_Attr_List', 'Binding_Name', 'Procedure_Name']

class Generic_Binding(Base):
    """
    <generic-binding> = GENERIC [ , <access-spec> ] :: <generic-spec> => <binding-name-list>
    """
    subclass_names = []
    use_names = ['Access_Spec', 'Generic_Spec', 'Binding_Name_List']

class Binding_Attr(Base):
    """
    <binding-attr> = PASS [ ( <arg-name> ) ]
                     | NOPASS
                     | NON_OVERRIDABLE
                     | <access-spec>
    """
    subclass_names = []
    use_names = ['Arg_Name', 'Access_Spec']

class Final_Binding(Base):
    """
    <final-binding> = FINAL [ :: ] <final-subroutine-name-list>
    """
    subclass_names = []
    use_names = ['Final_Subroutine_Name_List']

#<derived-type-spec>
#<type-param-spec>
#<structure-constructor>
#<component-spec>
#<component-data-source>

class Enum_Def(Base):
    """
    <enum-def> = <enum-def-stmt>
                     <enumerator-def-stmt>
                     [ <enumerator-def-stmt> ]...
                     <end-enum-stmt>
    """
    subclass_names = []
    use_names = ['Enum_Def_Stmt', 'Enumerator_Def_Stmt', 'End_Enum_Stmt']

class Enum_Def_Stmt(Base):
    """
    <enum-def-stmt> = ENUM, BIND(C)
    """
    subclass_names = []

class Enumerator_Def_Stmt(Base):
    """
    <enumerator-def-stmt> = ENUMERATOR [ :: ] <enumerator-list>
    """
    subclass_names = []
    use_names = ['Enumerator_List']

class Enumerator(Base):
    """
    <enumerator> = <named-constant> [ = <scalar-int-initialization-expr> ]
    """
    subclass_names = []
    use_names = ['Named_Constant', 'Scalar_Int_Initialization_Expr']

class End_Enumerator_Stmt(Base):
    """
    <end-enumerator-stmt> = END ENUM
    """
    subclass_names = []

#<array-constructor>
#<ac-spec>
#<ac-value>
#<ac-implied-do>
#<ac-implied-do-control>
#<ac-do-variable>

class Type_Declaration_Stmt(Base):
    """
    <type-declaration-stmt> = <declaration-type-spec> [ [ , <attr-spec> ]... :: ] <entity-decl-list>
    """
    subclass_names = []
    use_names = ['Declaration_Type_Spec', 'Attr_Spec_List', 'Entity_Decl_List']

    def match(string):
        line, repmap = string_replace_map(string)
        i = line.find(',')
        if i==-1:
            i = line.find('::')
        if i==-1:
            m = re.search(r'\s[a-z_]',line,re.I)
            if m is None: return
            i = m.start()
        type_spec = Declaration_Type_Spec(repmap(line[:i]))
        if type_spec is None: return
        line = line[i:].lstrip()
        if line.startswith(','):
            i = line.find('::')
            if i==-1: return
            attr_specs = Attr_Spec_List(repmap(line[1:i].strip()))
            if attr_specs is None: return
            line = line[i:]
        else:
            attr_specs = None
        if line.startswith('::'):
            line = line[2:].lstrip()
        entity_decls = Entity_Decl_List(repmap(line))
        if entity_decls is None: return
        return type_spec, attr_specs, entity_decls
    match = staticmethod(match)
    def init(self, *args):
        self.type_spec, self.attr_specs, self.entity_decls = args
        return
    def tostr(self):
        if self.attr_specs is None:
            return '%s :: %s' % (self.type_spec, self.entity_decls)
        else:
            return '%s, %s :: %s' % (self.type_spec, self.attr_specs, self.entity_decls)
    def torepr(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self.type_spec, self.attr_specs, self.entity_decls)

class Declaration_Type_Spec(Base):
    """
    <declaration-type-spec> = <intrinsic-type-spec>
                              | TYPE ( <derived-type-spec> )
                              | CLASS ( <derived-type-spec> )
                              | CLASS ( * )
    """
    subclass_names = ['Intrinsic_Type_Spec']
    use_names = ['Derived_Type_Spec']

    def match(string):
        if string[-1] != ')': return
        start = string[:4].upper()
        if start == 'TYPE':
            line = string[4:].lstrip()
            if not line.startswith('('): return
            return 'TYPE',Derived_Type_Spec(line[1:-1].strip())
        start = string[:5].upper()
        if start == 'CLASS':
            line = string[5:].lstrip()
            if not line.startswith('('): return
            line = line[1:-1].strip()
            if line=='*': return 'CLASS','*'
            return 'CLASS', Derived_Type_Spec(line)
        return
    match = staticmethod(match)
    init = Base.init_list
    def tostr(self): return '%s(%s)' % tuple(map(str,self.items))
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.items[0], self.items[1])

class Dimension_Attr_Spec(CallBase):
    """
    <dimension-attr-spec> = DIMENSION ( <array-spec> )
    """
    subclass_names = []
    use_names = ['Array_Spec']
    def match(string): return CallBase.match(pattern.abs_dimension, Array_Spec, string)
    match = staticmethod(match)
    def init(self, lhs, rhs):
        self.lhs = lhs.upper()
        self.rhs = rhs
        return

class Intent_Attr_Spec(CallBase):
    """
    <intent-attr-spec> = INTENT ( <intent-spec> )
    """
    subclass_names = []
    use_names = ['Intent_Spec']
    def match(string): return CallBase.match(pattern.abs_intent, Intent_Spec, string)
    match = staticmethod(match)
    def init(self, lhs, rhs):
        self.lhs = lhs.upper()
        self.rhs = rhs
        return

class Attr_Spec(STRINGBase):
    """
    <attr-spec> = <access-spec>
                  | ALLOCATABLE
                  | ASYNCHRONOUS
                  | DIMENSION ( <array-spec> )
                  | EXTERNAL
                  | INTENT ( <intent-spec> )
                  | INTRINSIC
                  | <language-binding-spec>
                  | OPTIONAL
                  | PARAMETER
                  | POINTER
                  | PROTECTED
                  | SAVE
                  | TARGET
                  | VALUE
                  | VOLATILE
    """
    subclass_names = ['Access_Spec', 'Language_Binding_Spec',
                      'Dimension_Attr_Spec', 'Intent_Attr_Spec']
    use_names = []
    def match(string): return STRINGBase.match(pattern.abs_attr_spec, string)
    match = staticmethod(match)

class Entity_Decl(Base):
    """
    <entity-decl> = <object-name> [ ( <array-spec> ) ] [ * <char-length> ] [ <initialization> ]
                    | <function-name> [ * <char-length> ]
    """
    subclass_names = []
    use_names = ['Object_Name', 'Array_Spec', 'Char_Length', 'Initialization', 'Function_Name']
    def match(string):
        m = pattern.name.match(string)
        if m is None: return
        name = Name(m.group())
        newline = string[m.end():].lstrip()
        if not newline: return name, None, None, None
        array_spec = None
        char_length = None
        init = None
        if newline.startswith('('):
            line, repmap = string_replace_map(newline)
            i = line.find(')')
            if i==-1: return
            array_spec = Array_Spec(repmap(line[1:i].strip()))
            newline = repmap(line[i+1:].lstrip())
        if newline.startswith('*'):
            line, repmap = string_replace_map(newline)
            i = line.find('=')
            if i!=-1:
                char_length = repmap(line[1:i].strip())
                newline = repmap(newline[i:].lstrip())
            else:
                char_length = repmap(newline[1:].strip())
                newline = ''
            char_length = Char_Length(char_length)
        if newline.startswith('='):
            init = Initialization(newline)
        else:
            assert newline=='',`newline`
        return name, array_spec, char_length, init
    match = staticmethod(match)
    def init(self, *args):
        self.name, self.array_spec, self.char_length, self.init = args
        return
    def tostr(self):
        s = str(self.name)
        if self.array_spec is not None:
            s += '(' + str(self.array_spec) + ')'
        if self.char_length is not None:
            s += '*' + str(self.char_length)
        if self.init is not None:
            s += ' ' + str(self.init)
        return s
    def torepr(self):
        return '%s(%r, %r, %r, %r)' \
               % (self.__class__.__name__, self.name, self.array_spec, self.char_length, self.init)
    
class Object_Name(Base):
    """
    <object-name> = <name>
    """
    subclass_names = ['Name']

class Initialization(Base):
    """
    <initialization> =  = <initialization-expr>
                       | => <null-init> 
    """
    subclass_names = []
    use_names = ['Initialization_Expr', 'Null_Init']
    def match(string):
        if string.startswith('=>'):
            return '=>', Null_Init(string[2:].lstrip())
        if string.startswith('='):
            return '=', Initialization_Expr(string[2:].lstrip())
        return
    match = staticmethod(match)
    def init(self, op, rhs):
        self.op = op
        self.rhs = rhs
        return
    def tostr(self): return '%s %s' % (self.op, self.rhs)
    def torepr(self): return '%s(%r, %r)' % (self.__class__.__name__, self.op, self.rhs)

class Null_Init(Base):
    """
    <null-init> = <function-reference>

    <function-reference> shall be a reference to the NULL intrinsic function with no arguments.
    """
    subclass_names = ['Function_Reference']

class Access_Spec(STRINGBase):
    """
    <access-spec> = PUBLIC
                    | PRIVATE
    """
    subclass_names = []
    def match(string): return STRINGBase.match(pattern.abs_access_spec, string)
    match = staticmethod(match)

class Language_Binding_Spec(Base):
    """
    <language-binding-spec> = BIND ( C [ , NAME = <scalar-char-initialization-expr> ] )
    """
    subclass_names = []
    use_names = ['Scalar_Char_Initialization_Expr']
    def match(string):
        start = string[:4].upper()
        if start != 'BIND': return
        line = string[4:].lstrip()
        if not line or line[0]+line[-1]!='()': return
        line = line[1:-1].strip()
        if not line: return
        start = line[0].upper()
        if start!='C': return
        line = line[1:].lstrip()
        if not line: return None,
        if not line.startswith(','): return
        line = line[1:].lstrip()
        start = line[:4].upper()
        if start!='NAME': return
        line=line[4:].lstrip()
        if not line.startswith('='): return
        return Scalar_Char_Initialization_Expr(line[1:].lstrip()),
    match = staticmethod(match)
    def init(self, name):
        self.name = name
    def tostr(self):
        if self.name is None: return 'BIND(C)'
        return 'BIND(C, NAME = %s)' % (self.name)
    def torepr(self):
        return '%s(%r)' % (self.__class__.__name__, self.name)

class Array_Spec(Base):
    """
    <array-spec> = <explicit-shape-spec-list>
                   | <assumed-shape-spec-list>
                   | <deferred-shape-spec-list>
                   | <assumed-size-spec>
    """
    subclass_names = ['Explicit_Shape_Spec_List', 'Assumed_Shape_Spec_List',
                      'Deferred_Shape_Spec_List', 'Assumed_Size_Spec']

class Explicit_Shape_Spec(SeparatorBase):
    """
    <explicit-shape-spec> = [ <lower-bound> : ] <upper-bound>
    """
    subclass_names = ['Upper_Bound']
    use_names = ['Lower_Bound']
    def match(string):
        line, repmap = string_replace_map(string)
        if ':' not in line: return
        lower,upper = line.split(':',1)
        for k in Base.findall(lower):
            lower = lower.replace(k, repmap[k])
        for k in Base.findall(upper):
            upper = upper.replace(k, repmap[k])
        return Lower_Bound(lower), Upper_Bound(upper)
    match = staticmethod(match)
    
class Lower_Bound(Base):
    """
    <lower-bound> = <specification-expr>
    """
    subclass_names = ['Specification_Expr']

class Upper_Bound(Base):
    """
    <upper-bound> = <specification-expr>
    """
    subclass_names = ['Specification_Expr']

class Assumed_Shape_Spec(Base):
    """
    <assumed-shape-spec> = [ <lower-bound> ] :
    """
    subclass_names = []
    use_names = ['Lower_Bound']

class Deferred_Shape_Spec(Base):
    """
    <deferred_shape_spec> = :
    """
    subclass_names = []

class Assumed_Size_Spec(Base):
    """
    <assumed-size-spec> = [ <explicit-shape-spec-list> , ] [ <lower-bound> : ] *
    """
    subclass_names = []
    use_names = ['Explicit_Shape_Spec_List', 'Lower_Bound']

class Intent_Spec(STRINGBase):
    """
    <intent-spec> = IN
                    | OUT
                    | INOUT
    """
    subclass_names = []
    def match(string): return STRINGBase.match(pattern.abs_intent_spec, string)
    match = staticmethod(match)

class Access_Stmt(Base):
    """
    <access-stmt> = <access-spec> [ [ :: ] <access-id-list> ]
    """
    subclass_names = []
    use_names = ['Access_Spec', 'Access_Id_List']

class Access_Id(Base):
    """
    <access-id> = <use-name>
                  | <generic-spec>
    """
    subclass_names = ['Use_Name', 'Generic_Spec']

class Allocatable_Stmt(Base):
    """
    <allocateble-stmt> = ALLOCATABLE [ :: ] <object-name> [ ( <deferred-shape-spec-list> ) ] [ , <object-name> [ ( <deferred-shape-spec-list> ) ] ]...
    """
    subclass_names = []
    use_names = ['Object_Name', 'Deferred_Shape_Spec_List']

class Asynchronous_Stmt(Base):
    """
    <asynchronous-stmt> = ASYNCHRONOUS [ :: ] <object-name-list>
    """
    subclass_names = []
    use_names = ['Object_Name_List']

class Bind_Stmt(Base):
    """
    <bind-stmt> = <language-binding-spec> [ :: ] <bind-entity-list>
    """
    use_names = ['Language_Binding_Spec', 'Bind_Entity_List']

class Bind_Entity(Base):
    """
    <bind-entity> = <entity-name>
                    | / <common-block-name> /
    """
    subclass_names = ['Entity_Name']
    use_names = ['Common_Block_Name']

class Data_Stmt(Base):
    """
    <data-stmt> = DATA <data-stmt-set> [ [ , ] <data-stmt-set> ]...
    """
    subclass_names = []
    use_names = ['Data_Stmt_Set']

class Data_Stmt_Set(Base):
    """
    <data-stmt-set> = <data-stmt-object-list> / <data-stmt-value-list> /
    """
    use_names = ['Data_Stmt_Object_List', 'Data_Stmt_Value_List']

class Data_Stmt_Object(Base):
    """
    <data-stmt-object> = <variable>
                         | <data-implied-do>
    """
    subclass_names = ['Variable', 'Data_Implied_Do']

class Data_Implied_Do(Base):
    """
    <data-implied-do> = ( <data-i-do-object-list> , <data-i-do-variable> = <scalar-int-expr > , <scalar-int-expr> [ , <scalar-int-expr> ] )
    """
    subclass_names = []
    use_names = ['Data_I_Do_Object_List', 'Data_I_Do_Variable', 'Scalar_Int_Expr']

class Data_I_Do_Object(Base):
    """
    <data-i-do-object> = <array-element>
                         | <scalar-structure-component>
                         | <data-implied-do>
    """
    subclass_names = ['Array_Element', 'Scalar_Structure_Component', 'Data_Implied_Do']

class Data_I_Do_Variable(Base):
    """
    <data-i-do-variable> = <scalar-int-variable>
    """
    subclass_names = ['Scalar_Int_Variable']

class Data_Stmt_Value(Base):
    """
    <data-stmt-value> = [ <data-stmt-repeat> * ] <data-stmt-constant>
    """
    subclass_names = []
    use_names = ['Data_Stmt_Repeat', 'Data_Stmt_Constant']

class Data_Stmt_Repeat(Base):
    """
    <data-stmt-repeat> = <scalar-int-constant>
                         | <scalar-int-constant-subobject>
    """
    subclass_names = ['Scalar_Int_Constant', 'Scalar_Int_Constant_Subobject']

class Data_Stmt_Constant(Base):
    """
    <data-stmt-constant> = <scalar-constant>
                           | <scalar-constant-subobject>
                           | <signed-int-literal-constant>
                           | <signed-real-literal-constant>
                           | <null-init>
                           | <structure-constructor>
    """
    subclass_names = ['Scalar_Constant', 'Scalar_Constant_Subobject',
                      'Signed_Int_Literal_Constant', 'Signed_Real_Literal_Constant',
                      'Null_Init', 'Structure_Constructor']

class Int_Constant_Subobject(Base):
    """
    <int-constant-subobject> = <constant-subobject>
    """
    subclass_names = ['Constant_Subobject']

class Scalar_Int_Constant_Subobject(Base):
    subclass_names = ['Int_Constant_Subobject']

class Scalar_Constant_Subobject(Base):
    subclass_names = ['Constant_Subobject']

class Constant_Subobject(Base):
    """
    <constant-subobject> = <designator>
    """
    subclass_names = ['Designator']

class Dimension_Stmt(Base):
    """
    <dimension-stmt> = DIMENSION [ :: ] <array-name> ( <array-spec> ) [ , <array-name> ( <array-spec> ) ]...
    """
    subclass_names = []
    use_names = ['Array_Name', 'Array_Spec']

class Intent_Stmt(Base):
    """
    <intent-stmt> = INTENT ( <intent-spec> ) [ :: ] <dummy-arg-name-list>
    """
    subclass_names = []
    use_names = ['Intent_Spec', 'Dummy_Arg_Name_List']

class Optional_Stmt(Base):
    """
    <optional-stmt> = OPTIONAL [ :: ] <dummy-arg-name-list>
    """
    subclass_names = []
    use_names = ['Dummy_Arg_Name_List']

class Parameter_Stmt(Base):
    """
    <parameter-stmt> = PARAMETER ( <named-constant-def-list> )
    """
    subclass_names = []
    use_names = ['Named_Constant_Def_List']

class Named_Constant_Def(Base):
    """
    <named-constant-def> = <named-constant> = <initialization-expr>
    """
    subclass_names = []
    use_names = ['Named_Constant', 'Initialization_Expr']

class Pointer_Stmt(Base):
    """
    <pointer-stmt> = POINTER [ :: ] <pointer-decl-list>
    """
    subclass_names = []
    use_names = ['Pointer_Decl_List']

class Pointer_Decl(Base):
    """
    <pointer-decl> = <object-name> [ ( <deferred-shape-spec-list> ) ]
                     | <proc-entity-name>
    """
    use_names = ['Object_Name', 'Deferred_Shape_Spec_List']
    subclass_names = ['Proc_Entity_Name']

class Protected_Stmt(Base):
    """
    <protected-stmt> = PROTECTED [ :: ] <entity-name-list>
    """
    subclass_names = []
    use_names = ['Entity_Name_List']

class Save_Stmt(Base):
    """
    <save-stmt> = SAVE [ [ :: ] <saved-entity-list> ]
    """
    subclass_names = []
    use_names = ['Saved_Entity_List']

class Saved_Entity(Base):
    """
    <saved-entity> = <object-name>
                     | <proc-pointer-name>
                     | / <common-block-name> /
    """
    subclass_names = ['Object_Name', 'Proc_Pointer_Name']
    use_names = ['Common_Block_Name']

class Proc_Pointer_Name(Base):
    """
    <proc-pointer-name> = <name>
    """
    subclass_names = ['Name']

class Target_Stmt(Base):
    """
    <target-stmt> = TARGET [ :: ] <object-name> [ ( <array-spec> ) ] [ , <object-name> [ ( <array-spec> ) ]]
    """
    subclass_names = []
    use_names = ['Object_Name', 'Array_Spec']

class Value_Stmt(Base):
    """
    <value-stmt> = VALUE [ :: ] <dummy-arg-name-list>
    """
    subclass_names = []
    use_names = ['Dummy_Arg_Name_List']

class Volatile_Stmt(Base):
    """
    <volatile-stmt> = VOLATILE [ :: ] <object-name-list>
    """
    subclass_names = []
    use_names = ['Object_Name_List']

class Implicit_Stmt(Base):
    """
    <implicit-stmt> = IMPLICIT <implicit-spec-list>
                      | IMPLICIT NONE
    """
    subclass_names = []
    use_names = ['Implicit_Spec_List']

class Implicit_Spec(Base):
    """
    <implicit-spec> = <declaration-type-spec> ( <letter-spec-list> )
    """
    subclass_names = []
    use_names = ['Declaration_Type_Spec', 'Letter_Spec_List']

class Letter_Spec(Base):
    """
    <letter-spec> = <letter> [ - <letter> ]
    """
    subclass_names = []

class Namelist_Stmt(Base):
    """
    <namelist-stmt> = NAMELIST / <namelist-group-name> / <namelist-group-object-list> [ [ , ] / <namelist-group-name> / <namelist-group-object-list> ]
    """
    subclass_names = []
    use_names = ['Namelist_Group_Name', 'Namelist_Group_Object_List']

class Namelist_Group_Object(Base):
    """
    <namelist-group-object> = <variable-name>
    """
    subclass_names = ['Variable_Name']

class Equivalence_Stmt(Base):
    """
    <equivalence-stmt> = EQUIVALENCE <equivalence-set-list>
    """
    subclass_names = []
    use_names = ['Equivalence_Set_List']

class Equivalence_Set(Base):
    """
    <equivalence-set> = ( <equivalence-object> , <equivalence-object-list> )
    """
    subclass_names = []
    use_names = ['Equivalence_Object', 'Equivalence_Object_List']

class Equivalence_Object(Base):
    """
    <equivalence-object> = <variable-name>
                           | <array-element>
                           | <substring>
    """
    subclass_names = ['Variable_Name', 'Array_Element', 'Substring']

class Common_Stmt(Base):
    """
    <common-stmt> = COMMON [ / [ <common-block-name> ] / ] <common-block-object-list> [ [ , ] / [ <common-block-name> ] / <common-block-object-list> ]...
    """
    subclass_names = []
    use_names = ['Common_Block_Name', 'Common_Block_Object_List']

class Common_Block_Object(Base):
    """
    <common-block-object> = <variable-name> [ ( <explicit-shape-spec-list> ) ]
                            | <proc-pointer-name>
    """
    subclass_names = ['Proc_Pointer_Name']
    use_names = ['Variable_Name', 'Explicit_Shape_Spec_List']

class Variable_Name(Base):
    """
    <variable-name> = <name>
    """
    subclass_names = ['Name']

class Int_Variable(Base):
    """
    <int-variable> = <variable>
    """
    subclass_names = ['Variable']

class Scalar_Int_Variable(Base):
    """
    <scalar-int-variable> = <int-variable>
    """
    subclass_names = ['Int_Variable']

class Char_Variable(Base):
    """
    <char-variable> = <variable>
    """
    subclass_names = ['Variable']

class Scalar_Default_Char_Variable(Base):
    """
    <scalar-default-char-variable> = <char-variable>
    """
    subclass_names = ['Char_Variable']

class Allocate_Stmt(Base):
    """
    <allocate-stmt> = ALLOCATE ( [ <type-spec> :: ] <allocation-list> [ , <alloc-opt-list> ] )
    """
    subclass_names = []
    use_names = ['Type_Spec', 'Allocation_List', 'Alloc_Opt_List']
    
class Alloc_Opt(Base):
    """
    <alloc-opt> = STAT = <stat-variable>
                  | ERRMSG = <errmsg-variable>
                  | SOURCE = <source-expr>
    """
    subclass_names = []
    use_names = ['Stat_Variable', 'Errmsg_Variable', 'Source_Expr',
                 ]

class Stat_Variable(Base):
    """
    <stat-variable> = <scalar-int-variable>
    """
    subclass_names = ['Scalar_Int_Variable']

class Errmsg_Variable(Base):
    """
    <errmsg-variable> = <scalar-default-char-variable>
    """
    subclass_names = ['Scalar_Default_Char_Variable']

class Source_Expr(Base):
    """
    <source-expr> = <expr>
    """
    subclass_names = ['Expr']

class Allocation(Base):
    """
    <allocation> = <allocate-object> [ <allocate-shape-spec-list> ]
                 | <variable-name>
    """
    subclass_names = ['Variable_Name']
    use_names = ['Allocate_Object', 'Allocate_Shape_Spec_List']

class Allocate_Shape_Spec(Base):
    """
    <allocate-shape-spec> = [ <lower-bound-expr> : ] <upper-bound-expr>
    """
    subclass_names = []
    use_names = ['Lower_Bound_Expr', 'Upper_Bound_Expr']

class Lower_Bound_Expr(Base):
    """
    <lower-bound-expr> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Upper_Bound_Expr(Base):
    """
    <upper-bound-expr> = <scalar-int-expr>
    """
    subclass_names = ['Scalar_Int_Expr']

class Nullify_Stmt(Base):
    """
    <nullify-stmt> = NULLIFY ( <pointer-object-list> )
    """
    subclass_names = []
    use_names = ['Pointer_Object_List']

class Pointer_Object(Base):
    """
    <pointer-object> = <variable-name>
                       | <structure-component>
                       | <proc-pointer-name>
    """
    subclass_names = ['Variable_Name', 'Structure_Component', 'Proc_Pointer_Name']

class Deallocate_Stmt(Base):
    """
    <deallocate-stmt> = DEALLOCATE ( <allocate-object-list> [ , <dealloc-opt-list> ] )
    """
    subclass_names = []
    use_names = ['Allocate_Object_List', 'Dealloc_Opt_List']

class Dealloc_Opt(Base):
    """
    <dealloc-opt> = STAT = <stat-variable>
                    | ERRMSG = <errmsg-variable>
    """
    subclass_names = []
    use_names = ['Stat_Variable', 'Errmsg_Variable']


class Logical_Expr(Base):
    """
    <logical-expr> = <expr>
    """
    subclass_names = ['Expr']

class Char_Expr(Base):
    """
    <char-expr> = <expr>
    """
    subclass_names = ['Expr']

class Default_Char_Expr(Base):
    """
    <default-char-expr> = <expr>
    """
    subclass_names = ['Expr']

class Int_Expr(Base):
    """
    <int-expr> = <expr>
    """
    subclass_names = ['Expr']

class Numeric_Expr(Base):
    """
    <numeric-expr> = <expr>
    """
    subclass_names = ['Expr']

class Specification_Expr(Base):
    """
    <specification-expr> = <expr>
    """
    subclass_names = ['Expr']

class Char_Initialization_Expr(Base):
    """
    <char-initialization-expr> = <expr>
    """
    subclass_names = ['Char_Expr']

class Scalar_Char_Initialization_Expr(Base):
    subclass_names = ['Char_Initialization_Expr']

class Initialization_Expr(Base):
    """
    <initialization-expr> = <expr>
    """
    subclass_names = ['Expr']

class Int_Initialization_Expr(Base):
    """
    <int-initialization-expr> = <expr>
    """
    subclass_names = ['Int_Expr']

class Logical_Initialization_Expr(Base):
    """
    <logical-initialization-expr> = <expr>
    """
    subclass_names = ['Logical_Expr']

class Where_Stmt(Base):
    """
    <where-stmt> = WHERE ( <mask-expr> ) <where-assignment-stmt>
    """
    subclass_names = []
    use_names = ['Mask_Expr', 'Where_Assignment_Stmt']

class Where_Construct(Base):
    """
    <where-construct> = <where-construct-stmt>
                              [ <where-body-construct> ]...
                            [ <masked-elsewhere-stmt>
                              [ <where-body-construct> ]...
                            ]...
                            [ <elsewhere-stmt>
                              [ <where-body-construct> ]... ]
                            <end-where-stmt>
    """
    subclass_names = []
    use_names = ['Where_Construct_Stmt', 'Where_Body_Construct',
                 'Elsewhere_Stmt', 'End_Where_Stmt'
                 ]

class Where_Construct_Stmt(Base):
    """
    <where-construct-stmt> = [ <where-construct-name> : ] WHERE ( <masked-expr> )
    """
    subclass_names = []
    use_names = ['Where_Construct_Name', 'Masked_Expr']

class Where_Body_Construct(Base):
    """
    <where-body-construct> = <where-assignment-stmt>
                             | <where-stmt>
                             | <where-construct>
    """
    subclass_names = ['Where_Assignment_Stmt', 'Where_Stmt', 'Where_Construct']

class Where_Assignment_Stmt(Base):
    """
    <where-assignment-stmt> = <assignment-stmt>
    """
    subclass_names = ['Assignment_Stmt']

class Mask_Expr(Base):
    """
    <mask-expr> = <logical-expr>
    """
    subclass_names = ['Logical_Expr']

class Masked_Elsewhere_Stmt(Base):
    """
    <masked-elsewhere-stmt> = ELSEWHERE ( <mask-expr> ) [ <where-construct-name> ]
    """
    subclass_names = []
    use_names = ['Mask_Expr', 'Where_Construct_Name']

class Elsewhere_Stmt(Base):
    """
    <elsewhere-stmt> = ELSEWHERE [ <where-construct-name> ]
    """
    subclass_names = []
    use_names = ['Where_Construct_Name']

class End_Where_Stmt(Base):
    """
    <end-where-stmt> = END WHERE [ <where-construct-name> ]
    """
    subclass_names = []
    use_names = ['Where_Construct_Name']

class Proc_Language_Binding_Spec(Base): # R1125
    subclass_names = ['Language_Binding_Spec']


class Dummy_Arg_Name(Base): # R1226
    """
    <dummy-arg-name> = <name>
    """
    subclass_names = ['Name']

class Subroutine_Stmt(Base): # R1232
    """
    <subroutine-stmt> = [ <prefix> ] SUBROUTINE <subroutine-name> [ ( [ <dummy-arg-list> ] ) [ <proc-language-binding-spec> ] ]
    """
    subclass_names = []
    use_names = ['Prefix', 'Subroutine_Name', 'Dummy_Arg_List', 'Proc_Language_Binding_Spec']
    def match(string):
        line, repmap = string_replace_map(string)
        m = pattern.subroutine.search(line)
        prefix = line[:m.start()].rstrip() or None
        if prefix is not None:
            prefix = Prefix(repmap(prefix))
        line = line[m.end():].lstrip()
        m = pattern.name.match(line)
        if m is None: return
        name = Subroutine_Name(m.group())
        line = line[m.end():].lstrip()
        dummy_args = None
        if line.startswith('('):
            i = line.find(')')
            if i==-1: return
            dummy_args = line[1:i].strip() or None
            if dummy_args is not None:
                dummy_args = Dummy_Arg_List(repmap(dummy_args))
            line = line[i+1:].lstrip()
        binding_spec = None
        if line:
            binding_spec = Proc_Language_Binding_Spec(repmap(line))
        return prefix, name, dummy_args, binding_spec
    match = staticmethod(match)
    def init(self, *args):
        self.prefix, self.name, self.dummy_args, self.binding_spec = args
        return
    def tostr(self):
        if self.prefix is not None:
            s = '%s SUBROUTINE %s' % (self.prefix, self.name)
        else:
            s = 'SUBROUTINE %s' % (self.name)
        if self.dummy_args is not None:
            s += '(%s)' % (self.dummy_args)
        if self.binding_spec is not None:
            s += ' %s' % (self.binding_spec)
        return s
    def torepr(self):
        return '%s(%r, %r, %r, %r)'\
               % (self.__class__.__name__, self.prefix, self.name, self.dummy_args, self.binding_spec)

class End_Subroutine_Stmt(EndStmtBase): # R1234
    """
    <end-subroutine-stmt> = END [ SUBROUTINE [ <subroutine-name> ] ]
    """
    subclass_names = []
    use_names = ['Subroutine_Name']
    def match(string): return EndStmtBase.match('SUBROUTINE', Subroutine_Name, string)
    match = staticmethod(match)

class Dummy_Arg(StringBase): # R1233
    """
    <dummy-arg> = <dummy-arg-name>
                  | *
    """
    subclass_names = ['Dummy_Arg_Name']
    def match(string):
        if string=='*': return '*',
        return
    match = staticmethod(match)

# Create *_List classes:
ClassType = type(Base)
_names = dir()
for clsname in _names:
    cls = eval(clsname)
    if not (isinstance(cls, ClassType) and issubclass(cls, Base) and not cls.__name__.endswith('Base')): continue
    names = getattr(cls, 'subclass_names', []) + getattr(cls, 'use_names', [])
    for n in names:
        if n in _names: continue
        if n.endswith('_List'):
            n = n[:-5]
            print 'Generating %s_List' % (n)
            exec '''\
class %s_List(SequenceBase):
    subclass_names = [\'%s\']
    use_names = []
    def match(string): return SequenceBase.match(r\',\', %s, string)
    match = staticmethod(match)
''' % (n, n, n)
        if n.endswith('_Name'):
            n = n[:-5]
            print 'Generating %s_Name' % (n)
            exec '''\
class %s_Name(Base):
    subclass_names = [\'Name\']
''' % (n)

