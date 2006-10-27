
from expressions import *

class Program(Base):
    """
    <program> = <program-unit>
                  [ <program-unit> ] ...
    """
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
    use_names = ['Function_Stmt', 'Specification_Part', 'Execution_Part',
                 'Internal_Subprogram_Part', 'End_Function_Stmt']

class Subroutine_Subprogram(Base):
    """
    <subroutine-subprogram> = <subroutine-stmt>
                                 [ <specification-part> ]
                                 [ <execution-part> ]
                                 [ <internal-subprogram-part> ]
                              <end-subroutine-stmt>
    """
    use_names = ['Subroutine_Stmt', 'Specification_Part', 'Execution_Part',
                 'Internal_Subprogram_Part', 'End_Subroutine_Stmt']

class Subroutine_Stmt(Base):
    """
    <subroutine-stmt> = [ <prefix> ] SUBROUTINE <subroutine-name> [ ( [ <dummy-arg-list> ] ) [ <proc-language-binding-spec> ] ]
    """
    use_names = ['Prefix', 'Subroutine_Name', 'Dummy_Arg_List', 'Proc_Language_Binding_Spec']

class End_Subroutine_Stmt(Base):
    """
    <end-subroutine-stmt> = END [ SUBROUTINE [ <subroutine-name> ] ]
    """
    use_names = ['Subroutine_Name']

class Specification_Part(Base):
    """
    <specification-part> = [ <use-stmt> ]...
                             [ <import-stmt> ]...
                             [ <implicit-part> ]
                             [ <declaration-construct> ]...
    """
    use_names = ['Use_Stmt', 'Import_Stmt', 'Implicit_Part', 'Declaration_Construct']

class Implicit_Part(Base):
    """
    <implicit-part> = [ <implicit-part-stmt> ]...
                        <implicit-stmt>
    """
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

class Derived_Type_Def(Base):
    """
    <derived-type-def> = <derived-type-stmt>
                           [ <type-param-def-stmt> ]...
                           [ <private-or-sequence> ]...
                           [ <component-part> ]
                           [ <type-bound-procedure-part> ]
                           <end-type-stmt>
    """
    use_names = ['Derived_Type_Stmt', 'Type_Param_Def_Stmt', 'Private_Or_Sequence',
                 'Component_Part', 'Type_Bound_Procedure_Part', 'End_Type_Stmt']

class Derived_Type_Stmt(Base):
    """
    <derived-type-stmt> = TYPE [ [ , <type-attr-spec-list> ] :: ] <type-name> [ ( <type-param-name-list> ) ]
    """
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
    use_names = ['Type_Name']
    
class Sequence_Stmt(Base):
    """
    <sequence-stmt> = SEQUENCE
    """
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
    use_names = ['Kind_Selector', 'Type_Param_Attr_Spec', 'Type_Param_Decl_List']

class Type_Param_Decl(Base):
    """
    <type-param-decl> = <type-param-name> [ = <scalar-int-initialization-expr> ]
    """
    use_names = ['Type_Param_Name', 'Scalar_Int_Initialization_Expr']

class Type_Param_Attr_Spec(Base):
    """
    <type-param-attr-spec> = KIND
                             | LEN
    """

class Component_Part(Base):
    """
    <component-part> = [ <component-def-stmt> ]...
    """
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
    use_names = ['Initialization-expr', 'Null_Init']

class Proc_Component_Def_Stmt(Base):
    """
    <proc-component-def-stmt> = PROCEDURE ( [ <proc-interface> ] ) , <proc-component-attr-spec-list> :: <proc-decl-list>
    """
    use_names = ['Proc_Interface', 'Proc_Component_Attr_Spec_List', 'Proc_Decl_List']

class Proc_Component_Attr_Spec(Base):
    """
    <proc-component-attr-spec> = POINTER
                                 | PASS [ ( <arg-name> ) ]
                                 | NOPASS
                                 | <access-spec>
    """
    use_names = ['Arg_Name', 'Access_Spec']

class Private_Components_Stmt(Base):
    """
    <private-components-stmt> = PRIVATE
    """

class Type_Bound_Procedure_Part(Base):
    """
    <type-bound-procedure-part> = <contains-stmt>
                                      [ <binding-private-stmt> ]
                                      <proc-binding-stmt>
                                      [ <proc-binding-stmt> ]...
    """
    use_names = ['Contains_Stmt', 'Binding_Private_Stmt', 'Proc_Binding_Stmt']

class Binding_Private_Stmt(Base):
    """
    <binding-private-stmt> = PRIVATE
    """

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
    use_names = ['Interface_Name', 'Binding_Attr_List', 'Binding_Name', 'Procedure_Name']

class Generic_Binding(Base):
    """
    <generic-binding> = GENERIC [ , <access-spec> ] :: <generic-spec> => <binding-name-list>
    """
    use_names = ['Access_Spec', 'Generic_Spec', 'Binding_Name_List']

class Binding_Attr(Base):
    """
    <binding-attr> = PASS [ ( <arg-name> ) ]
                     | NOPASS
                     | NON_OVERRIDABLE
                     | <access-spec>
    """
    use_names = ['Arg_Name', 'Access_Spec']

class Final_Binding(Base):
    """
    <final-binding> = FINAL [ :: ] <final-subroutine-name-list>
    """
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
    use_names = ['Enum_Def_Stmt', 'Enumerator_Def_Stmt', 'End_Enum_Stmt']

class Enum_Def_Stmt(Base):
    """
    <enum-def-stmt> = ENUM, BIND(C)
    """

class Enumerator_Def_Stmt(Base):
    """
    <enumerator-def-stmt> = ENUMERATOR [ :: ] <enumerator-list>
    """
    use_names = ['Enumerator_List']

class Enumerator(Base):
    """
    <enumerator> = <named-constant> [ = <scalar-int-initialization-expr> ]
    """
    use_names = ['Named_Constant', 'Scalar_Int_Initialization_Expr']

class End_Enumerator_Stmt(Base):
    """
    <end-enumerator-stmt> = END ENUM
    """

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
    use_names = ['Declaration_Type_Spec', 'Attr_Spec', 'Entity_Decl_List']

class Declaration_Type_Spec(Base):
    """
    <declaration-type-spec> = <intrinsic-type-spec>
                              | TYPE ( <derived-type-spec> )
                              | CLASS ( <derived-type-spec> )
                              | CLASS ( * )
    """
    subclass_names = ['Intrinsic_Type_Spec']
    use_names = ['Derived_Type_Spec']

class Attr_Spec(Base):
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
    subclass_names = ['Access_Spec', 'Language_Binding_Spec']
    use_names = ['Array_Spec', 'Intent_Spec']

class Entity_Decl(Base):
    """
    <entity-decl> = <object-name> [ ( <array-spec> ) ] [ * <char-length> ] [ <initialization> ]
                    | <function-name> [ * <char-length> ]
    """
    use_names = ['Object_Name', 'Array_Spec', 'Char_Length', 'Initialization', 'Function_Name']

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
    use_names = ['Initialization_Expr', 'Null_Init']

class Null_Init(Base):
    """
    <null-init> = <function-reference>

    <function-reference> shall be a reference to the NULL intrinsic function with no arguments.
    """
    subclass_names = ['Function_Reference']

class Access_Spec(Base):
    """
    <access-spec> = PUBLIC
                    | PRIVATE
    """

class Language_Binding_Spec(Base):
    """
    <language-binding-spec> = BIND ( C [ , NAME = <scalar-char-initialization-expr> ] )
    """
    use_names = ['Scalar_Char_Initialization_Expr']

class Array_Spec(Base):
    """
    <array-spec> = <explicit-shape-spec-list>
                   | <assumed-shape-spec-list>
                   | <deferred-shape-spec-list>
                   | <assumed-size-spec>
    """
    subclass_names = ['Explicit_Shape_Spec_List', 'Assumed_Shape_Spec_List',
                      'Deferred_Shape_Spec_List', 'Assumed_Size_Spec']

class Explicit_Shape_Spec(Base):
    """
    <explicit-shape-spec> = [ <lower-bound> : ] <upper-bound>
    """
    use_names = ['Lower_Bound', 'Upper_Bound']


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
    use_names = ['Lower_Bound']

class Deferred_Shape_Spec(Base):
    """
    <deferred_shape_spec> = :
    """

class Assumed_Size_Spec(Base):
    """
    <assumed-size-spec> = [ <explicit-shape-spec-list> , ] [ <lower-bound> : ] *
    """
    use_names = ['Explicit_Shape_Spec_List', 'Lower_Bound']

class Intent_Spec(Base):
    """
    <intent-spec> = IN
                    | OUT
                    | INOUT
    """

class Access_Stmt(Base):
    """
    <access-stmt> = <access-spec> [ [ :: ] <access-id-list> ]
    """
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
    use_names = ['Object_Name', 'Deferred_Shape_Spec_List']
