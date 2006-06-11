
from block import *
from stmt import *

basic_blocks = [ProgramBlock,PythonModuleBlock,ModuleBlock,BlockDataBlock,
                SubroutineBlock,FunctionBlock,InterfaceBlock,TypeBlock]
stmt_blocks = [DoBlock,IfThenBlock,SelectBlock]

Block.classes['free90'] = [ProgramBlock,ModuleBlock,BlockDataBlock,
                           SubroutineBlock,FunctionBlock,InterfaceBlock,TypeBlock] + stmt_blocks
Block.classes['fix90'] = Block.classes['free90']
Block.classes['fix77'] = [ProgramBlock,BlockDataBlock,SubroutineBlock,FunctionBlock] + stmt_blocks
Block.classes['pyf'] = [PythonModuleBlock] + Block.classes['free90']

ProgramBlock.classes['free90'] = [ModuleBlock,SubroutineBlock,FunctionBlock,InterfaceBlock,TypeBlock] + stmt_blocks
ProgramBlock.classes['fix90'] = ProgramBlock.classes['free90']
ProgramBlock.classes['fix77'] = [SubroutineBlock,FunctionBlock] + stmt_blocks
ProgramBlock.classes['pyf'] = ProgramBlock.classes['free90']

ModuleBlock.classes['free90'] = [ModuleBlock,SubroutineBlock,FunctionBlock,InterfaceBlock,TypeBlock]
ModuleBlock.classes['fix90'] = ModuleBlock.classes['free90']
ModuleBlock.classes['fix77'] = []
ModuleBlock.classes['pyf'] = ModuleBlock.classes['free90']

BlockDataBlock.classes['free90'] = [TypeBlock]
BlockDataBlock.classes['fix90'] = BlockDataBlock.classes['free90']
BlockDataBlock.classes['fix77'] = []
BlockDataBlock.classes['pyf'] = BlockDataBlock.classes['free90']


PythonModuleBlock.classes['free90'] = [ModuleBlock,SubroutineBlock,FunctionBlock,InterfaceBlock,TypeBlock]
PythonModuleBlock.classes['fix90'] = PythonModuleBlock.classes['free90']
PythonModuleBlock.classes['fix77'] = []
PythonModuleBlock.classes['pyf'] = PythonModuleBlock.classes['free90']

InterfaceBlock.classes['free90'] = [ModuleBlock,SubroutineBlock,FunctionBlock,InterfaceBlock,TypeBlock] + stmt_blocks
InterfaceBlock.classes['fix90'] = InterfaceBlock.classes['free90']
InterfaceBlock.classes['fix77'] = []
InterfaceBlock.classes['pyf'] = InterfaceBlock.classes['free90']

SubroutineBlock.classes['free90'] = [InterfaceBlock,TypeBlock] + stmt_blocks
SubroutineBlock.classes['fix90'] = SubroutineBlock.classes['free90']
SubroutineBlock.classes['fix77'] = stmt_blocks
SubroutineBlock.classes['pyf'] = SubroutineBlock.classes['free90']

FunctionBlock.classes = SubroutineBlock.classes

TypeBlock.classes['free90'] = [ModuleBlock, SubroutineBlock, FunctionBlock, InterfaceBlock, TypeBlock] + stmt_blocks
TypeBlock.classes['fix90'] = TypeBlock.classes['free90']
TypeBlock.classes['fix77'] = []
TypeBlock.classes['pyf'] = TypeBlock.classes['free90']

StatementBlock.classes['free90'] = stmt_blocks
StatementBlock.classes['fix90'] = StatementBlock.classes['free90']
StatementBlock.classes['fix77'] = stmt_blocks
StatementBlock.classes['pyf'] = StatementBlock.classes['free90']


# Initialize stmt_cls attributes

ProgramBlock.stmt_cls = Program
ModuleBlock.stmt_cls = Module
PythonModuleBlock.stmt_cls = PythonModule
BlockDataBlock.stmt_cls = BlockData
InterfaceBlock.stmt_cls = Interface
SubroutineBlock.stmt_cls = Subroutine
FunctionBlock.stmt_cls = Function
TypeBlock.stmt_cls = Type

IfThenBlock.stmt_cls = IfThen
DoBlock.stmt_cls = Do
SelectBlock.stmt_cls = Select

ProgramBlock.end_stmt_cls = EndProgram
ModuleBlock.end_stmt_cls = EndModule
PythonModuleBlock.end_stmt_cls = EndPythonModule
BlockDataBlock.end_stmt_cls = EndBlockData
InterfaceBlock.end_stmt_cls = EndInterface
SubroutineBlock.end_stmt_cls = EndSubroutine
FunctionBlock.end_stmt_cls = EndFunction
TypeBlock.end_stmt_cls = EndType

IfThenBlock.end_stmt_cls = EndIfThen
DoBlock.end_stmt_cls = EndDo
SelectBlock.end_stmt_cls = EndSelect
