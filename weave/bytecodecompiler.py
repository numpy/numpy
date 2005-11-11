#**************************************************************************#
#* FILE   **************    bytecodecompiler.py    ************************#
#************************************************************************ **#
#* Author: Patrick Miller February  9 2002                                *#
#* Copyright (C) 2002 Patrick J. Miller                                   *#
#**************************************************************************#
#*  *#
#**************************************************************************#
from types import *
import string
import inspect

##################################################################
#                       CLASS __DESCRIPTOR                       #
##################################################################
class __Descriptor:
    prerequisites = []
    refcount = 0
    def __repr__(self):
        return self.__module__+'.'+self.__class__.__name__

##################################################################
#                     CLASS TYPE_DESCRIPTOR                      #
##################################################################
class Type_Descriptor(__Descriptor):
    module_init_code = ''

##################################################################
#                   CLASS FUNCTION_DESCRIPTOR                    #
##################################################################
class Function_Descriptor(__Descriptor):
    def __init__(self,code,return_type,support=''):
	self.code	= code
	self.return_type	= return_type
        self.support = support
	return

        
            

haveArgument = 90 # Opcodes greater-equal to this have argument
byName = {
    'STOP_CODE': 0,
    'POP_TOP': 1,
    'ROT_TWO': 2,
    'ROT_THREE': 3,
    'DUP_TOP': 4,
    'ROT_FOUR': 5,
    'UNARY_POSITIVE': 10,
    'UNARY_NEGATIVE': 11,
    'UNARY_NOT': 12,
    'UNARY_CONVERT': 13,
    'UNARY_INVERT': 15,
    'BINARY_POWER': 19,
    'BINARY_MULTIPLY': 20,
    'BINARY_DIVIDE': 21,
    'BINARY_MODULO': 22,
    'BINARY_ADD': 23,
    'BINARY_SUBTRACT': 24,
    'BINARY_SUBSCR': 25,
    'BINARY_FLOOR_DIVIDE': 26,
    'BINARY_TRUE_DIVIDE': 27,
    'INPLACE_FLOOR_DIVIDE': 28,
    'INPLACE_TRUE_DIVIDE': 29,
    'SLICE': 30,
    'STORE_SLICE': 40,
    'DELETE_SLICE': 50,
    'INPLACE_ADD': 55,
    'INPLACE_SUBTRACT': 56,
    'INPLACE_MULTIPLY': 57,
    'INPLACE_DIVIDE': 58,
    'INPLACE_MODULO': 59,
    'STORE_SUBSCR': 60,
    'DELETE_SUBSCR': 61,
    'BINARY_LSHIFT': 62,
    'BINARY_RSHIFT': 63,
    'BINARY_AND': 64,
    'BINARY_XOR': 65,
    'BINARY_OR': 66,
    'INPLACE_POWER': 67,
    'GET_ITER': 68,
    'PRINT_EXPR': 70,
    'PRINT_ITEM': 71,
    'PRINT_NEWLINE': 72,
    'PRINT_ITEM_TO': 73,
    'PRINT_NEWLINE_TO': 74,
    'INPLACE_LSHIFT': 75,
    'INPLACE_RSHIFT': 76,
    'INPLACE_AND': 77,
    'INPLACE_XOR': 78,
    'INPLACE_OR': 79,
    'BREAK_LOOP': 80,
    'LOAD_LOCALS': 82,
    'RETURN_VALUE': 83,
    'IMPORT_STAR': 84,
    'EXEC_STMT': 85,
    'YIELD_VALUE': 86,
    'POP_BLOCK': 87,
    'END_FINALLY': 88,
    'BUILD_CLASS': 89,
    'STORE_NAME': 90,
    'DELETE_NAME': 91,
    'UNPACK_SEQUENCE': 92,
    'FOR_ITER': 93,
    'STORE_ATTR': 95,
    'DELETE_ATTR': 96,
    'STORE_GLOBAL': 97,
    'DELETE_GLOBAL': 98,
    'DUP_TOPX': 99,
    'LOAD_CONST': 100,
    'LOAD_NAME': 101,
    'BUILD_TUPLE': 102,
    'BUILD_LIST': 103,
    'BUILD_MAP': 104,
    'LOAD_ATTR': 105,
    'COMPARE_OP': 106,
    'IMPORT_NAME': 107,
    'IMPORT_FROM': 108,
    'JUMP_FORWARD': 110,
    'JUMP_IF_FALSE': 111,
    'JUMP_IF_TRUE': 112,
    'JUMP_ABSOLUTE': 113,
    'FOR_LOOP': 114,
    'LOAD_GLOBAL': 116,
    'CONTINUE_LOOP': 119,
    'SETUP_LOOP': 120,
    'SETUP_EXCEPT': 121,
    'SETUP_FINALLY': 122,
    'LOAD_FAST': 124,
    'STORE_FAST': 125,
    'DELETE_FAST': 126,
    'SET_LINENO': 127,
    'RAISE_VARARGS': 130,
    'CALL_FUNCTION': 131,
    'MAKE_FUNCTION': 132,
    'BUILD_SLICE': 133,
    'MAKE_CLOSURE': 134,
    'LOAD_CLOSURE': 135,
    'LOAD_DEREF': 136,
    'STORE_DEREF': 137,
    'CALL_FUNCTION_VAR': 140,
    'CALL_FUNCTION_KW': 141,
    'CALL_FUNCTION_VAR_KW': 142,
    }

# -----------------------------------------------
# Build one in the reverse sense
# -----------------------------------------------
byOpcode = {}
for name,op in map(None, byName.keys(), byName.values()):
    byOpcode[op] = name
    del name
    del op
    

##################################################################
#                       FUNCTION OPCODIZE                        #
##################################################################
def opcodize(s):
    "Slightly more readable form"
    length = len(s)
    i = 0
    answer = []
    while i < length:
        bytecode = ord(s[i])
        name = byOpcode[bytecode]
        if bytecode >= haveArgument:
            argument = 256*ord(s[i+2])+ord(s[i+1])
            i += 3
        else:
            argument = None
            i += 1
        answer.append((bytecode,argument,name))
    return answer



##################################################################
#                         FUNCTION LIST                          #
##################################################################
def listing(f):
    "Pretty print the internals of your function"
    assert type(f) == FunctionType,"Arg %r must be a function"%f
    filename = f.func_code.co_filename
    try:
        lines = open(filename).readlines()
    except:
        lines = None
    pc = 0
    s = ''
    lastLine = None
    for op,arg,name in opcodize(f.func_code.co_code):
        if lines and name == 'SET_LINENO':
            source = lines[arg-1][:-1]
            while lastLine and lastLine < arg-1:
                nonEmittingSource = lines[lastLine][:-1]
                lastLine += 1
                s += '%3s  %20s %5s : %s\n'%(
                    '','','',nonEmittingSource)
            lastLine = arg
        else:
            source = ''
        if arg is None: arg = ''
        s += '%3d] %20s %5s : %s\n'%(pc,name,arg,source)
        if op >= haveArgument:
            pc += 3
        else:
            pc += 1
    return s

##################################################################
#                     CLASS BYTECODEMEANING                      #
##################################################################
class ByteCodeMeaning:
    def fetch(self, pc,code):
        opcode = ord(code[pc])
        if opcode >= haveArgument:
            argument = 256*ord(code[pc+2])+ord(code[pc+1])
            next = pc+3
        else:
            argument = None
            next = pc+1
        return next,opcode,argument
    
    def execute(self,pc,opcode,argument):
        name = byOpcode[opcode]
        method = getattr(self,name)
        if argument is None:
            return apply(method,(pc,))
        else:
            return apply(method,(pc,argument,))

    def evaluate(self, pc,code):
        next, opcode,argument = self.fetch(pc,code)
        goto = self.execute(next,opcode,argument)
        if goto == -1:
            return None # Must be done
        elif goto is None:
            return next # Normal
        else:
            raise 'xx'

    symbols = { 0: 'less', 1: 'lesseq', 2: 'equal', 3: 'notequal',
                4: 'greater', 5: 'greatereq', 6: 'in', 7: 'not in',
                8: 'is', 9: 'is not', 10: 'exe match',
                11 : 'bad',
                }
    def cmp_op(self,opname):
        return self.symbols[opname]
    
    def STOP_CODE(self,pc):
        "Indicates end-of-code to the compiler, not used by the interpreter."
        raise NotImplementedError
    def POP_TOP(self,pc):
        "Removes the top-of-stack (TOS) item."
        raise NotImplementedError

    def ROT_TWO(self,pc):
        "Swaps the two top-most stack items."
        raise NotImplementedError

    def ROT_THREE(self,pc):
        "Lifts second and third stack item one position up, moves top down to position three."
        raise NotImplementedError

    def ROT_FOUR(self,pc):
        "Lifts second, third and forth stack item one position up, moves top down to position four."
        raise NotImplementedError

    def DUP_TOP(self,pc):
        "Duplicates the reference on top of the stack."
        raise NotImplementedError

    # Unary Operations take the top of the stack, apply the operation, and push the result back on the stack.

    def UNARY_POSITIVE(self,pc):
        "Implements TOS = +TOS."
        raise NotImplementedError

    def UNARY_NEGATIVE(self,pc):
        "Implements TOS = -TOS."
        raise NotImplementedError

    def UNARY_NOT(self,pc):
        "Implements TOS = not TOS."
        raise NotImplementedError

    def UNARY_CONVERT(self,pc):
        "Implements TOS = `TOS`."
        raise NotImplementedError

    def UNARY_INVERT(self,pc):
        "Implements TOS = ~TOS."
        raise NotImplementedError

    #Binary operations remove the top of the stack (TOS) and the second top-most stack item (TOS1) from the stack. They perform the operation, and put the result back on the stack.

    def BINARY_POWER(self,pc):
        "Implements TOS = TOS1 ** TOS."
        raise NotImplementedError

    def BINARY_MULTIPLY(self,pc):
        "Implements TOS = TOS1 * TOS."
        raise NotImplementedError

    def BINARY_DIVIDE(self,pc):
        "Implements TOS = TOS1 / TOS."
        raise NotImplementedError

    def BINARY_MODULO(self,pc):
        "Implements TOS = TOS1 % TOS."
        raise NotImplementedError

    def BINARY_ADD(self,pc):
        "Implements TOS = TOS1 + TOS."
        raise NotImplementedError

    def BINARY_SUBTRACT(self,pc):
        "Implements TOS = TOS1 - TOS."
        raise NotImplementedError

    def BINARY_SUBSCR(self,pc):
        "Implements TOS = TOS1[TOS]."
        raise NotImplementedError

    def BINARY_LSHIFT(self,pc):
        "Implements TOS = TOS1 << TOS."
        raise NotImplementedError

    def BINARY_RSHIFT(self,pc):
        "Implements TOS = TOS1 >> TOS."
        raise NotImplementedError

    def BINARY_AND(self,pc):
        "Implements TOS = TOS1 & TOS."
        raise NotImplementedError

    def BINARY_XOR(self,pc):
        "Implements TOS = TOS1 ^ TOS."
        raise NotImplementedError

    def BINARY_OR(self,pc):
        "Implements TOS = TOS1 | TOS."
        raise NotImplementedError

    #In-place operations are like binary operations, in that they remove TOS and TOS1, and push the result back on the stack, but the operation is done in-place when TOS1 supports it, and the resulting TOS may be (but does not have to be) the original TOS1.

    def INPLACE_POWER(self,pc):
        "Implements in-place TOS = TOS1 ** TOS."
        raise NotImplementedError

    def INPLACE_MULTIPLY(self,pc):
        "Implements in-place TOS = TOS1 * TOS."
        raise NotImplementedError

    def INPLACE_DIVIDE(self,pc):
        "Implements in-place TOS = TOS1 / TOS."
        raise NotImplementedError

    def INPLACE_MODULO(self,pc):
        "Implements in-place TOS = TOS1 % TOS."
        raise NotImplementedError

    def INPLACE_ADD(self,pc):
        "Implements in-place TOS = TOS1 + TOS."
        raise NotImplementedError

    def INPLACE_SUBTRACT(self,pc):
        "Implements in-place TOS = TOS1 - TOS."
        raise NotImplementedError

    def INPLACE_LSHIFT(self,pc):
        "Implements in-place TOS = TOS1 << TOS."
        raise NotImplementedError

    def INPLACE_RSHIFT(self,pc):
        "Implements in-place TOS = TOS1 >> TOS."
        raise NotImplementedError

    def INPLACE_AND(self,pc):
        "Implements in-place TOS = TOS1 & TOS."
        raise NotImplementedError

    def INPLACE_XOR(self,pc):
        "Implements in-place TOS = TOS1 ^ TOS."
        raise NotImplementedError

    def INPLACE_OR(self,pc):
        "Implements in-place TOS = TOS1 | TOS."
        raise NotImplementedError

    #The slice opcodes take up to three parameters.

    def SLICE_0(self,pc):
        "Implements TOS = TOS[:]."
        raise NotImplementedError

    def SLICE_1(self,pc):
        "Implements TOS = TOS1[TOS:]."
        raise NotImplementedError

    def SLICE_2(self,pc):
        "Implements TOS = TOS1[:TOS1]."
        raise NotImplementedError

    def SLICE_3(self,pc):
        "Implements TOS = TOS2[TOS1:TOS]."
        raise NotImplementedError

    #Slice assignment needs even an additional parameter. As any statement, they put nothing on the stack.

    def STORE_SLICE_0(self,pc):
        "Implements TOS[:] = TOS1."
        raise NotImplementedError

    def STORE_SLICE_1(self,pc):
        "Implements TOS1[TOS:] = TOS2."
        raise NotImplementedError

    def STORE_SLICE_2(self,pc):
        "Implements TOS1[:TOS] = TOS2."
        raise NotImplementedError

    def STORE_SLICE_3(self,pc):
        "Implements TOS2[TOS1:TOS] = TOS3."
        raise NotImplementedError

    def DELETE_SLICE_0(self,pc):
        "Implements del TOS[:]."
        raise NotImplementedError

    def DELETE_SLICE_1(self,pc):
        "Implements del TOS1[TOS:]."
        raise NotImplementedError

    def DELETE_SLICE_2(self,pc):
        "Implements del TOS1[:TOS]."
        raise NotImplementedError

    def DELETE_SLICE_3(self,pc):
        "Implements del TOS2[TOS1:TOS]."
        raise NotImplementedError

    def STORE_SUBSCR(self,pc):
        "Implements TOS1[TOS] = TOS2."
        raise NotImplementedError

    def DELETE_SUBSCR(self,pc):
        "Implements del TOS1[TOS]."
        raise NotImplementedError

    def PRINT_EXPR(self,pc):
        "Implements the expression statement for the interactive mode. TOS is removed from the stack and printed. In non-interactive mode, an expression statement is terminated with POP_STACK."
        raise NotImplementedError

    def PRINT_ITEM(self,pc):
        "Prints TOS to the file-like object bound to sys.stdout. There is one such instruction for each item in the print statement."
        raise NotImplementedError

    def PRINT_ITEM_TO(self,pc):
        "Like PRINT_ITEM, but prints the item second from TOS to the file-like object at TOS. This is used by the extended print statement."
        raise NotImplementedError

    def PRINT_NEWLINE(self,pc):
        "Prints a new line on sys.stdout. This is generated as the last operation of a print statement, unless the statement ends with a comma."
        raise NotImplementedError

    def PRINT_NEWLINE_TO(self,pc):
        "Like PRINT_NEWLINE, but prints the new line on the file-like object on the TOS. This is used by the extended print statement."
        raise NotImplementedError

    def BREAK_LOOP(self,pc):
        "Terminates a loop due to a break statement."
        raise NotImplementedError

    def LOAD_LOCALS(self,pc):
        "Pushes a reference to the locals of the current scope on the stack. This is used in the code for a class definition: After the class body is evaluated, the locals are passed to the class definition."
        raise NotImplementedError

    def RETURN_VALUE(self,pc):
        "Returns with TOS to the caller of the function."
        raise NotImplementedError

    def IMPORT_STAR(self,pc):
        "Loads all symbols not starting with _ directly from the module TOS to the local namespace. The module is popped after loading all names. This opcode implements from module import *."
        raise NotImplementedError

    def EXEC_STMT(self,pc):
        "Implements exec TOS2,TOS1,TOS. The compiler fills missing optional parameters with None."
        raise NotImplementedError

    def POP_BLOCK(self,pc):
        "Removes one block from the block stack. Per frame, there is a stack of blocks, denoting nested loops, try statements, and such."
        raise NotImplementedError

    def END_FINALLY(self,pc):
        "Terminates a finally clause. The interpreter recalls whether the exception has to be re-raised, or whether the function returns, and continues with the outer-next block."
        raise NotImplementedError

    def BUILD_CLASS(self,pc):
        "Creates a new class object. TOS is the methods dictionary, TOS1 the tuple of the names of the base classes, and TOS2 the class name."
        raise NotImplementedError

    #All of the following opcodes expect arguments. An argument is two bytes, with the more significant byte last.

    def STORE_NAME(self,pc,namei):
        "Implements name = TOS. namei is the index of name in the attribute co_names of the code object. The compiler tries to use STORE_LOCAL or STORE_GLOBAL if possible."
        raise NotImplementedError

    def DELETE_NAME(self,pc,namei):
        "Implements del name, where namei is the index into co_names attribute of the code object."
        raise NotImplementedError

    def UNPACK_SEQUENCE(self,pc,count):
        "Unpacks TOS into count individual values, which are put onto the stack right-to-left."
        raise NotImplementedError

    def DUP_TOPX(self,pc,count):
        "Duplicate count items, keeping them in the same order. Due to implementation limits, count should be between 1 and 5 inclusive."
        raise NotImplementedError

    def STORE_ATTR(self,pc,namei):
        "Implements TOS.name = TOS1, where namei is the index of name in co_names."
        raise NotImplementedError

    def DELETE_ATTR(self,pc,namei):
        "Implements del TOS.name, using namei as index into co_names."
        raise NotImplementedError

    def STORE_GLOBAL(self,pc,namei):
        "Works as STORE_NAME, but stores the name as a global."
        raise NotImplementedError

    def DELETE_GLOBAL(self,pc,namei):
        "Works as DELETE_NAME, but deletes a global name."
        raise NotImplementedError

    def LOAD_CONST(self,pc,consti):
        "Pushes co_consts[consti] onto the stack."
        raise NotImplementedError

    def LOAD_NAME(self,pc,namei):
        "Pushes the value associated with co_names[namei] onto the stack."
        raise NotImplementedError

    def BUILD_TUPLE(self,pc,count):
        "Creates a tuple consuming count items from the stack, and pushes the resulting tuple onto the stack."
        raise NotImplementedError

    def BUILD_LIST(self,pc,count):
        "Works as BUILD_TUPLE, but creates a list."
        raise NotImplementedError

    def BUILD_MAP(self,pc,zero):
        "Pushes a new empty dictionary object onto the stack. The argument is ignored and set to zero by the compiler."
        raise NotImplementedError

    def LOAD_ATTR(self,pc,namei):
        "Replaces TOS with getattr(TOS, co_names[namei]."
        raise NotImplementedError

    def COMPARE_OP(self,pc,opname):
        "Performs a Boolean operation. The operation name can be found in cmp_op[opname]."
        raise NotImplementedError

    def IMPORT_NAME(self,pc,namei):
        "Imports the module co_names[namei]. The module object is pushed onto the stack. The current namespace is not affected: for a proper import statement, a subsequent STORE_FAST instruction modifies the namespace."
        raise NotImplementedError

    def IMPORT_FROM(self,pc,namei):
        "Loads the attribute co_names[namei] from the module found in TOS. The resulting object is pushed onto the stack, to be subsequently stored by a STORE_FAST instruction."
        raise NotImplementedError

    def JUMP_FORWARD(self,pc,delta):
        "Increments byte code counter by delta."
        raise NotImplementedError

    def JUMP_IF_TRUE(self,pc,delta):
        "If TOS is true, increment the byte code counter by delta. TOS is left on the stack."
        raise NotImplementedError

    def JUMP_IF_FALSE(self,pc,delta):
        "If TOS is false, increment the byte code counter by delta. TOS is not changed."
        raise NotImplementedError

    def JUMP_ABSOLUTE(self,pc,target):
        "Set byte code counter to target."
        raise NotImplementedError

    def FOR_LOOP(self,pc,delta):
        "Iterate over a sequence. TOS is the current index, TOS1 the sequence. First, the next element is computed. If the sequence is exhausted, increment byte code counter by delta. Otherwise, push the sequence, the incremented counter, and the current item onto the stack."
        raise NotImplementedError

    def LOAD_GLOBAL(self,pc,namei):
        "Loads the global named co_names[namei] onto the stack."
        raise NotImplementedError

    def SETUP_LOOP(self,pc,delta):
        "Pushes a block for a loop onto the block stack. The block spans from the current instruction with a size of delta bytes."
        raise NotImplementedError

    def SETUP_EXCEPT(self,pc,delta):
        "Pushes a try block from a try-except clause onto the block stack. delta points to the first except block."
        raise NotImplementedError

    def SETUP_FINALLY(self,pc,delta):
        "Pushes a try block from a try-except clause onto the block stack. delta points to the finally block."
        raise NotImplementedError

    def LOAD_FAST(self,pc,var_num):
        "Pushes a reference to the local co_varnames[var_num] onto the stack."
        raise NotImplementedError

    def STORE_FAST(self,pc,var_num):
        "Stores TOS into the local co_varnames[var_num]."
        raise NotImplementedError

    def DELETE_FAST(self,pc,var_num):
        "Deletes local co_varnames[var_num]."
        raise NotImplementedError

    def LOAD_CLOSURE(self,pc,i):
        "Pushes a reference to the cell contained in slot i of the cell and free variable storage. The name of the variable is co_cellvars[i] if i is less than the length of co_cellvars. Otherwise it is co_freevars[i - len(co_cellvars)]."
        raise NotImplementedError

    def LOAD_DEREF(self,pc,i):
        "Loads the cell contained in slot i of the cell and free variable storage. Pushes a reference to the object the cell contains on the stack."
        raise NotImplementedError

    def STORE_DEREF(self,pc,i):
        "Stores TOS into the cell contained in slot i of the cell and free variable storage."
        raise NotImplementedError

    def SET_LINENO(self,pc,lineno):
        "Sets the current line number to lineno."
        raise NotImplementedError

    def RAISE_VARARGS(self,pc,argc):
        "Raises an exception. argc indicates the number of parameters to the raise statement, ranging from 0 to 3. The handler will find the traceback as TOS2, the parameter as TOS1, and the exception as TOS."
        raise NotImplementedError

    def CALL_FUNCTION(self,pc,argc):
        "Calls a function. The low byte of argc indicates the number of positional parameters, the high byte the number of keyword parameters. On the stack, the opcode finds the keyword parameters first. For each keyword argument, the value is on top of the key. Below the keyword parameters, the positional parameters are on the stack, with the right-most parameter on top. Below the parameters, the function object to call is on the stack."
        raise NotImplementedError

    def MAKE_FUNCTION(self,pc,argc):
        "Pushes a new function object on the stack. TOS is the code associated with the function. The function object is defined to have argc default parameters, which are found below TOS."
        raise NotImplementedError

    def MAKE_CLOSURE(self,pc,argc):
        "Creates a new function object, sets its func_closure slot, and pushes it on the stack. TOS is the code associated with the function. If the code object has N free variables, the next N items on the stack are the cells for these variables. The function also has argc default parameters, where are found before the cells."
        raise NotImplementedError

    def BUILD_SLICE(self,pc,argc):
        "Pushes a slice object on the stack. argc must be 2 or 3. If it is 2, slice(TOS1, TOS) is pushed; if it is 3, slice(TOS2, TOS1, TOS) is pushed. See the slice() built-in function for more information."
        raise NotImplementedError

    def EXTENDED_ARG(self,pc,ext):
        "Prefixes any opcode which has an argument too big to fit into the default two bytes. ext holds two additional bytes which, taken together with the subsequent opcode's argument, comprise a four-byte argument, ext being the two most-significant bytes."
        raise NotImplementedError

    def CALL_FUNCTION_VAR(self,pc,argc):
        "Calls a function. argc is interpreted as in CALL_FUNCTION. The top element on the stack contains the variable argument list, followed by keyword and positional arguments."
        raise NotImplementedError

    def CALL_FUNCTION_KW(self,pc,argc):
        "Calls a function. argc is interpreted as in CALL_FUNCTION. The top element on the stack contains the keyword arguments dictionary, followed by explicit keyword and positional arguments."
        raise NotImplementedError

    def CALL_FUNCTION_VAR_KW(self,pc,argc):
        "Calls a function. argc is interpreted as in CALL_FUNCTION. The top element on the stack contains the keyword arguments dictionary, followed by the variable-arguments tuple, followed by explicit keyword and positional arguments."
        raise NotImplementedError

    

##################################################################
#                         CLASS CXXCODER                         #
##################################################################
class CXXCoder(ByteCodeMeaning):

    ##################################################################
    #                    MEMBER TYPEDEF_BY_VALUE                     #
    ##################################################################
    def typedef_by_value(self,v):
        raise NotImplementedError # VIRTUAL
    
    ##################################################################
    #                        MEMBER __INIT__                         #
    ##################################################################
    def __init__(self,function,signature,name=None):
        assert type(function) == FunctionType,"Arg must be a user function"
        assert not function.func_defaults ,"Function cannot have default args (yet)"
        if name is None: name = function.func_name
        self.name = name
        self.function = function
        self.signature = signature
        self.codeobject = function.func_code
        self.__uid = 0 # Builds temps
        self.__indent = 1
        return

    ##################################################################
    #                        MEMBER EVALUATE                         #
    ##################################################################
    def evaluate(self, pc,code):
        # See if we posted any forwards for this offset
        if self.forwards.has_key(pc):
            for f in self.forwards[pc]:
                f()
            self.forwards[pc] = []
        return ByteCodeMeaning.evaluate(self,pc,code)
    
    ##################################################################
    #                        MEMBER GENERATE                         #
    ##################################################################
    def generate(self):
        self.forwards = {} # Actions on forward interprets
        self.__body = '' # Body will be built
        self.helpers = [] # headers and stuff

        # -----------------------------------------------
        # OK, crack open the function object and build
        # initial stack (not a real frame!)
        # -----------------------------------------------
        arglen = self.codeobject.co_argcount
        nlocals = self.codeobject.co_nlocals

        self.consts = self.codeobject.co_consts
        self.stack = list(self.codeobject.co_varnames)
        self.types = list(self.signature)+[None]*(nlocals-arglen)
        self.used = []
        for T in self.types:
            if T not in self.used: self.used.append(T)

        # -----------------------------------------------
        # One pass through the byte codes to generate
        # the body
        # -----------------------------------------------
        code = self.codeobject.co_code
        bytes = len(code)
        pc = 0
        while pc != None and pc < bytes:
            pc = self.evaluate(pc,code)

        # -----------------------------------------------
        # Return?
        # -----------------------------------------------
        if self.rtype == NoneType:
            rtype = 'void'
        else:
            rtype = self.rtype.cxxtype
            
        # -----------------------------------------------
        # Insert code body if available
        # -----------------------------------------------
        source = inspect.getsource(self.function)
        if not source: source = ''
        comments = inspect.getcomments(self.function)
        if comments: source = comments+source
        code = string.join(map(lambda x: '/////// '+x,string.split(source,'\n')),
                           '\n')+'\n'

        # -----------------------------------------------
        # Add in the headers
        # -----------------------------------------------
        code += '#include "Python.h"\n'
        for T in self.used:
            if T is None: continue
            for pre in T.prerequisites:
                code += pre
                code += '\n'

        # -----------------------------------------------
        # Real body
        # -----------------------------------------------
        code += '\n'
        code += '\nstatic %s %s('%(rtype,self.name)
        for i in range(len(self.signature)):
            if i != 0: code += ', '
            n = self.stack[i]
            t = self.types[i]
            code += '%s %s'%(t.cxxtype,n)
        code += ') {\n'
        code += ' PyObject* tempPY= 0;\n'


        # Add in non-argument temporaries
        # Assuming first argcount locals are positional args
        for i in range(self.codeobject.co_argcount,
                       self.codeobject.co_nlocals):
            t = self.types[i]
            code += '%s %s;\n'%(
                t.cxxtype,
                self.codeobject.co_varnames[i],
                )

        # Add in the body
        code += self.__body
        code += '}\n\n'
        return code


    ##################################################################
    #                      MEMBER WRAPPED_CODE                       #
    ##################################################################
    def wrapped_code(self):
        code = self.generate()
        
        # -----------------------------------------------
        # Wrapper
        # -----------------------------------------------
        code += 'static PyObject* wrapper_%s(PyObject*,PyObject* args) {\n'%self.name
        code += '  // Length check\n'
        code += '  if ( PyTuple_Size(args) != %d ) {\n'%len(self.signature)
        code += '     PyErr_SetString(PyExc_TypeError,"Expected %d arguments");\n'%len(self.signature)
        code += '     return 0;\n'
        code += '  }\n'

        code += '\n  // Load Py versions of args\n'
        for i in range(len(self.signature)):
            T = self.signature[i]
            code += '  PyObject* py_%s = PyTuple_GET_ITEM(args,%d);\n'%(
                self.codeobject.co_varnames[i],i
                )

            code += '  if ( !(%s) ) {\n'% \
                    T.check('py_'+self.codeobject.co_varnames[i])
            #code += '    PyObject_Print(py_A,stdout,0); puts("");\n'
            #code += '    printf("nd=%d typecode=%d\\n",((PyArrayObject*)py_A)->nd,((PyArrayObject*)py_A)->descr->type_num);\n'
            code += '    PyErr_SetString(PyExc_TypeError,"Bad type for arg %d (expected %s)");\n'%(
                i+1,
                T.__class__.__name__)
            code += '    return 0;\n'
            code += '  }\n'
        
        code += '\n  // Do conversions\n'
        argnames = []
        for i in range(len(self.signature)):
            T = self.signature[i]

            code += '  %s %s=%s;\n'%(
                T.cxxtype,
                self.codeobject.co_varnames[i],
                T.inbound('py_'+self.codeobject.co_varnames[i]),
                )
            code += '  if ( PyErr_Occurred() ) return 0;\n'
            argnames.append(self.codeobject.co_varnames[i])

        code += '\n  // Compute result\n'
        if self.rtype != NoneType:
            code += '  %s _result = '%(
                self.rtype.cxxtype,
                )
        else:
            code += '  '
        code += '%s(%s);\n'%(
            self.name,
            string.join(argnames,','),
            )


        code += '\n  // Pack return\n'
        if ( self.rtype == NoneType ):
            code += '  Py_INCREF(Py_None);\n'
            code += '  return Py_None;\n'
        else:
            result,owned = self.rtype.outbound('_result')
            if not owned:
                code += '  Py_INCREF(_result);\n'
            code += '  return %s;\n'%result
        code += '}\n'
        return code

    def indent(self):
        self.__indent += 1
        return

    def dedent(self):
        self.__indent -= 1
        return

    ##################################################################
    #                          MEMBER EMIT                           #
    ##################################################################
    def emit(self,s):
        self.__body += ' '*(3*self.__indent)
        self.__body += s
        self.__body += '\n'
        return

    ##################################################################
    #                          MEMBER PUSH                           #
    ##################################################################
    def push(self,v,t):
        self.stack.append(v)
        self.types.append(t)
        return

    ##################################################################
    #                           MEMBER POP                           #
    ##################################################################
    def pop(self):
        v = self.stack[-1]
        assert type(v) != TupleType
        del self.stack[-1]
        t = self.types[-1]
        assert type(t) != TupleType
        del self.types[-1]
        return v,t

    ##################################################################
    #                        MEMBER PUSHTUPLE                        #
    ##################################################################
    def pushTuple(self,V,T):
        assert type(V) == TupleType
        self.stack.append(V)
        assert type(V) == TupleType
        self.types.append(T)
        return


    ##################################################################
    #                        MEMBER POPTUPLE                         #
    ##################################################################
    def popTuple(self):
        v = self.stack[-1]
        assert type(v) == TupleType
        del self.stack[-1]
        t = self.types[-1]
        assert type(t) == TupleType
        del self.types[-1]
        return v,t
    ##################################################################
    #                        MEMBER MULTIARG                         #
    ##################################################################
    def multiarg(self):
        return type(self.stack[-1]) == TupleType
    
    ##################################################################
    #                         MEMBER UNIQUE                          #
    ##################################################################
    def unique(self):
        self.__uid += 1
        return 't%d'%self.__uid

    ##################################################################
    #                          MEMBER POST                           #
    ##################################################################
    def post(self,pc,action):
        if not self.forwards.has_key(pc):
            self.forwards[pc] = []
        self.forwards[pc].append(action)
        return

    ##################################################################
    #                       MEMBER EMIT_VALUE                        #
    ##################################################################
    def emit_value(self, v):
        descriptor = self.typedef_by_value(v)
    
        # Convert representation to CXX rhs
        rhs = descriptor.literalizer(v)
        lhs = self.unique()
        self.emit('%s %s = %s;'%(
            descriptor.cxxtype,
            lhs,
            rhs))
        self.push(lhs,descriptor)
        return        

    ##################################################################
    #                       MEMBER GLOBAL_INFO                       #
    ##################################################################
    def global_info(self,var_num):
        # This is the name value is known by
        var_name = self.codeobject.co_names[var_num]

        # First, figure out who owns this global
        import sys
        myHash = id(self.function.func_globals)
        for module_name in sys.modules.keys():
            module = sys.modules[module_name]
            if module and id(module.__dict__) == myHash:
                break
        else:
            raise ValueError,'Cannot locate module owning %s'%varname
        return module_name,var_name


    ##################################################################
    #                         MEMBER CODEUP                          #
    ##################################################################
    def codeup(self, rhs, rhs_type):
        lhs = self.unique()
        self.emit('%s %s = %s;\n'%(
            rhs_type.cxxtype,
            lhs,
            rhs))
        print self.__body
        self.push(lhs,rhs_type)
        return        
        

    ##################################################################
    #                          MEMBER BINOP                          #
    ##################################################################
    def binop(self,pc,symbol):
        v2,t2 = self.pop()
        v1,t1 = self.pop()

        if t1 == t2:
            rhs,rhs_type = t1.binop(symbol,v1,v2)
        else:
            rhs,rhs_type = t1.binopMixed(symbol,v1,v2,t2)

        self.codeup(rhs,rhs_type)
        return

    ##################################################################
    #                       MEMBER BINARY_XXX                        #
    ##################################################################
    def BINARY_ADD(self,pc):
        return self.binop(pc,'+')
    def BINARY_SUBTRACT(self,pc):
        return self.binop(pc,'-')
    def BINARY_MULTIPLY(self,pc):
        print 'MULTIPLY',self.stack[-2],self.types[-2],'*',self.stack[-1],self.types[-1]
        return self.binop(pc,'*')
    def BINARY_DIVIDE(self,pc):
        return self.binop(pc,'/')
    def BINARY_MODULO(self,pc):
        return self.binop(pc,'%')
    def BINARY_SUBSCR(self,pc):
        if self.multiarg():
            v2,t2 = self.popTuple()
        else:
            v2,t2 = self.pop()
            v2 = (v2,)
            t2 = (t2,)
        v1,t1 = self.pop()
        rhs,rhs_type = t1.getitem(v1,v2,t2)
        self.codeup(rhs,rhs_type)
        return

    def STORE_SUBSCR(self,pc):
        if self.multiarg():
            v2,t2 = self.popTuple()
        else:
            v2,t2 = self.pop()
            v2 = (v2,)
            t2 = (t2,)
        v1,t1 = self.pop()
        v0,t0 = self.pop()
        
        rhs,rhs_type = t1.setitem(v1,v2,t2)
        assert rhs_type == t0,"Store the right thing"
        self.emit('%s = %s;'%(rhs,v0))
        return

    def COMPARE_OP(self,pc,opname):
        symbol = self.cmp_op(opname) # convert numeric to name
        return self.binop(pc,symbol)


    ##################################################################
    #                       MEMBER PRINT_ITEM                        #
    ##################################################################
    def PRINT_ITEM(self,pc):
        # Printing correctly is tricky... best to let Python
        # do the real work here
        w = self.unique()
        self.emit('PyObject* %s = PySys_GetObject("stdout");'%w)
        self.emit('if (PyFile_SoftSpace(%s,1)) PyFile_WriteString(" ",%s);'%(w,w))
        v,t = self.pop()

        py = self.unique()
        code,owned = t.outbound(v)
        self.emit('PyObject* %s = %s;'%(py, code))
        self.emit('PyFile_WriteObject(%s,%s,Py_PRINT_RAW);'%(
            py,w))
        if owned:
            self.emit('Py_XDECREF(%s);'%py)
        return


    ##################################################################
    #                      MEMBER PRINT_NEWLINE                      #
    ##################################################################
    def PRINT_NEWLINE(self,pc):
        # Printing correctly is tricky... best to let Python
        # do the real work here
        w = self.unique()
        self.emit('PyObject* %s = PySys_GetObject("stdout");'%w)
        self.emit('PyFile_WriteString("\\n",%s);'%w);
        self.emit('PyFile_SoftSpace(%s,0);'%w);
        return
        
    ##################################################################
    #                       MEMBER SET_LINENO                        #
    ##################################################################
    def SET_LINENO(self,pc,lineno):
        self.emit('// %s:%d'%(self.codeobject.co_filename,lineno))
        return

    ##################################################################
    #                         MEMBER POP_TOP                         #
    ##################################################################
    def POP_TOP(self,pc):
        v,t = self.pop()
        return

    ##################################################################
    #                       MEMBER LOAD_CONST                        #
    ##################################################################
    def LOAD_CONST(self,pc,consti):
        # Fetch the constant
        k = self.consts[consti]
        t = type(k)
        print 'LOAD_CONST',repr(k),t

        # Fetch a None is just skipped
        if t == NoneType:
            self.push('<void>',t) 
            return

        self.emit_value(k)
        return


    ##################################################################
    #                       MEMBER BUILD_TUPLE                       #
    ##################################################################
    def BUILD_TUPLE(self,pc,count):
        "Creates a tuple consuming count items from the stack, and pushes the resulting tuple onto the stack."
        V = []
        T = []
        for i in range(count):
            v,t = self.pop()
            V.append(v)
            T.append(t)
        V.reverse()
        T.reverse()
        self.pushTuple(tuple(V),tuple(T))
        return

    ##################################################################
    #                        MEMBER LOAD_FAST                        #
    ##################################################################
    def LOAD_FAST(self,pc,var_num):
        v = self.stack[var_num]
        t = self.types[var_num]
        print 'LOADFAST',var_num,v,t
        for VV,TT in map(None, self.stack, self.types):
            print VV,':',TT
        if t is None:
            raise TypeError,'%s used before set?'%v
            print self.__body
            print 'PC',pc
        self.push(v,t)
        return


    ##################################################################
    #                        MEMBER LOAD_ATTR                        #
    ##################################################################
    def LOAD_ATTR(self,pc,namei):
        v,t = self.pop()
        attr_name = self.codeobject.co_names[namei]
        print 'LOAD_ATTR',namei,v,t,attr_name
        aType,aCode = t.get_attribute(attr_name)
        print 'ATTR',aType
        print aCode
        lhs = self.unique()
        rhs = v
        lhsType = aType.cxxtype
        self.emit(aCode%locals())
        self.push(lhs,aType)
        return


    ##################################################################
    #                       MEMBER STORE_ATTR                        #
    ##################################################################
    def STORE_ATTR(self,pc,namei):
        v,t = self.pop()
        attr_name = self.codeobject.co_names[namei]
        print 'STORE_ATTR',namei,v,t,attr_name
        v2,t2 = self.pop()
        print 'SA value',v2,t2
        aType,aCode = t.set_attribute(attr_name)
        print 'ATTR',aType
        print aCode
        assert t2 is aType
        rhs = v2
        lhs = v
        self.emit(aCode%locals())
        return

    ##################################################################
    #                       MEMBER LOAD_GLOBAL                       #
    ##################################################################
    def LOAD_GLOBAL(self,pc,var_num):
        # Figure out the name and load it
        try:
            F = self.function.func_globals[self.codeobject.co_names[var_num]]
        except:
            F = __builtins__[self.codeobject.co_names[var_num]]

        # For functions, we see if we know about this function
        if callable(F):
            self.push(F,type(F))
            return

        # We need the name of the module that matches
        # the global state for the function and
        # the name of the variable
        module_name,var_name = self.global_info(var_num)

        # We hope it's type is correct
        t = type(F)
        descriptor = typedefs[t]
        native = self.unique()
        py = self.unique()
        mod = self.unique()

        self.emit('')
        self.emit('PyObject* %s = PyImport_ImportModule("%s");'%(
            mod,module_name))
        self.emit('PyObject* %s = PyObject_GetAttrString(%s,"%s");'%(
            py,mod,var_name))
        self.emit('%s %s = %s;'%(
            descriptor.cxxtype,
            native,
            descriptor.inbound%py))

        self.push(native,t)
        return

    def SETUP_LOOP(self,pc,delta):
        "Pushes a block for a loop onto the block stack. The block spans from the current instruction with a size of delta bytes."
        return

    def FOR_LOOP(self,pc,delta):
        "Iterate over a sequence. TOS is the current index, TOS1 the sequence. First, the next element is computed. If the sequence is exhausted, increment byte code counter by delta. Otherwise, push the sequence, the incremented counter, and the current item onto the stack."
        # Pull off control variable and range info
        v2,t2 = self.pop()
        v1,t1 = self.pop()
        self.emit('for(%s=%s.low; %s<%s.high; %s += %s.step) {'%(
            v2,v1,v2,v1,v2,v1))

        # Put range back on for assignment
        self.push(v2,t2)
        return

    def JUMP_ABSOLUTE(self,pc,target):
        "Set byte code counter to target."
        self.emit('}')
        return

    def POP_BLOCK(self,pc):
        "Removes one block from the block stack. Per frame, there is a stack of blocks, denoting nested loops, try statements, and such."
        return


    ##################################################################
    #                       MEMBER STORE_FAST                        #
    ##################################################################
    def STORE_FAST(self,pc,var_num):

        v,t = self.pop()
        print 'STORE FAST',var_num,v,t

        save = self.stack[var_num]
        saveT = self.types[var_num]

        # See if type is same....
        # Note that None means no assignment made yet
        if saveT is None or t == saveT:
            if t.refcount:
                self.emit('Py_XINCREF(%s);'%v)
                self.emit('Py_XDECREF(%s);'%save)
            self.emit('%s = %s;\n'%(save,v))
            self.types[var_num] = t
            return

        raise TypeError,(t,saveT)

    ##################################################################
    #                      MEMBER STORE_GLOBAL                       #
    ##################################################################
    def STORE_GLOBAL(self,pc,var_num):

        # We need the name of the module that matches
        # the global state for the function and
        # the name of the variable
        module_name,var_name = self.global_info(var_num)

        # Convert the value to Python object
        v,t = self.pop()
        descriptor = typedefs[t]
        py = self.unique()
        code,owned = descriptor.outbound(v)
        self.emit('PyObject* %s = %s;'%(py,code))
        if not owned:
            self.emit('Py_INCREF(%s);'%py)
        mod = self.unique()
        self.emit('PyObject* %s = PyImport_ImportModule("%s");'%(
            mod,module_name))
        self.emit('PyObject_SetAttrString(%s,"%s",%s);'%(
            mod,var_name,py))
        self.emit('Py_DECREF(%s);'%py)
        return

    ##################################################################
    #                      MEMBER CALL_FUNCTION                      #
    ##################################################################
    def CALL_FUNCTION(self,pc,argc):
        # Pull args off stack
        args = []
        types = []
        for i in range(argc):
            v,t = self.pop()
            args = [v]+args
            types = [t]+types
 
        # Pull function object off stack and get descriptor
        f,t = self.pop()
        signature = (f,tuple(types))
        descriptor = self.function_by_signature(signature)
        #self.prerequisites += descriptor['prerequisite']+'\n'
        
        # Build a rhs
        rhs = descriptor.code%string.join(args,',')

        # Build a statement
        temp = self.unique()
        self.emit('%s %s = %s;\n'%(
            descriptor.return_type.cxxtype,
            temp,
            rhs))

        self.push(temp,descriptor.return_type)
        return


    ##################################################################
    #                      MEMBER JUMP_IF_FALSE                      #
    ##################################################################
    def JUMP_IF_FALSE(self,pc,delta):
        v,t = self.pop()
        self.push(v,t)
        # We need to do some work when we get to the
        # else part (put the value that's gonna get
        # popped back on the stack, emit } else {,
        # ...)
        action = lambda v=v,t=t,self=self: (
            self.emit('} else {'),
            self.push(v,t),
            )
        self.post(pc+delta,action)
        if t != IntType: raise TypeError, 'Invalid comparison type %s'%t
        self.emit('if (%s) {\n'%v)
        return
    

    ##################################################################
    #                      MEMBER JUMP_FORWARD                       #
    ##################################################################
    def JUMP_FORWARD(self,pc,delta):
        # We need to close the if after the delta
        action = lambda self=self: (
            self.emit('}'),
            )
        self.post(pc+delta,action)
        return
    
    ##################################################################
    #                      MEMBER RETURN_VALUE                       #
    ##################################################################
    def RETURN_VALUE(self,pc):
        v,t = self.pop()
        if hasattr(self,'rtype'):
            print v,t
            if t == NoneType: return # just the extra return
            raise ValueError,'multiple returns'
        self.rtype = t
        if t == NoneType:
            self.emit('return;')
        else:
            self.emit('return %s;'%v)
        print 'return with',v
        return

