import parser
import string
import copy
import os,sys
import ast_tools
import token,symbol
import slice_handler
import size_check
import converters

from ast_tools import *

from Numeric import *
# The following try/except so that non-SciPy users can still use blitz
try:
    from scipy_base.fastumath import *
except:
    pass # scipy_base.fastumath not available    
    
from types import *

import inline_tools
from inline_tools import attempt_function_call
function_catalog = inline_tools.function_catalog
function_cache = inline_tools.function_cache
  
def blitz(expr,local_dict=None, global_dict=None,check_size=1,verbose=0,**kw):
    # this could call inline, but making a copy of the
    # code here is more efficient for several reasons.
    global function_catalog
          
    # this grabs the local variables from the *previous* call
    # frame -- that is the locals from the function that called
    # inline.
    call_frame = sys._getframe().f_back
    if local_dict is None:
        local_dict = call_frame.f_locals
    if global_dict is None:
        global_dict = call_frame.f_globals

    # 1. Check the sizes of the arrays and make sure they are compatible.
    #    This is expensive, so unsetting the check_size flag can save a lot
    #    of time.  It also can cause core-dumps if the sizes of the inputs 
    #    aren't compatible.    
    if check_size and not size_check.check_expr(expr,local_dict,global_dict):
        raise 'inputs failed to pass size check.'
    
    # 2. try local cache    
    try:
        results = apply(function_cache[expr],(local_dict,global_dict))
        return results
    except: 
        pass
    try:
        results = attempt_function_call(expr,local_dict,global_dict)
    # 3. build the function    
    except ValueError:
        # This section is pretty much the only difference 
        # between blitz and inline
        ast = parser.suite(expr)
        ast_list = ast.tolist()
        expr_code = ast_to_blitz_expr(ast_list)
        arg_names = harvest_variables(ast_list)
        module_dir = global_dict.get('__file__',None)
        #func = inline_tools.compile_function(expr_code,arg_names,
        #                                    local_dict,global_dict,
        #                                    module_dir,auto_downcast = 1)
        func = inline_tools.compile_function(expr_code,arg_names,local_dict,        
                                             global_dict,module_dir,
                                             compiler='gcc',auto_downcast=1,
                                             verbose = verbose,
                                             type_converters = converters.blitz,
                                             **kw)
        function_catalog.add_function(expr,func,module_dir)
        try:                                            
            results = attempt_function_call(expr,local_dict,global_dict)
        except ValueError:                                                
            print 'warning: compilation failed. Executing as python code'
            exec expr in global_dict, local_dict
            
def ast_to_blitz_expr(ast_seq):
    """ Convert an ast_sequence to a blitz expression.
    """
    
    # Don't overwrite orignal sequence in call to transform slices.
    ast_seq = copy.deepcopy(ast_seq)    
    slice_handler.transform_slices(ast_seq)
    
    # Build the actual program statement from ast_seq
    expr = ast_tools.ast_to_string(ast_seq)
    
    # Now find and replace specific symbols to convert this to
    # a blitz++ compatible statement.
    # I'm doing this with string replacement here.  It could
    # also be done on the actual ast tree (and probably should from
    # a purest standpoint...).
    
    # this one isn't necessary but it helps code readability
    # and compactness. It requires that 
    #   Range _all = blitz::Range::all();
    # be included in the generated code.    
    # These could all alternatively be done to the ast in
    # build_slice_atom()
    expr = string.replace(expr,'slice(_beg,_end)', '_all' )    
    expr = string.replace(expr,'slice', 'blitz::Range' )
    expr = string.replace(expr,'[','(')
    expr = string.replace(expr,']', ')' )
    expr = string.replace(expr,'_stp', '1' )
    
    # Instead of blitz::fromStart and blitz::toEnd.  This requires
    # the following in the generated code.
    #   Range _beg = blitz::fromStart;
    #   Range _end = blitz::toEnd;
    #expr = string.replace(expr,'_beg', 'blitz::fromStart' )
    #expr = string.replace(expr,'_end', 'blitz::toEnd' )
    
    return expr + ';\n'

def test_function():
    expr = "ex[:,1:,1:] = k +  ca_x[:,1:,1:] * ex[:,1:,1:]" \
                         "+ cb_y_x[:,1:,1:] * (hz[:,1:,1:] - hz[:,:-1,1:])"\
                         "- cb_z_x[:,1:,1:] * (hy[:,1:,1:] - hy[:,1:,:-1])"        
    #ast = parser.suite('a = (b + c) * sin(d)')
    ast = parser.suite(expr)
    k = 1.
    ex = ones((1,1,1),typecode=Float32)
    ca_x = ones((1,1,1),typecode=Float32)
    cb_y_x = ones((1,1,1),typecode=Float32)
    cb_z_x = ones((1,1,1),typecode=Float32)
    hz = ones((1,1,1),typecode=Float32)
    hy = ones((1,1,1),typecode=Float32)
    blitz(expr)
