import pprint
import string
from ast_tools import * 

def slice_ast_to_dict(ast_seq):
    sl_vars = {}
    if type(ast_seq) in (ListType,TupleType):
        for pattern in slice_patterns:
            found,data = match(pattern,ast_seq)
            if found:    
                sl_vars = {'begin':'_beg',
                           'end':'_end', 
                           'step':'_stp',
                           'single_index':'_index'}
                for key in data.keys():
                    data[key] = ast_to_string(data[key])
                sl_vars.update(data)
                break;        
    return sl_vars
            
def build_slice_atom(slice_vars, position):
    # Note: This produces slices that are incorrect for Python
    # evaluation because of slicing being exclusive in Python
    # and inclusive for blitz on the top end of the range.
    # This difference should really be handle in a blitz specific transform,
    # but I've put it here for convenience. This doesn't cause any
    # problems in code, its just a maintance hassle (I'll forget I did it here)
    # and inelegant.  *FIX ME*.
    
        
    ###########################################################################
    #                        Handling negative indices.
    #
    # Range indices that begin with a negative sign, '-', are assumed to be
    # negative.  Blitz++ interprets negative indices differently than 
    # Python.  To correct this, we subtract negative indices from the length
    # of the array (at run-time).  If indices do not start with a negative 
    # sign, they are assumed to be positive.
    #
    # This scheme doesn't work in the general case.  For example, if you
    # are calculating negative indices from a math expression that doesn't
    # start with the negative sign, then it will be assumed positive and
    # hence generate wrong results (and maybe a seg-fault).
    # 
    # I think this case can might be remedied by calculating all ranges on
    # the fly, and then subtracting them from the length of the array in 
    # that dimension if they are negative.  This is major code bloat in the
    # funcitons and more work.  Save till later...
    ###########################################################################
    # I don't think the strip is necessary, but it insures
    # that '-' is the first sign for negative indices.
    if slice_vars['single_index'] != '_index':
        expr = '%(single_index)s' % slice_vars        
    else:    
        begin = string.strip(slice_vars['begin'])
        if begin[0] == '-':
            slice_vars['begin'] = 'N' + slice_vars['var']+`position`+begin;
    
        end = string.strip(slice_vars['end'])
        if end != '_end' and end[0] != '-':
            #compensate for blitz using inclusive indexing on top end 
            #of slice for positive indices.
            slice_vars['end'] = end + '-1'
        if end[0] == '-':
            slice_vars['end'] = 'N%s[%d]%s-1' % (slice_vars['var'],position,end)
        
        if slice_vars['step'] == '_stp':
            # this if/then isn't strictly necessary, it'll
            # just keep the output code a little cleaner
            expr = 'slice(%(begin)s,%(end)s)' % slice_vars        
        else:        
            expr = 'slice(%(begin)s,%(end)s,%(step)s)' % slice_vars    
    val =  atom_list(expr)
    return val

def transform_subscript_list(subscript_dict):
    # this is gonna edit the ast_list...        
    subscript_list = subscript_dict['subscript_list']

    var = subscript_dict['var']
    #skip the first entry (the subscript_list symbol)
    slice_position = -1
    for i in range(1,len(subscript_list)):
        #skip commas...
        if subscript_list[i][0] != token.COMMA:
            slice_position += 1
            slice_vars = slice_ast_to_dict(subscript_list[i])

            slice_vars['var'] = var
            # create a slice(b,e,s) atom and insert in 
            # place of the x:y:z atom in the tree.            
            subscript_list[i] = build_slice_atom(slice_vars, slice_position)
        
def harvest_subscript_dicts(ast_list):
    """ Needs Tests!
    """
    subscript_lists = []
    if type(ast_list)  == ListType:
        found,data = match(indexed_array_pattern,ast_list)
        # data is a dict with 'var' = variable name
        # and 'subscript_list' = to the ast_seq for the subscript list
        if found:
            subscript_lists.append(data)
        for item in ast_list:
            if type(item) == ListType:
                 subscript_lists.extend(harvest_subscript_dicts(item))
    return subscript_lists

def transform_slices(ast_list):
    """ Walk through an ast_list converting all x:y:z subscripts
        to slice(x,y,z) subscripts.
    """
    all_dicts = harvest_subscript_dicts(ast_list)
    for subscript_dict in all_dicts:
        transform_subscript_list(subscript_dict)

slice_patterns = []
CLN = (token.COLON,':')
CLN2= (symbol.sliceop, (token.COLON, ':'))
CLN2_STEP = (symbol.sliceop, (token.COLON, ':'),['step'])
# [begin:end:step]
slice_patterns.append((symbol.subscript, ['begin'],CLN,['end'], CLN2_STEP ))
# [:end:step]
slice_patterns.append((symbol.subscript,           CLN,['end'], CLN2_STEP ))
# [begin::step]
slice_patterns.append((symbol.subscript, ['begin'],CLN,          CLN2_STEP ))
# [begin:end:]
slice_patterns.append((symbol.subscript, ['begin'],CLN,['end'], CLN2      ))
# [begin::]
slice_patterns.append((symbol.subscript, ['begin'],CLN,          CLN2      ))
# [:end:]
slice_patterns.append((symbol.subscript,           CLN,['end'], CLN2,     ))
# [::step]
slice_patterns.append((symbol.subscript,           CLN,          CLN2_STEP ))
# [::]
slice_patterns.append((symbol.subscript,           CLN,          CLN2      ))

# begin:end variants
slice_patterns.append((symbol.subscript, ['begin'],CLN,['end']))
slice_patterns.append((symbol.subscript,           CLN,['end']))
slice_patterns.append((symbol.subscript, ['begin'],CLN))
slice_patterns.append((symbol.subscript,           CLN))  

# a[0] variant -- can't believe I left this out...
slice_patterns.append((symbol.subscript,['single_index']))  

indexed_array_pattern = \
           (symbol.power,
             (symbol.atom,(token.NAME, ['var'])),
             (symbol.trailer,
                (token.LSQB, '['),
                   ['subscript_list'],
                (token.RSQB, ']')
             )
           )
