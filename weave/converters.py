""" converters.py
"""


import base_spec
import scalar_spec
import sequence_spec
import common_spec


#--------------------------------------------------------
# The "standard" conversion classes
#--------------------------------------------------------

default = [scalar_spec.int_converter(),
           scalar_spec.float_converter(),
           scalar_spec.complex_converter(),
           sequence_spec.string_converter(),
           sequence_spec.list_converter(),
           sequence_spec.dict_converter(),
           sequence_spec.tuple_converter(),
           common_spec.file_converter(),
           common_spec.callable_converter(),
           common_spec.instance_converter(),]                          
          #common_spec.module_converter()]

try: 
    import standard_array_spec
    default.append(standard_array_spec.array_converter())
except: 
    pass    

try: 
    # this is currently safe because it doesn't import wxPython.
    import wx_spec
    default.append(wx_spec.wx_converter())
except: 
    pass    

#--------------------------------------------------------
# Blitz conversion classes
#
# same as default, but will convert Numeric arrays to blitz
# C++ classes 
#--------------------------------------------------------
import blitz_spec
blitz = [blitz_spec.array_converter()] + default

