""" support code and other things needed to compile support
    for numeric expressions in python.
    
    There are two sets of support code, one with templated
    functions and one without.  This is because msvc cannot
    handle the templated functions.  We need the templated
    versions for more complex support of numeric arrays with
    blitz. 
"""

import base_info

from conversion_code import scalar_support_code
#from conversion_code import non_template_scalar_support_code

class scalar_info(base_info.base_info):
    _warnings = ['disable: 4275', 'disable: 4101']
    _headers = ['<complex>','<math.h>']
    def support_code(self):
        return [scalar_support_code]
        # REMOVED WHEN TEMPLATE CODE REMOVED
        #if self.compiler != 'msvc':
        #     # maybe this should only be for gcc...
        #    return [scalar_support_code,non_template_scalar_support_code]
        #else:
        #    return [non_template_scalar_support_code]
            