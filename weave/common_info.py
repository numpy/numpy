""" Generic support code for handling standard Numeric arrays      
"""

import base_info

#############################################################
# Basic module support code
#############################################################

from conversion_code import module_support_code
from conversion_code import file_convert_code
from conversion_code import instance_convert_code
from conversion_code import callable_convert_code
from conversion_code import module_convert_code
from conversion_code import scalar_support_code
from conversion_code import non_template_scalar_support_code

class basic_module_info(base_info.base_info):
    _headers = ['"Python.h"']
    _support_code = [module_support_code]

class file_info(base_info.base_info):
    _headers = ['<stdio.h>']
    _support_code = [file_convert_code]

class instance_info(base_info.base_info):
    _support_code = [instance_convert_code]

class callable_info(base_info.base_info):
    _support_code = [callable_convert_code]

class module_info(base_info.base_info):
    _support_code = [module_convert_code]

class scalar_info(base_info.base_info):
    _warnings = ['disable: 4275', 'disable: 4101']
    _headers = ['<complex>','<math.h>']
    def support_code(self):
        if self.compiler != 'msvc':
             # maybe this should only be for gcc...
            return [scalar_support_code,non_template_scalar_support_code]
        else:
            return [non_template_scalar_support_code]
