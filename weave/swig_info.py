import base_info

import os, swig_info
local_dir,junk = os.path.split(os.path.abspath(swig_info.__file__))   
f = open(os.path.join(local_dir,'swig','swigptr.c'))
swig_support_code = f.read()
f.close()

class swig_info(base_info.base_info):
    _support_code = [swig_support_code]