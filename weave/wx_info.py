import swig_info

# THIS IS PLATFORM DEPENDENT FOR NOW. 
# YOU HAVE TO SPECIFY YOUR WXWINDOWS DIRECTORY

wx_dir = 'C:\\wx232\\include'

class wx_info(swig_info.swig_info):
    _headers = ['"wx/wx.h"']
    _include_dirs = [wx_dir]
    _define_macros=[('wxUSE_GUI', '1')]
    #_libraries = ['wx23_1']
    #_libraries = ['wxmswd']
    #_library_dirs = ['c:/wx232/lib']