
import sys, os, glob

def get_x11_info():
    if sys.platform  == 'win32':
        return {}

    #XXX: add other combinations if needed
    libs = ['X11']
    prefixes = ['/usr','/usr/local','/opt']
    x11_names = ['X11R6','X11']
    header_files = ['X.h','X11/X.h']

    x11_lib,x11_lib_dir,x11_inc_dir = None,None,None
    for p in prefixes:
        for n in x11_names:
            d = os.path.join(p,n)
            if not os.path.exists(d): continue
            if x11_lib is None:
                # Find library and its location
                for l in libs:
                    if glob.glob(os.path.join(d,'lib','lib%s.*' % l)):
                        x11_lib = l
                        x11_lib_dir = os.path.join(d,'lib')
                        break
            if x11_inc_dir is None:
                # Find the location of header file
                for h in header_files:
                    if os.path.exists(os.path.join(d,'include',h)):
                        x11_inc_dir = os.path.join(d,'include')
                        break
        if None not in [x11_lib,x11_inc_dir]:
            break
    if None in [x11_lib,x11_inc_dir]:
        return {}
    info = {}
    info['libraries'] = [x11_lib]
    info['library_dirs'] = [x11_lib_dir]
    info['include_dirs'] = [x11_inc_dir]
    return info

