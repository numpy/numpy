#!/usr/bin/env python

import sys,os,re,types
from glob import glob

class system_info:

    if sys.platform == 'win32':
        prefixes = [] # XXX: what is prefix in win32? C: maybe?
    else:
        prefixes = ['/usr','/usr/local','/opt']
    if sys.prefix not in prefixes:
        prefixes.append(sys.prefix)
    prefixes = filter(os.path.isdir,prefixes)

    def __init__ (self):
        self.info = {}
        self.need_refresh = 1
    def get_info(self):
        """ Return a dictonary with items that are compatible
            with scipy_distutils.setup keyword arguments.
        """
        if self.need_refresh:
            for p in self.prefixes:
                r = self.calc_info(p)
                if r: break
            self.need_refresh = 0
        return self.info
    def calc_info(self,prefix):
        """ Calculate info distionary. """

class fftw_info(system_info):
    pass

class atlas_info(system_info):
    pass

class x11_info(system_info):

    def calc_info(self, prefix):
        if sys.platform  == 'win32':
            return
        for x11_dir in combine_paths(prefix,['X11R6','X11']):
            inc_dir = None
            for d in combine_paths(x11_dir,['include','include/X11']):
                if combine_paths(d,'X.h'):
                    inc_dir = d
                    break
            if not d: return
            lib_dir = combine_paths(x11_dir,'lib')
            if not lib_dir: return
            info = {}
            if combine_paths(lib_dir,'libX11.a'):
                info['library_dirs'] = lib_dir
                info['libraries'] = ['X11']
            else:
                shared_libs = combine_paths(lib_dir,'libX11.so*')
                if not shared_libs: return
                info['extra_linker_arguments'] = [shared_libs[0]]
            info['include_dirs'] = [inc_dir]
            self.info = info
            return 1

def combine_paths(*args):
    """ Return a list of existing paths composed by all combinations of
        items from arguments.
    """
    r = []
    for a in args:
        if not a: continue
        if type(a) is types.StringType:
            a = [a]
        r.append(a)
    args = r
    if not args: return []
    if len (args)==1: return map(glob,args[0])
    if len (args)==2:
        result = []
        for a0 in args[0]:
            for a1 in args[1]:
                result.extend(glob(os.path.join(a0,a1)))
        return result
    return combine_paths(*(combine_paths(args[0],args[1])+args[2:]))

def show_all():
    import system_info
    match_info = re.compile(r'.*?_info').match
    for n in filter(match_info,dir(system_info)):
        if n=='system_info': continue
        c = getattr(system_info,n)()
        print '------------>',n
        print c.get_info()

if __name__ == "__main__":
    show_all()
