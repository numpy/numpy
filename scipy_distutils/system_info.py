#!/usr/bin/env python
"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Currently, the following
classes are available:
  atlas_info
  fftw_info
  x11_info
The following environment variables are used if defined:
  ATLAS - path to ATLAS library
  FFTW  - path to FFTW library

Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw'.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (or system_info could not find it).

Global parameters:
  prefixes - a list of prefixes for scanning the location of
             resources.
  system_info.static_first - a flag for indicating that static
             libraries are searched first than shared ones.
  system_info.verbose - show the results if set.

Author:
  Pearu Peterson <pearu@cens.ioc.ee>, February 2002
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

import sys,os,re,types,pprint
from distutils.errors import DistutilsError
from glob import glob

from distutils.sysconfig import get_config_vars

if sys.platform == 'win32':
    prefixes = ['C:\\'] # XXX: what is prefix in win32?
else:
    prefixes = ['/usr','/usr/local','/opt']
if sys.prefix not in prefixes:
    prefixes.append(sys.prefix)
prefixes = filter(os.path.isdir,prefixes) # XXX: Is this ok on win32? Is 'C:' dir?

so_ext = get_config_vars('SO')[0] or ''

def get_info(name):
    cl = {'atlas':atlas_info,
          'x11':x11_info,
          'fftw':fftw_info}.get(name.lower(),system_info)
    return cl().get_info()

class NotFoundError(DistutilsError):
    """Some third-party program or library is not found."""

class AtlasNotFoundError(NotFoundError):
    """
    Atlas (http://math-atlas.sourceforge.net/) libraries not found.
    Either install them in /usr/local/lib/atlas or /usr/lib/atlas
    and retry setup.py. One can use also ATLAS environment variable
    to indicate the location of Atlas libraries."""

class FFTWNotFoundError(NotFoundError):
    """
    FFTW (http://www.fftw.org/) libraries not found.
    Either install them in /usr/local/lib or /usr/lib and retry setup.py.
    One can use also FFTW environment variable to indicate
    the location of FFTW libraries."""

class F2pyNotFoundError(NotFoundError):
    """
    f2py2e (http://cens.ioc.ee/projects/f2py2e/) module not found.
    Get it from above location, install it, and retry setup.py."""

class NumericNotFoundError(NotFoundError):
    """
    Numeric (http://pfdubois.com/numpy/) module not found.
    Get it from above location, install it, and retry setup.py."""

class X11NotFoundError(NotFoundError):
    """X11 libraries not found."""

class system_info:

    """ get_info() is the only public method. Don't use others.
    """

    static_first = 1
    verbose = 1
    need_refresh = 1
    saved_results = {}
    
    def __init__ (self):
        self.__class__.info = {}
        #self.__class__.need_refresh = not self.info
        self.local_prefixes = []

    def set_info(self,**info):
        #self.__class__.info = info
        self.saved_results[self.__class__.__name__] = info
    def has_info(self):
        return self.saved_results.has_key(self.__class__.__name__)
    def get_info(self):
        """ Return a dictonary with items that are compatible
            with scipy_distutils.setup keyword arguments.
        """
        flag = 0
        if not self.has_info():
            flag = 1
            if self.verbose:
                print self.__class__.__name__ + ':'
            for p in self.local_prefixes + prefixes:
                if self.verbose:
                    print '  Looking in',p,'...'
                self.calc_info(p)
                if self.has_info(): break
            if self.verbose:
                if not self.has_info():
                    print '  NOT AVAILABLE'
                    self.set_info()
                else:
                    print '  FOUND:'
        res = self.saved_results.get(self.__class__.__name__)
        if self.verbose and flag:
            for k,v in res.items():
                print '    %s = %s'%(k,v)
            print
        return res

    def calc_info(self,prefix):
        """ Calculate info distionary. """

    def check_libs(self,lib_dir,libs,opt_libs =[]):
        """ If static or shared libraries are available then return
            their info dictionary. """
        mths = [self.check_static_libs,self.check_shared_libs]
        if not self.static_first:
            mths.reverse() # if one prefers shared libraries
        for m in mths:
            info = m(lib_dir,libs,opt_libs)
            if info is not None: return info

    def check_static_libs(self,lib_dir,libs,opt_libs =[]):
        #XXX: what are .lib and .dll files under win32?
        if len(combine_paths(lib_dir,['lib'+l+'.a' for l in libs])) == len(libs):
            info = {'libraries':libs,'library_dirs':[lib_dir]}
            if len(combine_paths(lib_dir,['lib'+l+'.a' for l in libs]))\
               ==len(opt_libs):
                info['libraries'].extend(opt_libs)
            return info

    def check_shared_libs(self,lib_dir,libs,opt_libs =[]):
        shared_libs = []
        for l in libs:
            p = shortest_path(combine_paths(lib_dir,'lib'+l+so_ext+'*'))
            if p is not None: shared_libs.append(p)
        if len(shared_libs) == len(libs):
            info = {'extra_objects':shared_libs}
            opt_shared_libs = []
            for l in opt_libs:
                p = shortest_path(combine_paths(lib_dir,'lib'+l+so_ext+'*'))
                if p is not None: opt_shared_libs.append(p)
            info['extra_objects'].extend(opt_shared_libs)
            return info


class fftw_info(system_info):

    def __init__(self):
        system_info.__init__(self)
        p = os.environ.get('FFTW')
        if p is not None:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                self.local_prefixes.insert(0,p)

    def calc_info(self,prefix):       
        lib_dirs = filter(os.path.isdir,
                          combine_paths(prefix,'lib',['fftw*','FFTW*']))
        if not lib_dirs:
            lib_dirs = filter(os.path.isdir,
                              combine_paths(prefix,['fftw*','FFTW*'],'lib'))
        if not lib_dirs:
            lib_dirs = filter(os.path.isdir,
                              combine_paths(prefix,['lib','fftw*','FFTW*']))
                                                          
        if not lib_dirs:
            lib_dirs = [prefix]
        incl_dirs = filter(os.path.isdir,
                           combine_paths(prefix,'include',['fftw*','FFTW*']))
        if not incl_dirs:
            incl_dirs = filter(os.path.isdir,
                               combine_paths(prefix,['fftw*','FFTW*'],'include'))
        if not incl_dirs:
            incl_dirs = filter(os.path.isdir,
                               combine_paths(prefix,['include','fftw*','FFTW*']))
        if not incl_dirs:
            incl_dirs = [prefix]
        incl_dir = None

        libs = ['fftw','rfftw']
        opt_libs = ['fftw_threads','rfftw_threads']
        info = None
        for d in lib_dirs:
            r = self.check_libs(d,libs,opt_libs)
            if r is not None:
                info = r
                break
        if info is not None:
            flag = 0
            for d in incl_dirs:
                if len(combine_paths(d,['fftw.h','rfftw.h']))==2:
                    dict_append(info,include_dirs=[d])
                    flag = 1
                    incl_dirs = [d]
                    incl_dir = d
                    break
            if flag:
                dict_append(info,define_macros=[('SCIPY_FFTW_H',1)])
            else:
                info = None

        if info is None:
            libs = ['dfftw','drfftw']
            opt_libs = ['dfftw_threads','drfftw_threads']
            for d in lib_dirs:
                r = self.check_libs(d,libs,opt_libs)
                if r is not None:
                    info = r
                    break
            if info is not None:
                flag = 0
                for d in incl_dirs:
                    if len(combine_paths(d,['dfftw.h','drfftw.h']))==2:
                        if incl_dir is None:
                            dict_append(info,include_dirs=[d])
                            incl_dirs = [d]
                            incl_dir = d
                        flag = 1
                        break
                if flag:
                    dict_append(info,define_macros=[('SCIPY_DFFTW_H',1)])
                else:
                    info = None
        
        libs = ['sfftw','srfftw']
        opt_libs = ['sfftw_threads','srfftw_threads']
        flag = 0
        for d in lib_dirs:
            r = self.check_libs(d,libs,opt_libs)
            if r is not None:
                if info is None: info = r
                else: dict_append(info,**r)
                flag = 1
                break
        if info is not None and flag:
            for d in incl_dirs:
                if len(combine_paths(d,['sfftw.h','srfftw.h']))==2:
                    if incl_dir is None:
                        dict_append(info,include_dirs=[d])
                    dict_append(info,define_macros=[('SCIPY_SFFTW_H',1)])
                    break
        if info is not None:
            self.set_info(**info)


class atlas_info(system_info):

    def __init__(self):
        system_info.__init__(self)
        p = os.environ.get('ATLAS')
        if p is not None:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                self.local_prefixes.insert(0,p)

    def calc_info(self, prefix):
        print combine_paths(prefix,'lib',['atlas*','ATLAS*'])
        lib_dirs = filter(os.path.isdir,combine_paths(prefix,'lib',
                                                      ['atlas*','ATLAS*']))
        if lib_dirs:
            other_dirs = filter(os.path.isdir,combine_paths(lib_dirs,'*'))
            other_dirs.extend(filter(os.path.isdir,combine_paths(prefix,'lib')))
            lib_dirs.extend(other_dirs)
        else:
            lib_dirs = filter(os.path.isdir,
                              combine_paths(prefix,['lib','atlas*','ATLAS*']))
        if not lib_dirs:
            lib_dirs = [prefix]

        h = (combine_paths(lib_dirs,'cblas.h') or [None])[0]
        if not h:
            h = (combine_paths(lib_dirs,'include','cblas.h') or [None])[0]
        if h: h = os.path.dirname(h)

        libs = ['lapack','f77blas','cblas','atlas']
        info = None
        for d in lib_dirs:
            r = self.check_libs(d,libs,[])
            if r is not None:
                info = r
                break
        if info is None: return
        if h: dict_append(info,include_dirs=[h])
        self.set_info(**info)


class blas_info(system_info):
    # For Fortran or optimized blas, not atlas.
    pass


class lapack_info(system_info):
    # For Fortran or optimized lapack, not atlas
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
            info = self.check_libs(lib_dir[0],['X11'],[])
            if info is None:
                continue
            dict_append(info,include_dirs=[inc_dir])
            self.set_info(**info)

def shortest_path(pths):
    pths.sort()
    if pths: return pths[0]

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
    if len(args)==1:
        result = reduce(lambda a,b:a+b,map(glob,args[0]),[])
    elif len (args)==2:
        result = []
        for a0 in args[0]:
            for a1 in args[1]:
                result.extend(glob(os.path.join(a0,a1)))
    else:
        result = combine_paths(*(combine_paths(args[0],args[1])+args[2:]))
    return result

def dict_append(d,**kws):
    for k,v in kws.items():
        if d.has_key(k):
            d[k].extend(v)
        else:
            d[k] = v

def show_all():
    import system_info
    import pprint
    match_info = re.compile(r'.*?_info').match
    for n in filter(match_info,dir(system_info)):
        if n in ['system_info','get_info']: continue
        c = getattr(system_info,n)()
        r = c.get_info()

if __name__ == "__main__":
    show_all()
