#!/usr/bin/env python
"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Currently, the following
classes are available:
  atlas_info
  blas_info
  lapack_info
  fftw_info
  x11_info

Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas'.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (or system_info could not find it).

Global parameters:
  system_info.search_static_first - search static libraries (.a)
             in precedence to shared ones (.so, .sl) if enabled.
  system_info.verbose - output the results to stdout if enabled.

The file 'site.cfg' in the same directory as this module is read
for configuration options. The format is that used by ConfigParser (i.e.,
Windows .INI style). The section DEFAULT has options that are the default
for each section. The available sections are fftw, atlas, and x11. Appropiate
defaults are used if nothing is specified.

The order of finding the locations of resources is the following:
 1. environment variable
 2. section in site.cfg
 3. DEFAULT section in site.cfg
Only the first complete match is returned.

Example:
----------
[DEFAULT]
library_dirs = /usr/lib:/usr/local/lib:/opt/lib
include_dirs = /usr/include:/usr/local/include:/opt/include
# search static libraries (.a) in preference to shared ones (.so)
search_static_first = 0

[fftw]
fftw_libs = fftw, rfftw
fftw_opt_libs = fftw_threaded, rfftw_threaded
# if the above aren't found, look for {s,d}fftw_libs and {s,d}fftw_opt_libs

[atlas]
library_dirs = /usr/lib/3dnow:/usr/lib/3dnow/atlas
# for overriding the names of the atlas libraries
atlas_libs = f77blas, cblas, atlas
lapack_libs = lapack

[x11]
library_dirs = /usr/X11R6/lib
include_dirs = /usr/X11R6/include
----------

Authors:
  Pearu Peterson <pearu@cens.ioc.ee>, February 2002
  David M. Cooke <cookedm@physics.mcmaster.ca>, April 2002

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the 
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

import sys,os,re,types
from distutils.errors import DistutilsError
from glob import glob
import ConfigParser

from distutils.sysconfig import get_config_vars

if sys.platform == 'win32':
    default_lib_dirs = ['C:\\'] # probably not very helpful...
    default_include_dirs = []
    default_x11_lib_dirs = []
    default_x11_include_dirs = []
else:
    default_lib_dirs = ['/usr/local/lib', '/opt/lib', '/usr/lib']
    default_include_dirs = ['/usr/local/include',
                            '/opt/include', '/usr/include']
    default_x11_lib_dirs = ['/usr/X11R6/lib','/usr/X11/lib']
    default_x11_include_dirs = ['/usr/X11R6/include','/usr/X11/include']

if os.path.join(sys.prefix, 'lib') not in default_lib_dirs:
    default_lib_dirs.insert(0,os.path.join(sys.prefix, 'lib'))
    default_include_dirs.append(os.path.join(sys.prefix, 'include'))

default_lib_dirs = filter(os.path.isdir, default_lib_dirs)
default_include_dirs = filter(os.path.isdir, default_include_dirs)

so_ext = get_config_vars('SO')[0] or ''

def get_info(name):
    cl = {'atlas':atlas_info,
          'x11':x11_info,
          'fftw':fftw_info,
          'blas':blas_info,
          'lapack':lapack_info,
          }.get(name.lower(),system_info)
    return cl().get_info()

class NotFoundError(DistutilsError):
    """Some third-party program or library is not found."""

class AtlasNotFoundError(NotFoundError):
    """
    Atlas (http://math-atlas.sourceforge.net/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file or by setting the ATLAS environment
    variable."""

class LapackNotFoundError(NotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file or by setting the LAPACK environment
    variable."""

class BlasNotFoundError(NotFoundError):
    """
    Blas (http://www.netlib.org/blas/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file or by setting the BLAS environment
    variable."""

class FFTWNotFoundError(NotFoundError):
    """
    FFTW (http://www.fftw.org/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file or by setting the FFTW environment
    variable."""

class F2pyNotFoundError(NotFoundError):
    """
    f2py2e (http://cens.ioc.ee/projects/f2py2e/) module not found.
    Get it from above location, install it, and retry setup.py."""

class NumericNotFoundError(NotFoundError):
    """
    Numeric (http://www.numpy.org/) module not found.
    Get it from above location, install it, and retry setup.py."""

class X11NotFoundError(NotFoundError):
    """X11 libraries not found."""

class system_info:

    """ get_info() is the only public method. Don't use others.
    """
    section = 'DEFAULT'
    dir_env_var = None
    search_static_first = 0 # XXX: disabled by default, may disappear in
                            # future unless it is proved to be useful.
    verbose = 1
    saved_results = {}

    def __init__ (self,
                  default_lib_dirs=default_lib_dirs,
                  default_include_dirs=default_include_dirs,
                  ):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {}
        defaults['library_dirs'] = os.pathsep.join(default_lib_dirs)
        defaults['include_dirs'] = os.pathsep.join(default_include_dirs)
        defaults['search_static_first'] = str(self.search_static_first)
        self.cp = ConfigParser.ConfigParser(defaults)
        cf = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                          'site.cfg')
        self.cp.read([cf])
        if not self.cp.has_section(self.section):
            self.cp.add_section(self.section)
        self.search_static_first = self.cp.getboolean(self.section,
                                                      'search_static_first')
        assert type(self.search_static_first) is type(0)

    def set_info(self,**info):
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
            if hasattr(self, 'calc_info'):
                self.calc_info()
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

    def get_paths(self, section, key):
        dirs = self.cp.get(section, key).split(os.pathsep)
        if os.environ.has_key(self.dir_env_var):
            dirs = os.environ[self.dir_env_var].split(os.pathsep) + dirs
        default_dirs = self.cp.get('DEFAULT', key).split(os.pathsep)
        dirs.extend(default_dirs)
        ret = []
        [ret.append(d) for d in dirs if os.path.isdir(d) and d not in ret]
        return ret

    def get_lib_dirs(self, key='library_dirs'):
        return self.get_paths(self.section, key)

    def get_include_dirs(self, key='include_dirs'):
        return self.get_paths(self.section, key)

    def get_libs(self, key, default):
        try:
            libs = self.cp.get(self.section, key)
        except ConfigParser.NoOptionError:
            return default
        return [a.strip() for a in libs.split(',')]

    def check_libs(self,lib_dir,libs,opt_libs =[]):
        """ If static or shared libraries are available then return
            their info dictionary. """
        if self.search_static_first:
            exts = ['.a',so_ext]
        else:
            exts = [so_ext,'.a']
        for ext in exts:
            info = self._check_libs(lib_dir,libs,opt_libs,ext)
            if info is not None: return info

    def _lib_list(self, lib_dir, libs, ext):
        assert type(lib_dir) is type('')
        liblist = []
        for l in libs:
            p = combine_paths(lib_dir, 'lib'+l+ext)
            if p:
                assert len(p)==1
                liblist.append(p[0])
        return liblist

    def _extract_lib_names(self,libs):
        return [os.path.splitext(os.path.basename(p))[0][3:] \
                for p in libs]

    def _check_libs(self,lib_dir,libs, opt_libs, ext):
        found_libs = self._lib_list(lib_dir, libs, ext)
        if len(found_libs) == len(libs):
            found_libs = self._extract_lib_names(found_libs)
            info = {'libraries' : found_libs, 'library_dirs' : [lib_dir]}
            opt_found_libs = self._lib_list(lib_dir, opt_libs, ext)
            if len(opt_found_libs) == len(opt_libs):
                opt_found_libs = self._extract_lib_names(opt_found_libs)
                info['libraries'].extend(opt_found_libs)
            return info


class fftw_info(system_info):
    section = 'fftw'
    dir_env_var = 'FFTW'

    def __init__(self):
        system_info.__init__(self)

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        incl_dir = None

        libs = self.get_libs('fftw_libs', ['fftw','rfftw'])
        opt_libs = self.get_libs('fftw_opt_libs',
                                 ['fftw_threads','rfftw_threads'])
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
            libs = self.get_libs('dfftw_libs', ['dfftw', 'drfftw'])
            opt_libs = self.get_libs('dfftw_opt_libs',
                                     ['dfftw_threads', 'drfftw_threads'])
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

        libs = self.get_libs('sfftw_libs', ['sfftw', 'srfftw'])
        opt_libs = self.get_libs('sfftw_opt_libs',
                                 ['sfftw_threads', 'srfftw_threads'])
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
    section = 'atlas'
    dir_env_var = 'ATLAS'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + combine_paths(d,['atlas*','ATLAS*']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        include_dirs = self.get_include_dirs()

        h = (combine_paths(lib_dirs+include_dirs,'cblas.h') or [None])[0]
        if h: h = os.path.dirname(h)
        info = None
        # lapack must appear before atlas
        lapack_libs = self.get_libs('lapack_libs', ['lapack'])
        for d in lib_dirs:
            lapack = self.check_libs(d,lapack_libs,[])
            if lapack is not None:
                info = lapack                
                break
        else:
            return
        atlas_libs = self.get_libs('atlas_libs', ['f77blas', 'cblas', 'atlas'])
        for d in lib_dirs:
            atlas = self.check_libs(d,atlas_libs,[])
            if atlas is not None:
                dict_append(info, **atlas)
                break
        else:
            return

        if h: dict_append(info,include_dirs=[h])
        self.set_info(**info)

class lapack_info(system_info):
    section = 'lapack'
    dir_env_var = 'LAPACK'

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()

        lapack_libs = self.get_libs('lapack_libs', ['lapack'])
        for d in lib_dirs:
            lapack = self.check_libs(d,lapack_libs,[])
            if lapack is not None:
                info = lapack                
                break
        else:
            return
        self.set_info(**info)

class blas_info(system_info):
    section = 'blas'
    dir_env_var = 'BLAS'

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()

        blas_libs = self.get_libs('blas_libs', ['blas'])
        for d in lib_dirs:
            blas = self.check_libs(d,blas_libs,[])
            if blas is not None:
                info = blas                
                break
        else:
            return
        self.set_info(**info)


class x11_info(system_info):
    section = 'x11'

    def __init__(self):
        system_info.__init__(self,
                             default_lib_dirs=default_x11_lib_dirs,
                             default_include_dirs=default_x11_include_dirs)

    def calc_info(self):
        if sys.platform  == 'win32':
            return
        lib_dirs = self.get_lib_dirs()
        include_dirs = self.get_include_dirs()
        x11_libs = self.get_libs('x11_libs', ['X11'])
        for lib_dir in lib_dirs:
            info = self.check_libs(lib_dir, x11_libs, [])
            if info is not None:
                break
        else:
            return
        inc_dir = None
        for d in include_dirs:
            if combine_paths(d, 'X11/X.h'):
                inc_dir = d
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir])
        self.set_info(**info)

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
            if k in ['library_dirs','include_dirs','define_macros']:
                [d[k].append(vv) for vv in v if vv not in d[k]]
            else:
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
