#!/usr/bin/env python
"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Currently, the following
classes are available:
  atlas_info
  blas_info
  lapack_info
  fftw_info,dfftw_info,sfftw_info
  fftw_threads_info,dfftw_threads_info,sfftw_threads_info
  djbfft_info
  x11_info
  lapack_src_info
  blas_src_info

Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas',
  'lapack_src', or 'blas_src'.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (system_info could not find it).

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
src_dirs = /usr/local/src:/opt/src
# search static libraries (.a) in preference to shared ones (.so)
search_static_first = 0

[fftw]
fftw_libs = fftw, rfftw
fftw_opt_libs = fftw_threaded, rfftw_threaded
# if the above aren't found, look for {s,d}fftw_libs and {s,d}fftw_opt_libs

[atlas]
library_dirs = /usr/lib/3dnow:/usr/lib/3dnow/atlas
# for overriding the names of the atlas libraries
atlas_libs = lapack, f77blas, cblas, atlas

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
    default_src_dirs = []
    default_x11_lib_dirs = []
    default_x11_include_dirs = []
else:
    default_lib_dirs = ['/usr/local/lib', '/opt/lib', '/usr/lib']
    default_include_dirs = ['/usr/local/include',
                            '/opt/include', '/usr/include']
    default_src_dirs = ['/usr/local/src', '/opt/src']
    default_x11_lib_dirs = ['/usr/X11R6/lib','/usr/X11/lib']
    default_x11_include_dirs = ['/usr/X11R6/include','/usr/X11/include']

if os.path.join(sys.prefix, 'lib') not in default_lib_dirs:
    default_lib_dirs.insert(0,os.path.join(sys.prefix, 'lib'))
    default_include_dirs.append(os.path.join(sys.prefix, 'include'))
    default_src_dirs.append(os.path.join(sys.prefix, 'src'))

default_lib_dirs = filter(os.path.isdir, default_lib_dirs)
default_include_dirs = filter(os.path.isdir, default_include_dirs)
default_src_dirs = filter(os.path.isdir, default_src_dirs)

so_ext = get_config_vars('SO')[0] or ''

def get_info(name):
    cl = {'atlas':atlas_info,
          'x11':x11_info,
          'fftw':fftw_info,
          'dfftw':dfftw_info,
          'sfftw':sfftw_info,
          'fftw_threads':fftw_threads_info,
          'dfftw_threads':dfftw_threads_info,
          'sfftw_threads':sfftw_threads_info,
          'djbfft':djbfft_info,
          'blas':blas_info,
          'lapack':lapack_info,
          'lapack_src':lapack_src_info,
          'blas_src':blas_src_info,
          }.get(name.lower(),system_info)
    return cl().get_info()

class NotFoundError(DistutilsError):
    """Some third-party program or library is not found."""

class AtlasNotFoundError(NotFoundError):
    """
    Atlas (http://math-atlas.sourceforge.net/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [atlas]) or by setting
    the ATLAS environment variable."""

class LapackNotFoundError(NotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [lapack]) or by setting
    the LAPACK environment variable."""

class LapackSrcNotFoundError(LapackNotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) sources not found.
    Directories to search for the sources can be specified in the
    scipy_distutils/site.cfg file (section [lapack_src]) or by setting
    the LAPACK_SRC environment variable."""

class BlasNotFoundError(NotFoundError):
    """
    Blas (http://www.netlib.org/blas/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [blas]) or by setting
    the BLAS environment variable."""

class BlasSrcNotFoundError(BlasNotFoundError):
    """
    Blas (http://www.netlib.org/blas/) sources not found.
    Directories to search for the sources can be specified in the
    scipy_distutils/site.cfg file (section [blas_src]) or by setting
    the BLAS_SRC environment variable."""

class FFTWNotFoundError(NotFoundError):
    """
    FFTW (http://www.fftw.org/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [fftw]) or by setting
    the FFTW environment variable."""

class DJBFFTNotFoundError(NotFoundError):
    """
    DJBFFT (http://cr.yp.to/djbfft.html) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [djbfft]) or by setting
    the DJBFFT environment variable."""

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
        defaults['src_dirs'] = os.pathsep.join(default_src_dirs)
        defaults['search_static_first'] = str(self.search_static_first)
        self.cp = ConfigParser.ConfigParser(defaults)
        cf = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                          'site.cfg')
        self.cp.read([cf])
        if not self.cp.has_section(self.section):
            self.cp.add_section(self.section)
        self.search_static_first = self.cp.getboolean(self.section,
                                                      'search_static_first')
        assert isinstance(self.search_static_first, int)

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
                v = str(v)
                if k=='sources' and len(v)>200: v = v[:60]+' ...\n... '+v[-60:]
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

    def get_src_dirs(self, key='src_dirs'):
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
    libs = ['fftw','rfftw']
    includes = ['fftw.h','rfftw.h']
    macros = [('SCIPY_FFTW_H',None)]

    def __init__(self):
        system_info.__init__(self)

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        incl_dir = None
        libs = self.get_libs(self.section+'_libs', self.libs)
        info = None
        for d in lib_dirs:
            r = self.check_libs(d,libs)
            if r is not None:
                info = r
                break
        if info is not None:
            flag = 0
            for d in incl_dirs:
                if len(combine_paths(d,self.includes))==2:
                    dict_append(info,include_dirs=[d])
                    flag = 1
                    incl_dirs = [d]
                    incl_dir = d
                    break
            if flag:
                dict_append(info,define_macros=self.macros)
            else:
                info = None
        if info is not None:
            self.set_info(**info)

class dfftw_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['drfftw','dfftw']
    includes = ['dfftw.h','drfftw.h']
    macros = [('SCIPY_DFFTW_H',None)]

class sfftw_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['srfftw','sfftw']
    includes = ['sfftw.h','srfftw.h']
    macros = [('SCIPY_SFFTW_H',None)]

class fftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['rfftw_threads','fftw_threads']
    includes = ['fftw_threads.h','rfftw_threads.h']
    macros = [('SCIPY_FFTW_THREADS_H',None)]

class dfftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['drfftw_threads','dfftw_threads']
    includes = ['dfftw_threads.h','drfftw_threads.h']
    macros = [('SCIPY_DFFTW_THREADS_H',None)]

class sfftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['srfftw_threads','sfftw_threads']
    includes = ['sfftw_threads.h','srfftw_threads.h']
    macros = [('SCIPY_SFFTW_THREADS_H',None)]

class djbfft_info(system_info):
    section = 'djbfft'
    dir_env_var = 'DJBFFTW'

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        info = None
        for d in lib_dirs:
            p = combine_paths (d,['djbfft.a'])
            if p:
                info = {'extra_objects':p}
                break
        if info is None:
            return
        for d in incl_dirs:
            if len(combine_paths(d,['fftc8.h','fftfreq.h']))==2:
                dict_append(info,include_dirs=[d],
                            define_macros=[('SCIPY_DJBFFT_H',None)])
                self.set_info(**info)
                return


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

        atlas_libs = self.get_libs('atlas_libs',
                                   ['lapack','f77blas', 'cblas', 'atlas'])
        for d in lib_dirs:
            atlas = self.check_libs(d,atlas_libs,[])
            if atlas is not None:
                info = atlas
                break
        else:
            return

        if h: dict_append(info,include_dirs=[h])

        # Check if lapack library is complete, only warn if it is not.
        lapack_dir = info['library_dirs'][0]
        lapack_name = info['libraries'][0]
        lapack_lib = None
        for e in ['.a',so_ext]:
            fn = os.path.join(lapack_dir,'lib'+lapack_name+e)
            if os.path.exists(fn):
                lapack_lib = fn
                break
        if lapack_lib is not None:
            fd = os.open(lapack_lib,os.O_RDONLY)
            sz = os.fstat(fd)[6]
            os.close(fd)
            import warnings
            if sz <= 4000*1024:
                message = """
*********************************************************************
    Lapack library (from ATLAS) is probably incomplete:
      size of %s is %sk (expected >4000k)

    Follow the instructions in the KNOWN PROBLEMS section of the file
    scipy/INSTALL.txt.
*********************************************************************
""" % (lapack_lib,sz/1024)
                warnings.warn(message)
            #info['size_liblapack'] = sz
            flag = 0 #os.system('nm %s | grep clapack_sgetri '%lapack_lib)
            #info['has_clapack_sgetri'] = not flag
            if flag:
                message = """
*********************************************************************
    Using probably old ATLAS version (<3.3.??):
      nm %s | grep clapack_sgetri
    returned %s
    ATLAS update is recommended. See scipy/INSTALL.txt.
*********************************************************************
""" % (lapack_lib,flag)
                warnings.warn(message)
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

class lapack_src_info(system_info):
    section = 'lapack_src'
    dir_env_var = 'LAPACK_SRC'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + combine_paths(d,['LAPACK*/SRC','SRC']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d,'dgesv.f')):
                src_dir = d
                break
        if not src_dir:
            #XXX: Get sources from netlib. May be ask first.
            return
        # The following is extracted from LAPACK-3.0/SRC/Makefile
        allaux='''
        ilaenv ieeeck lsame lsamen xerbla
        ''' # *.f
        laux = '''
        bdsdc bdsqr disna labad lacpy ladiv lae2 laebz laed0 laed1
        laed2 laed3 laed4 laed5 laed6 laed7 laed8 laed9 laeda laev2
        lagtf lagts lamch lamrg lanst lapy2 lapy3 larnv larrb larre
        larrf lartg laruv las2 lascl lasd0 lasd1 lasd2 lasd3 lasd4
        lasd5 lasd6 lasd7 lasd8 lasd9 lasda lasdq lasdt laset lasq1
        lasq2 lasq3 lasq4 lasq5 lasq6 lasr lasrt lassq lasv2 pttrf
        stebz stedc steqr sterf
        ''' # [s|d]*.f
        lasrc = '''
        gbbrd gbcon gbequ gbrfs gbsv gbsvx gbtf2 gbtrf gbtrs gebak
        gebal gebd2 gebrd gecon geequ gees geesx geev geevx gegs gegv
        gehd2 gehrd gelq2 gelqf gels gelsd gelss gelsx gelsy geql2
        geqlf geqp3 geqpf geqr2 geqrf gerfs gerq2 gerqf gesc2 gesdd
        gesv gesvd gesvx getc2 getf2 getrf getri getrs ggbak ggbal
        gges ggesx ggev ggevx ggglm gghrd gglse ggqrf ggrqf ggsvd
        ggsvp gtcon gtrfs gtsv gtsvx gttrf gttrs gtts2 hgeqz hsein
        hseqr labrd lacon laein lags2 lagtm lahqr lahrd laic1 lals0
        lalsa lalsd langb lange langt lanhs lansb lansp lansy lantb
        lantp lantr lapll lapmt laqgb laqge laqp2 laqps laqsb laqsp
        laqsy lar1v lar2v larf larfb larfg larft larfx largv larrv
        lartv larz larzb larzt laswp lasyf latbs latdf latps latrd
        latrs latrz latzm lauu2 lauum pbcon pbequ pbrfs pbstf pbsv
        pbsvx pbtf2 pbtrf pbtrs pocon poequ porfs posv posvx potf2
        potrf potri potrs ppcon ppequ pprfs ppsv ppsvx pptrf pptri
        pptrs ptcon pteqr ptrfs ptsv ptsvx pttrs ptts2 spcon sprfs
        spsv spsvx sptrf sptri sptrs stegr stein sycon syrfs sysv
        sysvx sytf2 sytrf sytri sytrs tbcon tbrfs tbtrs tgevc tgex2
        tgexc tgsen tgsja tgsna tgsy2 tgsyl tpcon tprfs tptri tptrs
        trcon trevc trexc trrfs trsen trsna trsyl trti2 trtri trtrs
        tzrqf tzrzf
        ''' # [s|c|d|z]*.f
        sd_lasrc = '''
        laexc lag2 lagv2 laln2 lanv2 laqtr lasy2 opgtr opmtr org2l
        org2r orgbr orghr orgl2 orglq orgql orgqr orgr2 orgrq orgtr
        orm2l orm2r ormbr ormhr orml2 ormlq ormql ormqr ormr2 ormr3
        ormrq ormrz ormtr rscl sbev sbevd sbevx sbgst sbgv sbgvd sbgvx
        sbtrd spev spevd spevx spgst spgv spgvd spgvx sptrd stev stevd
        stevr stevx syev syevd syevr syevx sygs2 sygst sygv sygvd
        sygvx sytd2 sytrd
        ''' # [s|d]*.f
        cz_lasrc = '''
        bdsqr hbev hbevd hbevx hbgst hbgv hbgvd hbgvx hbtrd hecon heev
        heevd heevr heevx hegs2 hegst hegv hegvd hegvx herfs hesv
        hesvx hetd2 hetf2 hetrd hetrf hetri hetrs hpcon hpev hpevd
        hpevx hpgst hpgv hpgvd hpgvx hprfs hpsv hpsvx hptrd hptrf
        hptri hptrs lacgv lacp2 lacpy lacrm lacrt ladiv laed0 laed7
        laed8 laesy laev2 lahef lanhb lanhe lanhp lanht laqhb laqhe
        laqhp larcm larnv lartg lascl laset lasr lassq pttrf rot spmv
        spr stedc steqr symv syr ung2l ung2r ungbr unghr ungl2 unglq
        ungql ungqr ungr2 ungrq ungtr unm2l unm2r unmbr unmhr unml2
        unmlq unmql unmqr unmr2 unmr3 unmrq unmrz unmtr upgtr upmtr
        ''' # [c|z]*.f
        #######
        sclaux = laux + ' econd '                  # s*.f
        dzlaux = laux + ' secnd '                  # d*.f
        slasrc = lasrc + sd_lasrc                  # s*.f
        dlasrc = lasrc + sd_lasrc                  # d*.f
        clasrc = lasrc + cz_lasrc + ' srot srscl ' # c*.f
        zlasrc = lasrc + cz_lasrc + ' drot drscl ' # z*.f
        oclasrc = ' icmax1 scsum1 '                # *.f
        ozlasrc = ' izmax1 dzsum1 '                # *.f
        sources = ['s%s.f'%f for f in (sclaux+slasrc).split()] \
                  + ['d%s.f'%f for f in (dzlaux+dlasrc).split()] \
                  + ['c%s.f'%f for f in (clasrc).split()] \
                  + ['z%s.f'%f for f in (zlasrc).split()] \
                  + ['%s.f'%f for f in (allaux+oclasrc+ozlasrc).split()]
        sources = [os.path.join(src_dir,f) for f in sources]
        #XXX: should we check here actual existence of source files?
        info = {'sources':sources}
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

class blas_src_info(system_info):
    section = 'blas_src'
    dir_env_var = 'BLAS_SRC'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + combine_paths(d,['blas']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d,'daxpy.f')):
                src_dir = d
                break
        if not src_dir:
            #XXX: Get sources from netlib. May be ask first.
            return
        blas1 = '''
        caxpy csscal dnrm2 dzasum saxpy srotg zdotc ccopy cswap drot
        dznrm2 scasum srotm zdotu cdotc dasum drotg icamax scnrm2
        srotmg zdrot cdotu daxpy drotm idamax scopy sscal zdscal crotg
        dcabs1 drotmg isamax sdot sswap zrotg cscal dcopy dscal izamax
        snrm2 zaxpy zscal csrot ddot dswap sasum srot zcopy zswap
        '''
        blas2 = '''
        cgbmv chpmv ctrsv dsymv dtrsv sspr2 strmv zhemv ztpmv cgemv
        chpr dgbmv dsyr lsame ssymv strsv zher ztpsv cgerc chpr2 dgemv
        dsyr2 sgbmv ssyr xerbla zher2 ztrmv cgeru ctbmv dger dtbmv
        sgemv ssyr2 zgbmv zhpmv ztrsv chbmv ctbsv dsbmv dtbsv sger
        stbmv zgemv zhpr chemv ctpmv dspmv dtpmv ssbmv stbsv zgerc
        zhpr2 cher ctpsv dspr dtpsv sspmv stpmv zgeru ztbmv cher2
        ctrmv dspr2 dtrmv sspr stpsv zhbmv ztbsv
        '''
        blas3 = '''
        cgemm csymm ctrsm dsyrk sgemm strmm zhemm zsyr2k chemm csyr2k
        dgemm dtrmm ssymm strsm zher2k zsyrk cher2k csyrk dsymm dtrsm
        ssyr2k zherk ztrmm cherk ctrmm dsyr2k ssyrk zgemm zsymm ztrsm
        '''
        sources = [os.path.join(src_dir,f+'.f') \
                   for f in (blas1+blas2+blas3).split()]
        #XXX: should we check here actual existence of source files?
        info = {'sources':sources}
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
