""" Auto test tools for SciPy

    Do not run this as root!  If you enter something
    like /usr as your test directory, it'll delete
    /usr/bin, usr/lib, etc.  So don't do it!!!
    
    
    Author: Eric Jones (eric@enthought.com)
"""
from distutils import file_util
from distutils import dir_util
from distutils.errors import DistutilsFileError
#import tarfile
import sys, os, stat, time
import gzip
import tempfile, cStringIO
import urllib
import logging

if sys.platform == 'cygwin':
    local_repository = "/cygdrive/i/tarballs"
elif sys.platform == 'win32':    
    local_repository = "i:\tarballs"
else:
    local_repository = "/home/shared/tarballs"

local_mail_server = "enthought.com"

python_ftp_url = "ftp://ftp.python.org/pub/python"
numeric_url = "http://prdownloads.sourceforge.net/numpy"
f2py_url = "http://cens.ioc.ee/projects/f2py2e/2.x"
scipy_url = "ftp://www.scipy.org/pub"
blas_url = "http://www.netlib.org/blas"
lapack_url = "http://www.netlib.org/lapack"
#atlas_url = "http://prdownloads.sourceforge.net/math-atlas"
atlas_url = "http://www.scipy.org/Members/eric"


#-----------------------------------------------------------------------------
# Generic installation class. 
# built to handle downloading/untarring/building/installing arbitrary software
#-----------------------------------------------------------------------------

class package_installation:    
    def __init__(self,version='', dst_dir = '.',
                 logger = None, python_exe='python'):
        #---------------------------------------------------------------------
        # These should be defined in sub-class before calling this
        # constructor
        #---------------------------------------------------------------------
        # 
        #self.package_url -- The name of the url where tarball can be found.
        #self.package_base_name -- The base name of the source tarball.
        #self.package_dir_name -- Top level directory of unpacked tarball
        #self.tarball_suffix -- usually tar.gz or .tgz
        #self.build_type -- 'make' or 'setup' for makefile or python setup file
        
        # Version of the software package.
        self.version = version

        # Only used by packages built with setup.py
        self.python_exe = python_exe
        
        # Directory where package is unpacked/built/installed
        self.dst_dir = os.path.abspath(dst_dir)        
        
        if not logger:
            self.logger = logging
        else:
            self.logger = logger    

        # make sure the destination exists
        make_dir(self.dst_dir,logger=self.logger)

        # Construct any derived names built from the above names.
        self.init_names()
        
    def init_names(self):            
        self.package_dir = os.path.join(self.dst_dir,self.package_dir_name)
        self.tarball = self.package_base_name + '.' + self.tarball_suffix

    def get_source(self):
        """ Grab the source tarball from a repository.
        
            Try a local repository first.  If the file isn't found,
            grab it from an ftp site.
        """
        local_found = 0
        if self.local_source_up_to_date():
            try:
                self.get_source_local()
                local_found = 1                
            except DistutilsFileError:
                pass
        
        if not local_found:
            self.get_source_ftp()
                    
    def local_source_up_to_date(self):
        """ Hook to test whether a file found in the repository is current
        """
        return 1
        
    def get_source_local(self):
        """ Grab the requested tarball from a local repository of source
            tarballs.  If it doesn't exist, an error is raised.
        """
        file = os.path.join(local_repository,self.tarball)        
        dst_file = os.path.join(self.dst_dir,self.tarball)
        self.logger.info("Searching local repository for %s" % file)
        try:
            copy_file(file,dst_file,self.logger)
        except DistutilsFileError, msg:
            self.logger.info("Not found:",msg)
            raise
        
    def get_source_ftp(self):
        """ Grab requested tarball from a ftp site specified as a url.           
        """
        url = '/'.join([self.package_url,self.tarball])
     
        self.logger.info('Opening: %s' % url)
        f = urllib.urlopen(url)
        self.logger.info('Downloading: this may take a while')
        contents = f.read(-1)
        f.close()
        self.logger.info('Finished download (size=%d)' % len(contents))
     
        output_file = os.path.join(self.dst_dir,self.tarball)
        write_file(output_file,contents,self.logger)

        # Put file in local repository so we don't have to download it again.
        self.logger.info("Caching file in repository" )
        src_file = output_file
        repos_file = os.path.join(local_repository,self.tarball)        
        copy_file(src_file,repos_file,self.logger)

    def unpack_source(self,sub_dir = None):
        """ equivalent to 'tar -xzvf file' in the given sub_dir
        """       
        tarfile = os.path.join(self.dst_dir,self.tarball)
        old_dir = None
        
        # copy and move into sub directory if it is specified.
        if sub_dir:
            dst_dir = os.path.join(self.dst_dir,sub_dir)
            dst_file = os.path.join(dst_dir,self.tarball)
            copy_file(tarfile,dst_file)
            change_dir(dst_dir,self.logger)
        try:
            try:
                # occasionally the tarball is not zipped, try this first.
                untar_file(self.tarball,self.dst_dir,
                           self.logger,silent_failure=1)
            except:
                # otherwise, handle the fact that it is zipped        
                dst = os.path.join(self.dst_dir,'tmp.tar')        
                decompress_file(tarfile,dst,self.logger)                
                untar_file(dst,self.dst_dir,self.logger)
                remove_file(dst,self.logger)
        finally:
            if old_dir:
                unchange_dir(self.logger)

    #def auto_configure(self):
    #    cmd = os.path.join('.','configure')
    #    try:
    #        text = run_command(cmd,self.package_dir,self.logger,log_output=0)
    #    except ValueError, e:
    #        status, text = e
    #        self.logger.exception('Configuration Error:\n'+text)
    def auto_configure(self):
        cmd = os.path.join('.','configure')
        text = run_command(cmd,self.package_dir,self.logger)
        
    def build_with_make(self):
        cmd = 'make'
        text = run_command(cmd,self.package_dir,self.logger)
        
    def install_with_make(self, prefix = None):
        if prefix is None:
            prefix = os.path.abspath(self.dst_dir)
        cmd = 'make install prefix=%s' % prefix
        text = run_command(cmd,self.package_dir,self.logger)
        
    def python_setup(self):
        cmd = self.python_exe + ' setup.py install'
        text = run_command(cmd,self.package_dir,self.logger)
        
    def _make(self,**kw):
        """ This generally needs to be overrridden in the derived class,
            but this will suffice for the standard configure/make process.            
        """
        self.logger.info("### Begin Configure: %s" % self.package_base_name)
        self.auto_configure()
        self.logger.info("### Finished Configure: %s" % self.package_base_name)
        self.logger.info("### Begin Build: %s" % self.package_base_name)
        self.build_with_make()
        self.logger.info("### Finished Build: %s" % self.package_base_name)
        self.logger.info("### Begin Install: %s" % self.package_base_name)
        self.install_with_make()
        self.logger.info("### Finished Install: %s" % self.package_base_name)

    def install(self):
        self.logger.info('####### Building:    %s' % self.package_base_name)
        self.logger.info('        Version:     %s' % self.version)
        self.logger.info('        Url:         %s' % self.package_url)
        self.logger.info('        Install dir: %s' % self.dst_dir)
        self.logger.info('        Package dir: %s' % self.package_dir)
        self.logger.info('        Suffix:      %s' % self.tarball_suffix)
        self.logger.info('        Build type:  %s' % self.build_type)

        self.logger.info("### Begin Get Source: %s" % self.package_base_name)
        self.get_source()
        self.unpack_source()
        self.logger.info("### Finished Get Source: %s" % self.package_base_name)

        if self.build_type == 'setup':
            self.python_setup()
        else:    
            self._make()
        self.logger.info('####### Finished Building: %s' % self.package_base_name)            
            
#-----------------------------------------------------------------------------
# Installation class for Python itself.
#-----------------------------------------------------------------------------
        
class python_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        # Specialization for Python.        
        self.package_base_name = 'Python-'+version
        self.package_dir_name = self.package_base_name
        self.package_url = '/'.join([python_ftp_url,version])
        self.tarball_suffix = 'tgz'
        self.build_type = 'make'
        
        package_installation.__init__(self,version,dst_dir,logger,python_exe)

    def write_install_config(self):    
        """ Make doesn't seem to install scripts in the correct places.
        
            Writing this to the python directory will solve the problem.
            [install_script]
            install-dir=<directory_name> 
        """
        self.logger.info('### Writing Install Script Hack')
        text = "[install_scripts]\n"\
               "install-dir='%s'" % os.path.join(self.dst_dir,'bin')
        file = os.path.join(self.package_dir,'setup.cfg')               
        write_file(file,text,self.logger,mode='w')
        self.logger.info('### Finished writing Install Script Hack')

    def install_with_make(self):
        """ Scripts were failing to install correctly, so a setuo.cfg
            file is written to force installation in the correct place.
        """        
        self.write_install_config()
        package_installation.install_with_make(self)

    def get_exe_name(self):
        pyname = os.path.join('.','python')
        cmd = pyname + """ -c "import sys;print '%d.%d' % sys.version_info[:2]" """
        text = run_command(cmd,self.package_dir,self.logger)
        exe = os.path.join(self.dst_dir,'bin','python'+text)
        return exe

#-----------------------------------------------------------------------------
# Installation class for Blas.
#-----------------------------------------------------------------------------

class blas_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        # Specialization for for "slow" blas
        self.package_base_name = 'blas'
        self.package_dir_name = 'BLAS'
        self.package_url = blas_url
        self.tarball_suffix = 'tgz'
        self.build_type = 'make'
                
        self.platform = 'LINUX'
        package_installation.__init__(self,version,dst_dir,logger,python_exe)

    def unpack_source(self,subdir=None):
        """ Dag.  blas.tgz doesn't have directory information -- its
            just a tar ball of fortran source code.  untar it in the
            BLAS directory
        """
        package_installation.unpack_source(self,self.package_dir_name)
            
    def auto_configure(self):
        # nothing to do.
        pass
    def build_with_make(self, **kw):
        libname = 'blas_LINUX.a'
        cmd = 'g77 -funroll-all-loops -fno-f2c -O3 -c *.f;ar -cru %s' % libname
        text = run_command(cmd,self.package_dir,self.logger)
        
    def install_with_make(self, **kw):
        # not really using make -- we'll just copy the file over.        
        src_file = os.path.join(self.package_dir,'blas_%s.a' % self.platform)
        dst_file = os.path.join(self.dst_dir,'lib','libblas.a')
        self.logger.info("Installing blas")
        copy_file(src_file,dst_file,self.logger)
        
#-----------------------------------------------------------------------------
# Installation class for Lapack.
#-----------------------------------------------------------------------------

class lapack_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        # Specialization for Lapack 3.0 + updates        
        self.package_base_name = 'lapack'
        self.package_dir_name = 'LAPACK'
        self.package_url = lapack_url
        self.tarball_suffix = 'tgz'
        self.build_type = 'make'
        
        self.platform = 'LINUX'
        package_installation.__init__(self,version,dst_dir,logger,python_exe)

    def auto_configure(self):
        # perhaps this should actually override auto_conifgure
        # before make, we need to copy the appropriate setup file in.
        # should work anywhere g77 works...
        make_inc = 'make.inc.' + self.platform
        src_file = os.path.join(self.package_dir,'INSTALL',make_inc)
        dst_file = os.path.join(self.package_dir,'make.inc')
        copy_file(src_file,dst_file,self.logger)
        
    def build_with_make(self, **kw):
        cmd = 'make install lapacklib'
        text = run_command(cmd,self.package_dir,self.logger)
        
    def install_with_make(self, **kw):
        # not really using make -- we'll just copy the file over.
        src_file = os.path.join(self.package_dir,'lapack_%s.a' % self.platform)
        dst_file = os.path.join(self.dst_dir,'lib','liblapack.a')        
        copy_file(src_file,dst_file,self.logger)

#-----------------------------------------------------------------------------
# Installation class for Numeric
#-----------------------------------------------------------------------------

class numeric_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        self.package_base_name = 'Numeric-'+version
        self.package_dir_name = self.package_base_name
        self.package_url = numeric_url
        self.tarball_suffix = 'tar.gz'
        self.build_type = 'setup'        

        package_installation.__init__(self,version,dst_dir,logger,python_exe)


#-----------------------------------------------------------------------------
# Installation class for f2py
#-----------------------------------------------------------------------------

class f2py_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        # Typical file format: F2PY-2.13.175-1250.tar.gz
        self.package_base_name = 'F2PY-'+version
        self.package_dir_name = self.package_base_name
        self.package_url = f2py_url
        self.tarball_suffix = 'tar.gz'
        self.build_type = 'setup'        
                
        package_installation.__init__(self,version,dst_dir,logger,python_exe)


#-----------------------------------------------------------------------------
# Installation class for Atlas.
# This is a binary install *NOT* a source install.
# The source install is a pain to automate.
#-----------------------------------------------------------------------------

class atlas_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        #self.package_base_name = 'atlas' + version
        #self.package_dir_name = 'ATLAS'
        self.package_base_name = 'atlas-RH7.1-PIII'
        self.package_dir_name = 'atlas'
        self.package_url = atlas_url
        self.tarball_suffix = 'tgz'
        self.build_type = 'make'        
        
        package_installation.__init__(self,version,dst_dir,logger,python_exe)

    def auto_configure(self,**kw):
        pass
    def build_with_make(self,**kw):
        pass
    def install_with_make(self, **kw):
        # just copy the tree over.
        dst = os.path.join(self.dst_dir,'lib','atlas')
        self.logger.info("Installing Atlas")
        copy_tree(self.package_dir,dst,self.logger)

#-----------------------------------------------------------------------------
# Installation class for scipy
#-----------------------------------------------------------------------------

class scipy_installation(package_installation):
    
    def __init__(self,version='', dst_dir = '.',logger=None,python_exe='python'):
        
        self.package_base_name = 'scipy_snapshot'
        self.package_dir_name = 'scipy'
        self.package_url = scipy_url
        self.tarball_suffix = 'tgz'
        self.build_type = 'setup'
        
        package_installation.__init__(self,version,dst_dir,logger,python_exe)
                    
    def local_source_up_to_date(self):
        """ Hook to test whether a file found in the repository is current
        """
        file = os.path.join(local_repository,self.tarball)
        up_to_date = 0
        try:
            file_time = os.stat(file)[stat.ST_MTIME]        
            fyear,fmonth,fday = time.localtime(file_time)[:3]
            year,month,day = time.localtime()[:3]
            if fyear == year and fmonth == month and fday == day:
                up_to_date = 1
                self.logger.info("Repository file up to date: %s" % file)
        except OSError, msg:
            pass
        return up_to_date
                
#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------


#if os.name == 'nt':
#    def exec_command(command):
#        """ not sure how to get exit status on nt. """
#        in_pipe,out_pipe = os.popen4(command)
#        in_pipe.close()
#        text = out_pipe.read()
#        return 0, text
#else:
#    import commands
#    exec_command = commands.getstatusoutput
   
# This may not work on Win98... The above stuff was to handle these machines.
import commands
exec_command = commands.getstatusoutput

def copy_file(src,dst,logger=None):
    if not logger:
        logger = logging
    logger.info("Copying %s->%s" % (src,dst))        
    try:
        file_util.copy_file(src,dst)
    except Exception, e:     
        logger.exception("Copy Failed")        
        raise

def copy_tree(src,dst,logger=None):
    if not logger:
        logger = logging
    logger.info("Copying directory tree %s->%s" % (src,dst))        
    try:
        dir_util.copy_tree(src,dst)
    except Exception, e:     
        logger.exception("Copy Failed")        
        raise

def remove_tree(directory,logger=None):
    if not logger:
        logger = logging
    logger.info("Removing directory tree %s" % directory)        
    try:
        dir_util.remove_tree(directory)
    except Exception, e:     
        logger.exception("Remove failed: %s" % e)        
        raise

def remove_file(file,logger=None):
    if not logger:
        logger = logging
    logger.info("Remove file %s" % file)        
    try:
        os.remove(file)
    except Exception, e:     
        logger.exception("Remove failed")        
        raise

def write_file(file,contents,logger=None,mode='wb'):
    if not logger:
        logger = logging
    logger.info('Write file: %s' % file)
    try:
        new_file = open(file,mode)
        new_file.write(contents)
        new_file.close()
    except Exception, e:     
        logger.exception("Write failed")        
        raise

def make_dir(name,logger=None):
    if not logger:
        logger = logging
    logger.info('Make directory: %s' % name)
    try:        
        dir_util.mkpath(os.path.abspath(name))
    except Exception, e:     
        logger.exception("Make Directory failed")        
        raise

# I know, I know...
old_dir = []

def change_dir(d, logger = None):
    if not logger:
        logger = logging
    global old_dir 
    cwd = os.getcwd()   
    old_dir.append(cwd)
    d = os.path.abspath(d)
    if d != old_dir[-1]:
        logger.info("Change directory: %s" % d)            
        try:
            os.chdir(d)
        except Exception, e:     
            logger.exception("Change directory failed")
            raise        
        #if d == '.':
        #    import sys,traceback
        #    f = sys._getframe()
        #    traceback.print_stack(f)

def unchange_dir(logger=None):
    if not logger:
        logger = logging            
    global old_dir
    try:
        cwd = os.getcwd()
        d = old_dir.pop(-1)            
        try:
            if d != cwd:
                logger.info("Change directory : %s" % d)
                os.chdir(d)
        except Exception, e:     
            logger.exception("Change directory failed")
            raise                    
    except IndexError:
        logger.exception("Change directory failed")
        
def decompress_file(src,dst,logger = None):
    if not logger:
        logger = logging
    logger.info("Upacking %s->%s" % (src,dst))
    try:
        f = gzip.open(src,'rb')
        contents = f.read(-1)
        f = open(dst, 'wb')
        f.write(contents)
    except Exception, e:     
        logger.exception("Unpack failed")
        raise        

    
def untar_file(file,dst_dir='.',logger = None,silent_failure = 0):    
    if not logger:
        logger = logging
    logger.info("Untarring file: %s" % (file))
    try:
        run_command('tar -xf ' + file,directory = dst_dir,
                    logger=logger, silent_failure = silent_failure)
    except Exception, e:
        if not silent_failure:     
            logger.exception("Untar failed")
        raise        

def unpack_file(file,logger = None):
    """ equivalent to 'tar -xzvf file'
    """
    dst = 'tmp.tar'
    decompress_file(file,dst,logger)                
    untar_file(dst.logger)
    remove_file(dst,logger)        


def run_command(cmd,directory='.',logger=None,silent_failure = 0):
    if not logger:
        logger = logging
    change_dir(directory,logger)    
    try:        
        msg = 'Running: %s' % cmd
        logger.info(msg)    
        status,text = exec_command(cmd)
        if status and silent_failure:
            msg = '(failed silently)'
            logger.info(msg)    
        if status and text and not silent_failure:
            logger.error('Command Failed (status=%d)\n'% status +text)
    finally:
        unchange_dir(logger)
    if status:
        raise ValueError, (status,text)
    return text            

def mail_report(from_addr,to_addr,subject,mail_server,
                build_log, test_results,info):
    
    msg = ''
    msg = msg + 'To: %s\n'   % to_addr
    msg = msg + 'Subject: %s\n' % subject
    msg = msg + '\r\n\r\n'

    for k,v in info.items():   
        msg = msg + '%s: %s\n' % (k,v)
    msg = msg + test_results + '\n'
    msg = msg + '-----------------------------\n' 
    msg = msg + '--------  BUILD LOG   -------\n' 
    msg = msg + '-----------------------------\n' 
    msg = msg + build_log
    print msg
    
    # mail results
    import smtplib 
    server = smtplib.SMTP(mail_server)    
    server.sendmail(from_addr, to_addr, msg)
    server.quit()
    

def full_scipy_build(build_dir = '.',
                     test_level = 10,
                     python_version  = '2.2.1',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot'):
    
    # for now the atlas version is ignored.  Only the 
    # binaries for RH are supported at the moment.

    build_info = {'python_version' : python_version,
                  'test_level'     : test_level,
                  'numeric_version': numeric_version,
                  'f2py_version'   : f2py_version,
                  'atlas_version'  : atlas_version,
                  'scipy_version'  : scipy_version}
                    
    dst_dir = os.path.join(build_dir,sys.platform)

    logger = logging.Logger("SciPy Test")
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    log_stream = cStringIO.StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    # also write to stderr
    stderr = logging.StreamHandler()
    stderr.setFormatter(fmt)
    logger.addHandler(stderr)

    try:
        try:    
        
            # before doing anything, we need to wipe the 
            # /bin, /lib, /man, and /include directories
            # in dst_dir.  Don't run as root.            
            make_dir(dst_dir,logger=logger)            
            change_dir(dst_dir   , logger)
            for d in ['bin','lib','man','include']:
                try:            remove_tree(d, logger)
                except OSError: pass                
            unchange_dir(logger)
            
            python = python_installation(version=python_version,
                                         logger = logger,
                                         dst_dir = dst_dir)
            python.install()
            
            python_name = python.get_exe_name()
        
            numeric = numeric_installation(version=numeric_version,
                                           dst_dir = dst_dir,
                                           logger = logger,
                                           python_exe=python_name)
            numeric.install()
            
            f2py =  f2py_installation(version=f2py_version,
                                      logger = logger,
                                      dst_dir = dst_dir,
                                      python_exe=python_name)
            f2py.install()                                
        
            # download files don't have a version specified    
            #lapack =  lapack_installation(version='',
            #                              dst_dir = dst_dir
            #                              python_exe=python_name)
            #lapack.install()                                
        
            # download files don't have a version specified    
            #blas =  blas_installation(version='',
            #                          logger = logger,
            #                          dst_dir = dst_dir,
            #                          python_exe=python_name)
            #blas.install()                                
            
            # ATLAS
            atlas =  atlas_installation(version=atlas_version,
                                        logger = logger,
                                        dst_dir = dst_dir,
                                        python_exe=python_name)
            atlas.install()
            
            # version not currently used -- need to fix this.
            scipy =  scipy_installation(version=scipy_version,
                                        logger = logger,
                                        dst_dir = dst_dir,
                                        python_exe=python_name)
            scipy.install()                                
        
            # The change to tmp makes sure there isn't a scipy directory in 
            # the local scope.
            # All tests are run.
            logger.info('Beginning Test')
            cmd = python_name +' -c "import sys,scipy;suite=scipy.test(%d);"'\
                                % test_level
            test_results = run_command(cmd, logger=logger,
                                       directory = tempfile.gettempdir())
            build_info['results'] = 'test completed (check below for pass/fail)'
        except Exception, msg:
            test_results = ''
            build_info['results'] = 'build failed: %s' % msg
            logger.exception('Build failed')
    finally:    
        to_addr = "scipy-testlog@scipy.org"
        from_addr = "scipy-test@enthought.com"
        subject = '%s,py%s,num%s,scipy%s' % (sys.platform,python_version,
                                            numeric_version,scipy_version) 
        build_log = log_stream.getvalue()
        mail_report(from_addr,to_addr,subject,local_mail_server,
                    build_log,test_results,build_info)

if __name__ == '__main__':
    build_dir = '/tmp/scipy_test'
    level = 10

    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.2.1',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # an older python
    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # an older numeric
    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '20.3',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # This fails because multiarray doesn't have 
    # arange defined.
    """
    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '20.0.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '19.0.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '18.4.1',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')
    """
