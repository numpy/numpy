#!/usr/bin/env python

import os,sys,tempfile

def run_command(cmd):
    print 'Running %r:' % (cmd)
    s = os.system(cmd)
    print '------'
def run():
    _path = os.getcwd()
    os.chdir(tempfile.gettempdir())
    print '------'
    print 'os.name=%r' % (os.name)
    print '------'
    print 'sys.platform=%r' % (sys.platform)
    print '------'
    print 'sys.version:'
    print sys.version
    print '------'
    print 'sys.prefix:'
    print sys.prefix
    print '------'
    print 'sys.path=%r' % (':'.join(sys.path))
    print '------'
    try:
        import Numeric
        has_Numeric = 1
    except ImportError:
        print 'Failed to import Numeric:',sys.exc_value
        has_Numeric = 0
    try:
        import numarray
        has_numarray = 1
    except ImportError:
        print 'Failed to import numarray:',sys.exc_value
        has_numarray = 0
    try:
        import scipy.base
        has_newscipy = 1
    except ImportError:
        print 'Failed to import new scipy:', sys.exc_value
        has_newscipy = 0
    try:
        import f2py2e
        has_f2py2e = 1
    except ImportError:
        print 'Failed to import f2py2e:',sys.exc_value
        has_f2py2e = 0
    try:
        import scipy.distutils
        has_scipy_distutils = 2
    except ImportError:
        try:
            import scipy_distutils
            has_scipy_distutils = 1
        except ImportError:
            print 'Failed to import scipy_distutils:',sys.exc_value
            has_scipy_distutils = 0
    if has_Numeric:
        try:
            print 'Found Numeric version %r in %s' % \
                  (Numeric.__version__,Numeric.__file__)
        except Exception,msg:
            print 'error:',msg
            print '------'
    if has_numarray:
        try:
            print 'Found numarray version %r in %s' % \
                  (numarray.__version__,numarray.__file__)
        except Exception,msg:
            print 'error:',msg
            print '------'
    if has_newscipy:
        try:
            print 'Found new scipy version %r in %s' % \
                  (scipy.__version__, scipy.__file__)
        except Exception,msg:
            print 'error:', msg
            print '------'
    if has_f2py2e:
        try:
            print 'Found f2py2e version %r in %s' % \
                  (f2py2e.__version__.version,f2py2e.__file__)
        except Exception,msg:
            print 'error:',msg
            print '------'
    if has_scipy_distutils:
        try:
            if has_scipy_distutils==2:
                print 'Found scipy.distutils version %r in %r' % (\
            scipy.distutils.__version__,
            scipy.distutils.__file__)
            else:
                print 'Found scipy_distutils version %r in %r' % (\
            scipy_distutils.scipy_distutils_version.scipy_distutils_version,
            scipy_distutils.__file__)
            print '------'
        except Exception,msg:
            print 'error:',msg
            print '------'
        try:
            if has_scipy_distutils==1:
                print 'Importing scipy_distutils.command.build_flib ...',
                import scipy_distutils.command.build_flib as build_flib
                print 'ok'
                print '------'
                try:
                    print 'Checking availability of supported Fortran compilers:'
                    for compiler_class in build_flib.all_compilers:
                        compiler_class(verbose=1).is_available()
                        print '------'
                except Exception,msg:
                    print 'error:',msg
                    print '------'
        except Exception,msg:
            print 'error:',msg,'(ignore it, build_flib is obsolute for scipy.distutils 0.2.2 and up)'
            print '------'
        try:
            if has_scipy_distutils==2:
                print 'Importing scipy.distutils.fcompiler ...',
                import scipy.distutils.fcompiler as fcompiler
            else:
                print 'Importing scipy_distutils.fcompiler ...',
                import scipy_distutils.fcompiler as fcompiler
            print 'ok'
            print '------'
            try:
                print 'Checking availability of supported Fortran compilers:'
                fcompiler.show_fcompilers()
                print '------'
            except Exception,msg:
                print 'error:',msg
                print '------'
        except Exception,msg:
            print 'error:',msg
            print '------'
        try:
            if has_scipy_distutils==2:
                print 'Importing scipy.distutils.cpuinfo ...',
                from scipy.distutils.cpuinfo import cpuinfo
                print 'ok'
                print '------'
            else:
                try:
                    print 'Importing scipy_distutils.command.cpuinfo ...',
                    from scipy_distutils.command.cpuinfo import cpuinfo
                    print 'ok'
                    print '------'
                except Exception,msg:
                    print 'error:',msg,'(ignore it)'
                    print 'Importing scipy_distutils.cpuinfo ...',
                    from scipy_distutils.cpuinfo import cpuinfo
                    print 'ok'
                    print '------'
            cpu = cpuinfo()
            print 'CPU information:',
            for name in dir(cpuinfo):
                if name[0]=='_' and name[1]!='_' and getattr(cpu,name[1:])():
                    print name[1:],
            print '------'
        except Exception,msg:
            print 'error:',msg
            print '------'
    os.chdir(_path)
if __name__ == "__main__":
    run()
