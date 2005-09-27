#!/usr/bin/env python
"""
Usage:
  runme.py <scipy_distutils commands/options and --no-wrap-functions>

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision: 1.4 $
$Date: 2002/01/09 21:56:31 $
Pearu Peterson
"""

__version__ = "$Id: runme.py,v 1.4 2002/01/09 21:56:31 pearu Exp $"

import sys,os,string

if __name__ == "__main__":
    __file__ = sys.argv[0]

if os.name == 'nt':
    def run_command(command):
        """ not sure how to get exit status on nt. """
        in_pipe,out_pipe = os.popen4(command)
        in_pipe.close()
        text = out_pipe.read()
        return 0, text
else:
    import commands
    run_command = commands.getstatusoutput


def main():
    """
    Ensure that code between try: ... finally: is executed with
    exactly the same conditions regardless of the way how this
    function is called (as a script from an arbitrary path or
    as a module member function, but not through execfile function).
    """
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    def mess(text,sys_stdout=sys_stdout):
        sys.stdout.write(text)
        sys_stdout.write(text)
    _log = open(os.path.abspath(__file__)+'.log','w')
    sys.stdout = sys.stderr = _log
    _sys_argv = sys.argv
    mess('Running %s\n'%(`string.join(sys.argv,' ')`))
    mess(' log is saved to %s\n'%(_log.name))
    _path = os.path.abspath(os.path.dirname(__file__))
    _f2pypath = os.path.normpath(os.path.join(_path,'..','..'))
    sys.path.insert(0,_f2pypath)
    _cwd = os.getcwd()
    os.chdir(_path)

    
    try:
        ############## CODE TO BE TESTED #################

        import shutil
        wdir = os.path.abspath('tmp_wdir')
        if os.path.exists(wdir):
            print ' removing ',wdir
            shutil.rmtree(wdir,1)
        print ' making ',wdir
        os.mkdir(wdir)
        shutil.copy('geniotest.py',wdir)
        cwd = os.getcwd()
        os.chdir(wdir)

        run_command(sys.executable+' geniotest.py')
        import f2py2e as f2py
        sys_argv = sys.argv
        mess(' f2py-ing..\n')
        pyf_sources = ['iotest.pyf','iotestrout.f']
        f2py_opts = []
        try:
            i = sys_argv.index('--no-wrap-functions')
        except ValueError:
            i = -1
        if i>=0:
            f2py_opts.append('no-wrap-functions')
            sys.argv = sys.argv[:i] + sys.argv[i+1:]

        if len(sys.argv)==1:
            sys.argv = sys.argv + ['build']
        if sys.argv[-1]=='build':
            sys.argv = sys.argv + ['--build-platlib','.']

        ############## building extension module ###########
        from scipy_distutils.core import setup,Extension
        ext = Extension('iotest',pyf_sources,f2py_options = f2py_opts)

        mess(' running setup..\n')
        setup(ext_modules = [ext])
        #####################################################

        sys.argv = sys_argv
        mess(' running tests..')
        status,output=run_command(sys.executable + ' runiotest.py')
        if status:
            mess('failed\n')
        else:
            succ,fail = string.count(output,'SUCCESS'),string.count(output,'FAILURE')
            mess('%s passed, %s failed\n'%(succ,fail))
            if fail:
                import re
                res = re.findall(r'FAILURE.*\n',output)
                mess(string.join(res,''))
        print 30*'*'+' TEST OUTPUT '+30*'*'
        print output
        print 30*'*'+' END OF TEST OUTPUT '+30*'*'            
        os.chdir(cwd)

        ############## END OF CODE TO BE TESTED #################
    finally:
        os.chdir(_cwd)
        del sys.path[0]
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
        sys.argv = _sys_argv
        _log.close()


if __name__ == "__main__":
    main()
