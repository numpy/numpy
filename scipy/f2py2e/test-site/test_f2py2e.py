#!/usr/bin/env python

import sys,os,shutil

if __name__ == "__main__":
    __file__ = sys.argv[0]

test_dirs = ['b']
#test_dirs = ['arr_from_obj']

def main():
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    def mess(text,sys_stdout=sys_stdout):
        sys.stdout.write(text)
        sys_stdout.write(text)
        sys_stdout.flush()
    _log = open(os.path.abspath(__file__)+'.log','w')
    #sys.stdout = sys.stderr = _log

    _path = os.path.abspath(os.path.dirname(__file__))
    _f2pypath = os.path.normpath(os.path.join(_path,'..'))
    sys.path.insert(0,_f2pypath)
    _cwd = os.getcwd()
    os.chdir(_path)
    _sys_argv = sys.argv

    if len(sys.argv)==1:
        sys.argv = sys.argv + ['build']
    if sys.argv[-1]=='build':
        sys.argv = sys.argv + ['--build-base','tmp_build',
                               '--build-platlib','.']
    
    for d in test_dirs:
        mess('----------------------------------\n')
        mess('Running tests in directory %s\n'%(d))
        mess('----------------------------------\n')
        wdir = os.path.abspath(os.path.join(d,'tmp_wdir'))
        if os.path.exists(wdir):
            print ' removing ',wdir
            shutil.rmtree(wdir,1)
        print ' copying',d,'to',wdir
        shutil.copytree(d,wdir)
        print ' cd',wdir
        os.chdir(wdir)

        try:
            result = __import__('runme').main()
            mess('----------------------------------\n')
            if result:
                mess('\t'+d+' tests ok\n')
            else:
                mess('\t'+d+' tests FAILED\n')
        except:
            mess('----------------------------------\n')
            mess('\t'+d+' crashed (%s)\n'%(sys.exc_value))
        mess('==================================\n')
        os.chdir(_path)
    mess('See %s\n'%(_log.name))
    os.chdir(_cwd)
    del sys.path[0]
    sys.argv = _sys_argv
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr
    sys.argv = _sys_argv

    _log.close()


if __name__ == "__main__":
    main()
