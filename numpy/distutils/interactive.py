
import os
import sys
from pprint import pformat

__all__ = ['interactive_sys_argv']

def show_information(*args):
    print 'Python',sys.version
    for a in ['platform','prefix','byteorder','path']:
        print 'sys.%s = %s' % (a,pformat(getattr(sys,a)))
    for a in ['name']:
        print 'os.%s = %s' % (a,pformat(getattr(os,a)))
    if hasattr(os,'uname'):
        print 'system,node,release,version,machine = ',os.uname()    

def show_environ(*args):
    for k,i in os.environ.items():
        print '  %s = %s' % (k, i)

def show_fortran_compilers(*args):
    from fcompiler import show_fcompilers
    show_fcompilers({})

def show_compilers(*args):
    from distutils.ccompiler import show_compilers
    show_compilers()

def show_tasks(argv):
    print """\

Tasks: 
  i   - Show python/platform/machine information
  ie   - Show environment information
  f   - Show Fortran compilers information
  f<name> - Set Fortran compiler
  c       - Show C compilers information
  c<name> - Set C compiler
  e   - Edit proposed sys.argv[1:].

Task aliases:
  1         - Build
  2         - Install
  2<prefix> - Install with prefix.
  3         - Inplace build
  4         - Source distribution
  5         - Binary distribution

Proposed sys.argv = %s
    """ % (argv)


from exec_command import splitcmdline

def edit_argv(*args):
    argv = args[0]
    s = raw_input('Edit argv [%s]: ' % (' '.join(argv[1:])))
    if s:
        argv[1:] = splitcmdline(s)
    return
    
def interactive_sys_argv(argv):
    print '='*72
    print 'Starting interactive session'
    print '-'*72

    try:
        try:
            import readline
        except ImportError:
            pass
        else:
            import tempfile
            tdir = tempfile.gettempdir()
            username = os.environ.get('USER',os.environ.get('USERNAME','UNKNOWN'))
            histfile = os.path.join(tdir,".pyhist_interactive_setup-" + username)
            try:
                try: readline.read_history_file(histfile)
                except IOError: pass
                import atexit
                atexit.register(readline.write_history_file, histfile)
            except AttributeError: pass
    except Exception, msg:
        print msg

    task_dict = {'i':show_information,
                 'ie':show_environ,
                 'f':show_fortran_compilers,
                 'c':show_compilers,
                 'e':edit_argv,
                 }
    c_compiler_name = None
    f_compiler_name = None

    while 1:
        show_tasks(argv)
        try:
            task = raw_input('Choose a task (^D to quit, Enter to continue with setup): ').lower()
        except EOFError:
            print
            task = 'quit'
        if task=='': break
        if task=='quit': sys.exit()
        task_func = task_dict.get(task,None)
        if task_func is None:
            if task[0]=='c':
                c_compiler_name = task[1:]
            if task[0]=='f':
                f_compiler_name = task[1:]
            if task[0]=='2' and len(task)>1:
                prefix = task[1:]
                task = task[0]
            else:
                prefix = None
            if task == '4':
                argv[1:] = ['sdist','-f']
            elif task in '1235':
                cmd_opts = {'config':[],'config_fc':[],
                            'build_ext':[],'build_src':[],
                            'build_clib':[]}
                if c_compiler_name is not None:
                    c = '--compiler=%s' % (c_compiler_name)
                    cmd_opts['config'].append(c)
                    cmd_opts['build_ext'].append(c)
                    cmd_opts['build_clib'].append(c)
                if f_compiler_name is not None:
                    c = '--fcompiler=%s' % (f_compiler_name)
                    cmd_opts['config_fc'].append(c)
                    cmd_opts['build_ext'].append(c)
                    cmd_opts['build_clib'].append(c)
                if task=='3':
                    cmd_opts['build_ext'].append('--inplace')
                    cmd_opts['build_src'].append('--inplace')
                conf = []
                for k,opts in cmd_opts.items():
                    if opts:
                        conf.extend([k]+opts)
                if task=='1':
                    argv[1:] = conf+['build']
                elif task=='2':
                    if prefix is not None:
                        argv[1:] = conf+['install','--prefix=%s' % (prefix)]
                    else:
                        argv[1:] = conf+['install']
                elif task=='3':
                    argv[1:] = conf+['build']
                elif task=='5':
                    if sys.platform=='win32':
                        argv[1:] = conf+['bdist_wininst']
                    else:
                        argv[1:] = conf+['bdist']
            else:
                print 'Skipping unknown task:',`task`
        else:
            print '-'*68
            try:
                task_func(argv)
            except Exception,msg:
                print 'Failed running task %s: %s' % (task,msg)
                break
            print '-'*68
        print

    print '-'*72
    return argv

