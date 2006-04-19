
import os
import sys
from pprint import pformat

__all__ = ['interactive_sys_argv']

def show_information():
    print 'Python',sys.version
    for a in ['platform','prefix','byteorder','path']:
        print 'sys.%s = %s' % (a,pformat(getattr(sys,a)))
    if hasattr(os,'uname'):
        print 'system,node,release,version,machine = ',os.uname()    

def show_environ():
    for k,i in os.environ.items():
        print '  %s = %s' % (k, i)

def show_fortran_compilers():
    from fcompiler import show_fcompilers
    show_fcompilers({})

def show_tasks():
    print """\

Tasks: 
  i   - Show python/platform/machine information
  e   - Show environment information
  f   - Show Fortran compilers information
  c   - Continue with running setup
  q   - Quit setup script
    """

def interactive_sys_argv(argv):
    print '='*72
    print 'Starting interactive session'
    print '-'*72

    task_dict = {'i':show_information,
                 'e':show_environ,
                 'f':show_fortran_compilers}

    while 1:
        show_tasks()
        task = raw_input('Choose a task: ').lower()
        if task=='c': break
        if task=='q': sys.exit()
        for t in task:
            task_func = task_dict.get(t,None)
            if task_func is None:
                print 'Skipping task:',`t`
                continue
            print '-'*68
            try:
                task_func()
            except Exception,msg:
                print 'Failed running task %s: %s' % (task,msg)
                break
            print '-'*68
        print

    print '-'*72
    argv.append('--help-commands') # for testing
    return argv

