#! /usr/bin/env python
# Last Change: Wed Oct 31 08:00 PM 2007 J

# This module defines some helper functions, to be used by high level checkers

from copy import deepcopy

_arg2env = {'cpppath' : 'CPPPATH',
            'cflags' : 'CFLAGS',
            'libpath' : 'LIBPATH',
            'libs' : 'LIBS',
            'linkflags' : 'LINKFLAGS',
            'rpath' : 'RPATH',
            'frameworks' : 'FRAMEWORKS'}

def save_and_set(env, opts):
    """keys given as config opts args."""
    saved_keys = {}
    keys = opts.data.keys()
    for k in keys:
        saved_keys[k] = (env.has_key(_arg2env[k]) and\
                         deepcopy(env[_arg2env[k]])) or\
                        []
    kw = zip([_arg2env[k] for k in keys], [opts.data[k] for k in keys])
    kw = dict(kw)
    env.AppendUnique(**kw)
    return saved_keys

def restore(env, saved_keys):
    keys = saved_keys.keys()
    kw = zip([_arg2env[k] for k in keys], 
             [saved_keys[k] for k in keys])
    kw = dict(kw)
    env.Replace(**kw)

def check_symbol(context, headers, sym, extra = r''):
    # XXX: add dep vars in code
    #code = [r'#include <%s>' %h for h in headers]
    code = []
    code.append(r'''
#undef %(func)s
#ifdef __cplusplus
extern "C" 
#endif
char %(func)s();

int main()
{
return %(func)s();
return 0;
}
''' % {'func' : sym})
    code.append(extra)
    return context.TryLink('\n'.join(code), '.c')

class ConfigOpts:
    # Any added key should be added as an argument to __init__ 
    _keys = ['cpppath', 'cflags', 'libpath', 'libs', 'linkflags', 'rpath',
             'frameworks']
    def __init__(self, cpppath = None, cflags = None, libpath = None, libs = None, 
                 linkflags = None, rpath = None, frameworks = None):
        data = {}

        if not cpppath:
            data['cpppath'] = []
        else:
            data['cpppath'] = cpppath

        if not cflags:
            data['cflags'] = []
        else:
            data['cflags'] = cflags

        if not libpath:
            data['libpath'] = []
        else:
            data['libpath'] = libpath

        if not libs:
            data['libs'] = []
        else:
            data['libs'] = libs

        if not linkflags:
            data['linkflags'] = []
        else:
            data['linkflags'] = linkflags

        if not rpath:
            data['rpath'] = []
        else:
            data['rpath'] = rpath

        if not frameworks:
            data['frameworks'] = []
        else:
            data['frameworks'] = frameworks

        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __repr__(self):
        msg = [r'%s : %s' % (k, i) for k, i in self.data.items()]
        return '\n'.join(msg)

class ConfigRes():
    def __init__(self, name, cfgopts, origin, version = None):
        self.name = name
        self.data = cfgopts.data
        self.origin = origin
        self.version = version

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def is_customized(self):
        return bool(self.origin)

    def __repr__(self):
        msg = ['Using %s' % self.name]
        if self.is_customized():
            msg += [  'Customized items site.cfg:']
        else:
            msg += ['  Using default configuration:']

        msg += ['  %s : %s' % (k, i) for k, i in self.data.items() if len(i) > 0]
        msg += ['  Version is : %s' % self.version]
        return '\n'.join(msg)

    def __str__(self):
        return self.__repr__()

def _check_headers(context, cpppath, cflags, headers, autoadd):          
    """Try to compile code including the given headers."""       
    env = context.env        
         
    #----------------------------        
    # Check headers are available        
    #----------------------------        
    oldCPPPATH = (env.has_key('CPPPATH') and deepcopy(env['CPPPATH'])) or []         
    oldCFLAGS = (env.has_key('CFLAGS') and deepcopy(env['CFLAGS'])) or []        
    env.AppendUnique(CPPPATH = cpppath)          
    env.AppendUnique(CFLAGS = cflags)        
    # XXX: handle context        
    hcode = ['#include <%s>' % h for h in headers]       
         
    # HACK: we add cpppath in the command of the source, to add dependency of        
    # the check on the cpppath.          
    hcode.extend(['#if 0', '%s' % cpppath, '#endif\n'])          
    src = '\n'.join(hcode)       
         
    ret = context.TryCompile(src, '.c')          
    if ret == 0 or autoadd == 0:         
        env.Replace(CPPPATH = oldCPPPATH)        
        env.Replace(CFLAGS = oldCFLAGS)          
        return 0         
        
    return ret 

def check_include_and_run(context, name, cpppath, headers, run_src, libs,
                          libpath, linkflags, cflags, autoadd = 1):
    """This is a basic implementation for generic "test include and run"
    testers.
    
    For example, for library foo, which implements function do_foo, and with
    include header foo.h, this will:
        - test that foo.h is found and compilable by the compiler
        - test that the given source code can be compiled. The source code
          should contain a simple program with the function.
          
    Arguments:
        - name: name of the library
        - cpppath: list of directories
        - headers: list of headers
        - run_src: the code for the run test
        - libs: list of libraries to link
        - libpath: list of library path.
        - linkflags: list of link flags to add."""

    context.Message('Checking for %s ... ' % name)
    env = context.env

    ret = _check_headers(context, cpppath, cflags, headers, autoadd)
    if not ret:
         context.Result('Failed: %s include not found' % name)

    #------------------------------
    # Check a simple example works
    #------------------------------
    oldLIBPATH = (env.has_key('LIBPATH') and deepcopy(env['LIBPATH'])) or []
    oldLIBS = (env.has_key('LIBS') and deepcopy(env['LIBS'])) or []
    # XXX: RPATH, drawbacks using it ?
    oldRPATH = (env.has_key('RPATH') and deepcopy(env['RPATH'])) or []
    oldLINKFLAGS = (env.has_key('LINKFLAGS') and deepcopy(env['LINKFLAGS'])) or []
    env.AppendUnique(LIBPATH = libpath)
    env.AppendUnique(LIBS = libs)
    env.AppendUnique(RPATH = libpath)
    env.AppendUnique(LINKFLAGS = linkflags)

    # HACK: we add libpath and libs at the end of the source as a comment, to
    # add dependency of the check on those.
    src = '\n'.join(['#include <%s>' % h for h in headers] +\
                    [run_src, '#if 0', '%s' % libpath, 
                     '%s' % headers, '%s' % libs, '#endif'])
    ret = context.TryLink(src, '.c')
    if (not ret or not autoadd):
        # If test failed or autoadd = 0, restore everything
        env.Replace(LIBS = oldLIBS)
        env.Replace(LIBPATH = oldLIBPATH)
        env.Replace(RPATH = oldRPATH)
        env.Replace(LINKFLAGS = oldLINKFLAGS)

    if not ret:
        context.Result('Failed: %s test could not be linked and run' % name)
        return 0

    context.Result(ret)
    return ret
