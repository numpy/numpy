#! /usr/bin/env python
# Last Change: Tue Oct 30 09:00 PM 2007 J

# This module defines some helper functions, to be used by high level checkers

from copy import deepcopy

_arg2env = {'cpppath' : 'CPPPATH',
            'cflags' : 'CFLAGS',
            'libpath' : 'LIBPATH',
            'libs' : 'LIBS',
            'linkflags' : 'LINKFLAGS'}

def save_set(env, opts):
    """keys given as config opts args."""
    saved_keys = {}
    keys = opts.keys()
    for k in keys:
        saved_keys[k] = (env.has_key(_arg2env[k]) and\
                         deepcopy(env[_arg2env[k]])) or\
                        []
    kw = zip([_arg2env[k] for k in keys], [opts[k] for k in keys])
    kw = dict(kw)
    env.AppendUnique(**kw)
    return saved_keys

def restore(env, saved_keys):
    keys = saved_keys.keys()
    kw = zip([_arg2env[k] for k in keys], 
             [saved_keys[k] for k in keys])
    kw = dict(kw)
    env.Replace(**kw)

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

    ret = _check_headers(context, cpppath, cflags, headers)
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
