#! /usr/bin/env python
# Last Change: Tue Oct 30 05:00 PM 2007 J

# This module defines some helper functions, to be used by high level checkers

from copy import deepcopy

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
    if not ret:
        env.Replace(CPPPATH = oldCPPPATH)
        env.Replace(CFLAGS = oldCFLAGS)
        context.Result('Failed: %s include not found' % name)
        return 0

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
     
