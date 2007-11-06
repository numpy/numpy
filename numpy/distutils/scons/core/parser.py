#! /usr/bin/env python
# Last Change: Mon Nov 05 07:00 PM 2007 J
import re
import shlex

# Those are obtained from distutils.sysconfig.get_config_vars('CFLAGS')
tiger_x86_cflags = r'-arch ppc -arch i386 -sysroot '\
'/Developers/SDKs/MacOSX10.4u.sdk -fno-strict-aliasing -Wno-long-double '\
'-no-cpp-precomp -DNDEBUG -f -O3'
linux_x86_cflags = r'-fno-strict-aliasing -DNDEBUG -g -O2 -Wall '\
        '-Wstrict-prototypes'
linux_x86_ldflags = r'-pthread -shared -Wl,-O1'

# List of arguments taking an option (with space)
cflags_argument_options = [r'-arch', r'-sysroot']
ldflags_argument_options = [r'-arch', r'-sysroot', r'-L']

def is_option(token):
    return token.startswith('-')

def has_argument(token):
    """Returns True if current token is an option expecting an argument."""
    for i in cflags_argument_options:
        if token.startswith(i):
            return True
    return False

def cc_process_argument(lexer, token):
    """Merge subsequent tokens while no option flag is detected."""
    token_list = [token]
    token = lexer.get_token()
    while token is not None and not is_option(token):
        token_list.append(token)
        token = lexer.get_token()
    return token_list, token

def ld_process_argument(lexer, token):
    """Merge subsequent tokens while no option flag is detected."""
    token_list = [token]
    token = lexer.get_token()
    while token is not None and not is_option(token):
        token_list.append(token)
        token = lexer.get_token()
    return token_list, token

def parse_posix_ld_token(lexer, token):
    if not token:
        return None

    if has_argument(token):
        token_list, next_token = ld_process_argument(lexer, token)
        token = ' '.join(token_list)
    else:
        next_token = lexer.get_token()

    if token.startswith('-L'):
        print "token %s is a libpath flag" % token
    elif token.startswith('-W'):
        print "token %s is a linker flag" % token
    elif token.startswith('-O'):
        print "token %s is a optim flag" % token
    elif token.startswith('-g'):
        print "token %s is a debug related flag" % token
    elif token.startswith('-pthread'):
        print "token %s is a thread related flag" % token
    else:
        print "token %s is unknown (extra)" % token

    return next_token

def parse_posix_cc_token(lexer, token):
    if not token:
        return None

    if has_argument(token):
        token_list, next_token = cc_process_argument(lexer, token)
        token = ' '.join(token_list)
    else:
        next_token = lexer.get_token()

    if token.startswith('-W'):
        print "token %s is a warning flag" % token
    elif token.startswith('-D'):
        print "token %s is a define flag" % token
    elif token.startswith('-f'):
        print "token %s is a compilation flag" % token
    elif token.startswith('-O'):
        print "token %s is a optim flag" % token
    elif token.startswith('-g'):
        print "token %s is a debug related flag" % token
    elif token.startswith('-pthread'):
        print "token %s is a thread related flag" % token
    else:
        print "token %s is unknown (extra)" % token

    return next_token

if __name__ == '__main__':
    def parse_cflags(flags):
        a = shlex.shlex(flags, posix = True)
        a.whitespace_split = True
        t = a.get_token()
        while t:
            t = parse_posix_cc_token(a, t)

    def parse_ldflags(flags):
        a = shlex.shlex(flags, posix = True)
        a.whitespace_split = True
        t = a.get_token()
        while t:
            t = parse_posix_ld_token(a, t)

    for i in [linux_x86_cflags, tiger_x86_cflags]:
        parse_cflags(i)

    for i in [linux_x86_ldflags]:
        parse_ldflags(i)
