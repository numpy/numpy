#! Last Change: Mon Nov 12 07:00 PM 2007 J

# This module defines some functions/classes useful for testing fortran-related
# features (name mangling, F77/C runtime, etc...).

# KEEP THIS INDEPENDENT OF SCONS, PLEASE !!!

import sys
import re
import os

GCC_DRIVER_LINE = re.compile('^Driving:')
POSIX_STATIC_EXT = re.compile('\S+\.a')
POSIX_LIB_FLAGS = re.compile('-l\S+')
MERGE_SPACE_R1 = re.compile('^-[LRuYz]$')

# linkflags which match those are ignored
LINKFLAGS_IGNORED = [r'-lang*', r'-lcrt[a-zA-Z0-9]*\.o', r'-lc', r'-lSystem', r'-libmil', r'-LIST:*', r'-LNO:*']
if os.name == 'nt':
    LINKFLAGS_IGNORED.extend([r'-lfrt*', r'-luser32',
	    r'-lkernel32', r'-ladvapi32', r'-lmsvcrt',
	    r'-lshell32', r'-lmingw', r'-lmoldname'])
else:
    LINKFLAGS_IGNORED.append(r'-lgcc*')

RLINKFLAGS_IGNORED = [re.compile(i) for i in LINKFLAGS_IGNORED]

# linkflags which match those are the one we are interested in
LINKFLAGS_INTERESTING = [r'-[lLR][a-zA-Z0-9]*']
RLINKFLAGS_INTERESTING = [re.compile(i) for i in LINKFLAGS_INTERESTING]

def gnu_to_ms_link(linkflags):
    # XXX: This is bogus. Instead of manually playing with those flags, we
    # should use scons facilities, but this is not so easy because we want to
    # use posix environment and MS environment at the same time
    newflags = []
    for flag in linkflags:
        if flag.startswith('-L'):
            newflags.append('/LIBPATH:%s' i[2:])
        elif flag.startswith('-l'):
            newflags.append('lib%s.a' % i[2:])
    return newflags

def _check_link_verbose_posix(lines):
    """Returns true if useful link options can be found in output.

    POSIX implementation.

    Expect lines to be a list of lines."""
    for line in lines:
        if not GCC_DRIVER_LINE.search(line):
            if POSIX_STATIC_EXT.search(line) or POSIX_LIB_FLAGS.search(line):
                return True
    return False

def check_link_verbose(lines):
    """Return true if useful link option can be found in output."""
    if sys.platform == 'win32':
        raise NotImplementedError("FIXME: not implemented on win32")
    else:
        return _check_link_verbose_posix(lines)

def merge_space(line):
    """For options taking an argument, merge them to avoid spaces.
    
    line should be a list of tokens."""
    nline = []
    for i in range(len(line)):
        if MERGE_SPACE_R1.match(line[i]):
            merged = [line[i]]
            if not (line[i+1][0] == '-'):
                merged.append(line[i+1])
                i += 1
            nline.append(''.join(merged))
        else:
            nline.append(line[i])
    return nline

def homo_libpath_flags(flags):
    """For arguments like -YP, transform them into -L."""
    nflags = []
    for i in flags:
        if i[:4] == "-YP,":
            i = i.replace('-YP,', '-L')
            i = i.replace(':', ' -L')
            nflags.extend(i.split(' '))
        else:
            nflags.append(i)
    return nflags

def match_ignore(str):
    if [i for i in RLINKFLAGS_IGNORED if i.match(str)]:
        return True
    else:
        return False

def match_interesting(str):
    pop = [i for i in RLINKFLAGS_INTERESTING if i.match(str)]
    if pop:
        return True
    else:
        return False

def parse_f77link(lines):
    """Given the output of verbose link of F77 compiler, this returns a list of
    flags necessary for linking using the standard linker."""
    # TODO: On windows ?
    # TODO: take into account quotting...
    # Those options takes an argument, so concatenate any following item
    # until the end of the line or a new option.
    remove_space = ['-[LRuYz]*']
    final_flags = []
    for line in lines:
        # Here we go (convention for wildcard is shell, not regex !)
        #   1 TODO: we first get some root .a libraries
        #   2 TODO: take everything starting by -bI:*
        #   3 Ignore the following flags: -lang* | -lcrt*.o | -lc |
        #   -lgcc* | -lSystem | -libmil | -LANG:=* | -LIST:* | -LNO:*)
        #   4 TODO: take into account -lkernel32
        #   5 For options of the kind -[[LRuYz]], as they take one argument
        #   after, we have to somewhat keep it. We do as autoconf, that is
        #   removing space between the flag and its argument.
        #   6 For -YP,*: take and replace by -Larg where arg is the old argument
        #   7 For -[lLR]*: take
        if not GCC_DRIVER_LINE.match(line):
            flags = line.split()

            # Step 3
            flags = [i for i in flags if not match_ignore(i)]

            # Step 5
            flags = merge_space(flags)

            # Step 6
            flags = homo_libpath_flags(flags)

            # Step 7
            good_flags = [i for i in flags if match_interesting(i)]

            final_flags.extend(good_flags)
    return final_flags

if __name__ == '__main__':
    pass
