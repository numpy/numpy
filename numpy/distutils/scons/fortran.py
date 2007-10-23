#! Last Change: Tue Oct 23 08:00 PM 2007 J

# This module defines some functions/classes useful for testing fortran-related
# features (name mangling, F77/C runtime, etc...).

# KEEP THIS INDEPENDENT OF SCONS, PLEASE !!!

import sys
import re

GCC_DRIVER_LINE = re.compile('^Driving:')
POSIX_STATIC_EXT = re.compile('\S+\.a')
POSIX_LIB_FLAGS = re.compile('-l\S+')

def _check_link_verbose_posix(lines):
    """Returns true if useful link options can be found in output.

    Expect lines to be a list of lines."""
    for line in lines:
        if not GCC_DRIVER_LINE.search(line):
            #print line
            #print POSIX_STATIC_EXT.search(line) 
            #print POSIX_LIB_FLAGS.search(line)
            if POSIX_STATIC_EXT.search(line) or POSIX_LIB_FLAGS.search(line):
                return True
    return False

def check_link_verbose(lines):
    if sys.platform == 'win32':
        raise NotImplementedError("FIXME: not implemented on win32")
    else:
        return _check_link_verbose_posix(lines)

merge_space_r1 = re.compile('^-[LRuYz]$')

def merge_space(line):
    """matcher should be a callable, line a list of tokens."""
    nline = []
    for i in range(len(line)):
        ##print "hoho is %s" % line[i]
        if merge_space_r1.match(line[i]):
            ##print '%s matched !' % line[i]
            merged = [line[i]]
            if not (line[i+1][0] == '-'):
                merged.append(line[i+1])
                i += 1
            nline.append(''.join(merged))
            ##print '\t%s matched !' % ''.join(merged)
        else:
            nline.append(line[i])
    return nline

def homo_libpath_flags(flags):
    nflags = []
    #print 'flags is %s' % flags
    #print "len flags is %d" % len(flags)
    for i in flags:
        if i[:4] == "-YP,":
            i = i.replace('-YP,', '-L')
            i = i.replace(':', ' -L')
            nflags.append(i)
        else:
            nflags.append(i)
    return nflags

def parse_f77link(lines):
    """Given the output of verbose link of F77 compiler, this returns a list of
    flags necessary for linking using the standard linker."""
    # TODO: On windows ?
    # TODO: take into account quotting...
    # TODO: those regex are really bad... Should try to get as similar as
    # possible to autoconf here.
    # XXX: this is really messy, clean it up !
    ignored = ['-lang*', r'-lcrt[a-zA-Z0-9]*\.o', '-lc', '-lgcc*',
               '-lSystem', '-libmil', '-LIST:*', '-LNO:*']
    inter = ['-[lLR][a-zA-Z0-9]*']
    # Those options takes an argument, so concatenate any following item
    # until the end of the line or a new option.
    remove_space = ['-[LRuYz]*']
    import re
    rignored = [re.compile(i) for i in ignored]
    rinter = [re.compile(i) for i in inter]
    # We ignore lines starting with Driving
    rgccignored = re.compile('^Driving:')
    final_flags = []
    for line in lines:
        # Here we go (convention for wildcard is shell, not regex !)
        #   1 TODO: we first get some root .a libraries
        #   2 TODO: take everything starting by -bI:*
        #   3 TODO: ignore the following flags: -lang* | -lcrt*.o | -lc |
        #   -lgcc* | -lSystem | -libmil | -LANG:=* | -LIST:* | -LNO:*)
        #   4 TODO: take into account -lkernel32
        #   5 For options of the kind -[[LRuYz]], as they take one argument
        #   after, we have to somewhat keep it. We do as autoconf, that is
        #   removing space between the flag and its argument.
        #   6 For -YP,*: take and replace by -Larg where arg is the old argument
        #   7 For -[lLR]*: take
        if not rgccignored.match(line):
            # Step 5
            flags = merge_space(line.split())
            # Step 6
            ##print 'homo flags are: %s (%d items)' % (flags, len(flags))
            flags = homo_libpath_flags(flags)
            #print 'homo flags are: %s ' % flags
            def match_ignore(str):
                if [i for i in rignored if i.match(str)]:
                    return True
                else:
                    return False

            def match_interesting(str):
                pop = [i for i in rinter if i.match(str)]
                if pop:
                    #print "pop %s" % str
                    return True
                else:
                    return False

            # Step 3
            good_flags = [i for i in flags if not match_ignore(i)]
            #print good_flags
            good_flags = [i for i in good_flags if match_interesting(i)]
            final_flags.extend(good_flags)
    return final_flags

if __name__ == '__main__':
    pass
