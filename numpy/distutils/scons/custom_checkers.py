# Sensible default for common types on common platforms.
_DEFAULTS = {
   'short' : [2,],
   'int' : [4, 2],
   'long' : [4, 8],
   'long long' : [8, 4],
   # Normally, there is no need to check unsigned types, because they are
   # guaranteed to be of the same size than their signed counterpart.
   'unsigned short' : [2,],
   'unsigned int' : [4, 2],
   'unsigned long' : [4, 8],
   'unsigned long long' : [8, 4],
   'float' : [4,],
   'double' : [8,],
   'long double' : [12,],
   'size_t' : [4,],
}

def CheckTypeSize(context, type, includes = None, language = 'C', size = None):
   """This check can be used to get the size of a given type, or to check whether
   the type is of expected size.

   Arguments:
       - type : str
           the type to check
       - includes : sequence
           list of headers to include in the test code before testing the type
       - language : str
           'C' or 'C++'
       - size : int
           if given, will test wether the type has the given number of bytes.
           If not given, will test against a list of sizes (all sizes between
           0 and 16 bytes are tested).

       Returns:
           status : int
               0 if the check failed, or the found size of the type if
               the check succeeded."""
   minsz = 0
   maxsz = 16

   if includes:
       src = "\n".join([r"#include <%s>\n" % i for i in includes])
   else:
       src = ""

   if language == 'C':
       ext = '.c'
   elif language == 'C++':
       ext = '.cpp'
   else:
       raise NotImplementedError("%s is not a recognized language" % language)

   # test code taken from autoconf: this is a pretty clever hack to find that
   # a type is of a given size using only compilation. This speeds things up
   # quite a bit compared to straightforward code using TryRun
   src += r"""
typedef %s scons_check_type;

int main()
{
   static int test_array[1 - 2 * !(((long int) (sizeof(scons_check_type))) <= %d)];
   test_array[0] = 0;

   return 0;
}
"""

   if size:
       # Only check if the given size is the right one
       context.Message('Checking %s is %d bytes... ' % (type, size))
       st = context.TryCompile(src % (type, size), ext)
       context.Result(st)

       if st:
           return size
       else:
           return 0
   else:
       # Check against a list of sizes.
       context.Message('Checking size of %s ... ' % type)

       # Try sensible defaults first
       try:
           szrange = _DEFAULTS[type]
       except KeyError:
           szrange = []
       szrange.extend(xrange(minsz, maxsz))
       st = 0

       # Actual test
       for sz in szrange:
           st = context.TryCompile(src % (type, sz), ext)
           if st:
               break

       if st:
           context.Result('%d' % sz)
           return sz
       else:
           context.Result('Failed !')
           return 0
