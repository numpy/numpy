# size to try first for built-in types
_PREF_SIZE = {
    'short' : (2,),
    'int' : (4,),
    'long' : (4,8),
    'long long' : (8,4),
    'float' : (4,),
    'double' : (8,),
    'long double' : (12,),
    'size_t' : (4,),
    'Py_intptr_t' : (4,),
}

def CheckSizeof(context, type, include = None, language = None):
    """include should be a list of header."""
    if language:
        raise NotImplementedError("language arg not supported yet!")

    msg = context.Message('Checking size of type %s ... ' % type)
    if include:
        strinc = ["#include <%s>" % s for s in include]
        src = "\n".join(strinc)
    else:
        src = ""
    src += """
typedef %s check_sizeof;

int main()
{
    static int test_array[1 - 2 * !( (long int) (sizeof(check_sizeof)) <= %d)];
    test_array[0] = 0;

    return 0;
}
"""

    st = 0
    # First try sensible default
    if _PREF_SIZE.has_key(type):
        fs = _PREF_SIZE[type]
        for i in fs:
            st = context.TryCompile(src % (type, i), '.c')
            if st:
                break
    # General 
    maxsize = 16
    if not st:
        for i in xrange(0, maxsize):
            st = context.TryCompile(src % (type, i), '.c')
            if st:
                break

    if st:
        context.Result('%d bytes' % i)
    else:
        context.Result('Failed !')

    return st

