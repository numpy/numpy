import os
import re

import waflib.Configure
import waflib.Tools.c_config
from waflib import Logs, Utils

DEFKEYS = waflib.Tools.c_config.DEFKEYS
DEFINE_COMMENTS = "define_commentz"

def to_header(dct):
    if 'header_name' in dct:
        dct = Utils.to_list(dct['header_name'])
        return ''.join(['#include <%s>\n' % x for x in dct])
    return ''

# Make the given string safe to be used as a CPP macro
def sanitize_string(s):
    key_up = s.upper()
    return re.sub('[^A-Z0-9_]', '_', key_up)

def validate_arguments(self, kw):
    if not 'env' in kw:
        kw['env'] = self.env.derive()
    if not "compile_mode" in kw:
        kw["compile_mode"] = "c"
    if not 'compile_filename' in kw:
        kw['compile_filename'] = 'test.c' + \
                ((kw['compile_mode'] == 'cxx') and 'pp' or '')
    if not 'features' in kw:
        kw['features'] = [kw['compile_mode']]
    if not 'execute' in kw:
        kw['execute'] = False
    if not 'okmsg' in kw:
        kw['okmsg'] = 'yes'
    if not 'errmsg' in kw:
        kw['errmsg'] = 'no !'

def try_compile(self, kw):
    self.start_msg(kw["msg"])
    ret = None
    try:
        ret = self.run_c_code(**kw)
    except self.errors.ConfigurationError as e:
        self.end_msg(kw['errmsg'], 'YELLOW')
        if Logs.verbose > 1:
            raise
        else:
            self.fatal('The configuration failed')
    else:
        kw['success'] = ret
        self.end_msg(self.ret_msg(kw['okmsg'], kw))

@waflib.Configure.conf
def check_header(self, header_name, **kw):
    code = """
%s

int main()
{
}
""" % to_header({"header_name": header_name})

    kw["code"] = code
    kw["define_comment"] = "/* Define to 1 if you have the `%s' header. */" % header_name
    kw["define_name"] = "HAVE_%s" % sanitize_string(header_name)
    if not "features" in kw:
        kw["features"] = ["c"]
    kw["msg"] = "Checking for header %r" % header_name

    validate_arguments(self, kw)
    try_compile(self, kw)
    ret = kw["success"]

    self.post_check(**kw)
    if not kw.get('execute', False):
        return ret == 0
    return ret

@waflib.Configure.conf
def check_declaration(self, symbol, **kw):
    code = r"""
int main()
{
#ifndef %s
    (void) %s;
#endif
    ;
    return 0;
}
""" % (symbol, symbol)

    kw["code"] = to_header(kw) + code
    kw["msg"] = "Checking for macro %r" % symbol
    kw["errmsg"] = "not found"
    kw["okmsg"] = "yes"

    validate_arguments(self, kw)
    try_compile(self, kw)
    ret = kw["success"]

    kw["define_name"] = "HAVE_DECL_%s" % sanitize_string(symbol)
    kw["define_comment"] = "/* Set to 1 if %s is defined. */" % symbol
    self.post_check(**kw)
    if not kw.get('execute', False):
        return ret == 0
    return ret

def do_binary_search(conf, type_name, kw):
    code = """\
typedef %(type)s waf_check_sizeof_type;
int main ()
{
    static int test_array [1 - 2 * !(((long) (sizeof (waf_check_sizeof_type))) >= 0)];
    test_array [0] = 0

    ;
    return 0;
}
""" % {"type": type_name}
    kw["code"] = to_header(kw) + code

    try:
        conf.run_c_code(**kw)
    except conf.errors.ConfigurationError, e:
        conf.end_msg("failed !")
        if waflib.Logs.verbose > 1:
            raise
        else:
            conf.fatal("The configuration failed !")

    body = r"""
typedef %(type)s waf_check_sizeof_type;
int main ()
{
    static int test_array [1 - 2 * !(((long) (sizeof (waf_check_sizeof_type))) <= %(size)s)];
    test_array [0] = 0

    ;
    return 0;
}
"""
    # The principle is simple: we first find low and high bounds
    # of size for the type, where low/high are looked up on a log
    # scale. Then, we do a binary search to find the exact size
    # between low and high
    low = 0
    mid = 0
    while True:
        try:
            kw["code"] = to_header(kw) + body % {"type": type_name, "size": mid}
            validate_arguments(conf, kw)
            conf.run_c_code(**kw)
            break
        except conf.errors.ConfigurationError:
            #log.info("failure to test for bound %d" % mid)
            low = mid + 1
            mid = 2 * mid + 1

    high = mid
    ret = None
    # Binary search:
    while low != high:
        mid = (high - low) / 2 + low
        try:
            kw["code"] = to_header(kw) + body % {"type": type_name, "size": mid}
            validate_arguments(conf, kw)
            ret = conf.run_c_code(**kw)
            high = mid
        except conf.errors.ConfigurationError:
            low = mid + 1

    return low

@waflib.Configure.conf
def check_type_size(conf, type_name, expected_sizes=None, **kw):
    kw["define_name"] = "SIZEOF_%s" % sanitize_string(type_name)
    kw["define_comment"] = "/* The size of `%s', as computed by sizeof. */" % type_name
    kw["msg"] = "Checking sizeof(%s)" % type_name

    validate_arguments(conf, kw)
    conf.start_msg(kw["msg"])

    if expected_sizes is not None:
        try:
            val = int(expected_sizes)
        except TypeError:
            values = expected_sizes
        else:
            values = [val]

        size = None
        for value in values:
            code = """\
    typedef %(type)s waf_check_sizeof_type;
    int main ()
    {
        static int test_array [1 - 2 * !(((long) (sizeof (waf_check_sizeof_type))) == %(size)d)];
        test_array [0] = 0

        ;
        return 0;
    }
    """ % {"type": type_name, "size": value}
            kw["code"] = to_header(kw) + code
            try:
                conf.run_c_code(**kw)
                size = value
                break
            except conf.errors.ConfigurationError:
                pass
        if size is None:
            size = do_binary_search(conf, type_name, kw)
    else:
        size = do_binary_search(conf, type_name, kw)

    kw["define_value"] = size
    kw["success"] = 0
    conf.end_msg(size)
    conf.post_check(**kw)
    return size

