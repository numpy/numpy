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

