#! /usr/bin/env python
# Last Change: Mon Oct 29 06:00 PM 2007 J
import os

def add_info(env, name, opt):
    cfg = env['NUMPY_PKG_CONFIG']
    cfg[name] = opt

def write_info(env):
    print "File is %s" % env['NUMPY_PKG_CONFIG_FILE']
    print "Info is %s" % env['NUMPY_PKG_CONFIG']
    dir = os.path.dirname(env['NUMPY_PKG_CONFIG_FILE'])
    if not os.path.exists(dir):
        os.makedirs()
    f = open(env['NUMPY_PKG_CONFIG_FILE'], 'w')
    f.writelines("config = %s" % str(env['NUMPY_PKG_CONFIG']))
    f.close()


