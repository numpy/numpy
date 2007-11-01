#! /usr/bin/env python
# Last Change: Thu Nov 01 06:00 PM 2007 J
import os

def add_info(env, name, opt):
    cfg = env['NUMPY_PKG_CONFIG']
    cfg[name] = str(opt)

def write_info(env):
    dir = os.path.dirname(env['NUMPY_PKG_CONFIG_FILE'])
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = open(env['NUMPY_PKG_CONFIG_FILE'], 'w')
    f.writelines("config = %s" % str(env['NUMPY_PKG_CONFIG']))
    f.close()
