#! /usr/bin/env python
# Last Change: Thu Nov 22 12:00 PM 2007 J
import os

def add_info(env, name, opt):
    cfg = env['NUMPY_PKG_CONFIG']
    cfg[name] = opt

def write_info(env):
    dir = os.path.dirname(env['NUMPY_PKG_CONFIG_FILE'])
    if not os.path.exists(dir):
        os.makedirs(dir)
    cfg = env['NUMPY_PKG_CONFIG']
    config_str = {}
    for k, i in cfg.items():
        config_str[k] = str(i)
    f = open(env['NUMPY_PKG_CONFIG_FILE'], 'w')
    f.writelines("config = %s" % str(config_str))
    f.close()
