#! /usr/bin/env python
# Last Change: Mon Oct 29 06:00 PM 2007 J

class opt_info:
    def __init__(self, name, site = 0):
        """If not available, set name to ''."""
        self.name = name
        if len(name) > 0:
            self.available = 1
        else:
            self.available = 0
        self.site = site

    def __str__(self):
        if self.available:
            if self.site:
                msg = ['Tweaked from site.cfg']
            else:
                msg = ['Use %s' % self.name]
        else:
            msg = ['None available']

        return '\n'.join(msg)

    def __repr__(self):
        return self.__str__()

def add_info(env, name, opt):
    cfg = env['NUMPY_PKG_CONFIG']
    cfg[name] = opt

def write_info(env):
    print "File is %s" % env['NUMPY_PKG_CONFIG_FILE']
    print "Info is %s" % env['NUMPY_PKG_CONFIG']
    f = open(env['NUMPY_PKG_CONFIG_FILE'], 'w')
    f.writelines("config = %s" % str(env['NUMPY_PKG_CONFIG']))
    f.close()


