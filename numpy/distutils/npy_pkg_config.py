from ConfigParser import SafeConfigParser, NoOptionError
import re

VAR = re.compile('\$\{([a-zA-Z0-9_-]+)\}')

cfg = 'config.ini'

config = SafeConfigParser()
n = config.read(cfg)
#print "Reading: ", n

class FormatError(IOError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class MetaInfo(object):
    def __init__(self, name, description, version, requires=None):
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version

    def __str__(self):
        m = ['Name: %s' % self.name]
        m.append('Description: %s' % self.description)
        if self.requires:
            m.append('Requires:')
        else:
            m.append('Requires: %s' % ",".join(self.requires))

        return "\n".join(m)


class PathsInfo(object):
    def __init__(self, prefix=None, exec_prefix=None, libdir=None, includedir=None):
        self._raw_data = {}
        if prefix:
            self._raw_data['prefix'] = prefix
        if exec_prefix:
            self._raw_data['exec_prefix'] = exec_prefix
        if libdir:
            self._raw_data['libdir'] = libdir
        if includedir:
            self._raw_data['includedir'] = includedir

        self._re = {}
        self._re_sub = {}

        self._init_parse()

    def _init_parse(self):
        for k, v in self._raw_data.items():
            self._re[k] = re.compile(r'\$\{%s\}' % k)
            self._re_sub[k] = v

    def interpolate(self, value):
        # Brute force: we keep interpolating until there is no '${var}' anymore
        # or until interpolated string is equal to input string
        def _interpolate(value):
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        while VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue

        return value

def parse_meta(config):
    if not config.has_section('meta'):
        raise FormatError("No meta section found !")

    d = {}
    for name, value in config.items('meta'):
        d[name] = value

    for k in ['name', 'description', 'version']:
        if not d.has_key(k):
            raise FormatError("Option %s (section [meta]) is mandatory, "
                "but not found" % k)

    if not d.has_key('requires'):
        d['requires'] = None

    return MetaInfo(name=d['name'], description=d['description'], version=d['version'],
            requires=d['requires'])

def parse_paths(config):
    if not config.has_section('default'):
        raise FormatError("No default section found !")

    d = {}

    paths = ['prefix', 'exec_prefix', 'libdir', 'includedir']
    for p in paths:
        try:
            d[p] = config.get('default', p)
        except NoOptionError:
            pass

    return PathsInfo(**d)

meta = parse_meta(config)
paths_info = parse_paths(config)

def get_libs(config, paths_info):
    l = config.get('default', 'Libs')
    return paths_info.interpolate(l)

def get_cflags(config, paths_info):
    c = config.get('default', 'Cflags')
    return paths_info.interpolate(c)

def get_version(meta):
    ver = meta.version
    print ver

print get_libs(config, paths_info)
print get_cflags(config, paths_info)
print get_version(meta)
#print config.items('default')
