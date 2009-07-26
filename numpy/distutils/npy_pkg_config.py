from ConfigParser import SafeConfigParser, NoOptionError
import re

VAR = re.compile('\$\{([a-zA-Z0-9_-]+)\}')

class FormatError(IOError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class LibraryInfo(object):
    def __init__(self, name, description, version, sections, requires=None):
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version
        self._sections = sections
            
    def sections(self):
        return self._sections.keys()

    def cflags(self, section="default"):
        if not self._sections[section].has_key("cflags"):
            return ""
        return self._sections[section]["cflags"]

    def libs(self, section="default"):
        if not self._sections[section].has_key("libs"):
            return ""
        return self._sections[section]["libs"]

    def __str__(self):
        m = ['Name: %s' % self.name]
        m.append('Description: %s' % self.description)
        if self.requires:
            m.append('Requires:')
        else:
            m.append('Requires: %s' % ",".join(self.requires))
        m.append('Version: %s' % self.version)

        return "\n".join(m)

class VariableSet(object):
    def __init__(self, d):
        self._raw_data = dict([(k, v) for k, v in d.items()])

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
        d['requires'] = []

    return d

def parse_variables(config):
    if not config.has_section('variables'):
        raise FormatError("No variables section found !")

    d = {}

    for name, value in config.items("variables"):
        d[name] = value

    return VariableSet(d)

def parse_sections(config):
    return meta_d, r

def pkg_to_filename(pkg_name):
    return "%s.ini" % pkg_name

# TODO:
#   - implements --cflags, --libs
#   - implements version comparison (modversion + atleast)
#   - implements non default section

def read_config(filename):
    config = SafeConfigParser()
    n = config.read(filename)
    if not len(n) >= 1:
        raise IOError("Could not find file %s" % filename)

    meta_d = parse_meta(config)
    varset = parse_variables(config)

    # Parse "normal" sections
    secs = config.sections()
    secs = [s for s in secs if not s in ["meta", "variables"]]

    r = {}

    # XXX: this is a mess
    # XXX: cache the LibraryInfo instances
    for s in secs:
        d = {}
        if config.has_option(s, "depends"):
            tmp = read_config(pkg_to_filename(config.get(s, "depends")))
        for name, value in config.items(s):
            d[name] = varset.interpolate(value)
            if config.has_option(s, "depends") and not name == "depends":
                if s in tmp.sections():
                    d[name] += ' %s' % getattr(tmp, name)(s)
        r[s] = d

    return LibraryInfo(name=meta_d["name"], description=meta_d["description"],
            version=meta_d["version"], sections=r)

if __name__ == '__main__':
    import sys
    from optparse import OptionParser
    import glob

    parser = OptionParser()
    parser.add_option("--cflags", dest="cflags", action="store_true",
                      help="output all preprocessor and compiler flags")
    parser.add_option("--libs", dest="libs", action="store_true",
                      help="output all linker flags")
    parser.add_option("--use-section", dest="section",
                      help="use this section instead of default for options")
    parser.add_option("--version", dest="version", action="store_true",
                      help="output version")
    parser.add_option("--atleast-version", dest="min_version",
                      help="Minimal version")
    parser.add_option("--list-all", dest="list_all", action="store_true",
                      help="Minimal version")

    (options, args) = parser.parse_args(sys.argv)

    if len(args) < 2:
        raise ValueError("Expect package name on the command line:")

    if options.list_all:
        files = glob.glob("*.ini")
        for f in files:
            info = read_config(f)
            print "%s\t%s - %s" % (info.name, info.name, info.description)
    pkg_name = args[1]
    fname = pkg_to_filename(pkg_name)
    info = read_config(fname)
    
    if options.section:
        section = options.section
    else:
        section = "default"

    if options.cflags:
        print info.cflags(section)
    if options.libs:
        print info.libs(section)
    if options.version:
        print info.version
    if options.min_version:
        print info.version >= options.min_version
