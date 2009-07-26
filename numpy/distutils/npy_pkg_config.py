from ConfigParser import SafeConfigParser, NoOptionError
import re

_VAR = re.compile('\$\{([a-zA-Z0-9_-]+)\}')

class FormatError(IOError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class LibraryInfo(object):
    def __init__(self, name, description, version, sections, vars, requires=None):
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version
        self._sections = sections
        self.vars = vars
            
    def sections(self):
        return self._sections.keys()

    def cflags(self, section="default"):
        return self.vars.interpolate(self._sections[section]['cflags'])

    def libs(self, section="default"):
        return self.vars.interpolate(self._sections[section]['libs'])

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
            self._init_parse_var(k, v)

    def _init_parse_var(self, name, value):
        self._re[name] = re.compile(r'\$\{%s\}' % name)
        self._re_sub[name] = value

    def interpolate(self, value):
        # Brute force: we keep interpolating until there is no '${var}' anymore
        # or until interpolated string is equal to input string
        def _interpolate(value):
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        while _VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue

        return value

    def variables(self):
        return self._raw_data.keys()

    # Emulate a dict to set/get variables values
    def __getitem__(self, name):
        return self._raw_data[name]

    def __setitem__(self, name, value):
        self._raw_data[name] = value
        self._init_parse_var(name, value)

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

def parse_config(filename):
    config = SafeConfigParser()
    n = config.read(filename)
    if not len(n) >= 1:
        raise IOError("Could not find file %s" % filename)

    # Parse meta and variables sections
    meta = parse_meta(config)

    vars = {}
    if config.has_section('variables'):
        for name, value in config.items("variables"):
            vars[name] = value

    # Parse "normal" sections
    secs = [s for s in config.sections() if not s in ['meta', 'variables']]
    sections = {}

    requires = {}
    for s in secs:
        d = {}
        if config.has_option(s, "requires"):
            requires[s] = config.get(s, 'requires')
            
        for name, value in config.items(s):
            d[name] = value
        sections[s] = d

    return meta, vars, sections, requires

def read_config(filename):
    def _read_config(f):
        meta, vars, sections, reqs = parse_config(f)
        # recursively add sections and variables of required libraries
        for rname, rvalue in reqs.items():
            nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))

            # Update var dict for variables not in 'top' config file
            for k, v in nvars.items():
                if not vars.has_key(k):
                    vars[k] = v

            # Update sec dict
            for oname, ovalue in nsections[rname].items():
                sections[rname][oname] += ' %s' % ovalue

        return meta, vars, sections, reqs

    meta, vars, sections, reqs = _read_config(filename)

    return LibraryInfo(name=meta["name"], description=meta["description"],
            version=meta["version"], sections=sections, vars=VariableSet(vars))

# TODO:
#   - implements version comparison (modversion + atleast)

# Trivial cache to cache LibraryInfo instances creation. To be really
# efficient, the cache should be handled in read_config, since a same file can
# be parsed many time outside LibraryInfo creation, but I doubt this will be a
# problem in practice
_CACHE = {}
def get_info(pkgname):
    try:
        return _CACHE[pkgname]
    except KeyError:
        v = read_config(pkg_to_filename(pkgname))
        _CACHE[pkgname] = v
        return v

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
    info = get_info(pkg_name)
    
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
