from numpyenv import GetNumpyEnvironment, GetNumpyOptions
from libinfo_scons import NumpyCheckLib

# XXX: all this should be put in another files eventually once it is getting in
# shape

def _get_empty(dict, key):
    print "++++++ Deprecated, do not use _get_empty +++++++++"
    try:
        return dict[key]
    except KeyError, e:
        return []

def cfgentry2list(entry):
    """This convert one entry in a section of .cfg file to something usable in
    scons."""
    pass
