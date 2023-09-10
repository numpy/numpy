"""
Due to compatibility, numpy has a very large number of different naming
conventions for the scalar types (those subclassing from `numpy.generic`).
This file produces a convoluted set of dictionaries mapping names to types,
and sometimes other mappings too.

.. data:: allTypes
    A dictionary of names to types that will be exposed as attributes through
    ``np.core.numerictypes.*``

.. data:: sctypeDict
    Similar to `allTypes`, but maps a broader set of aliases to their types.

.. data:: sctypes
    A dictionary keyed by a "type group" string, providing a list of types
    under that group.

"""

import numpy.core.multiarray as ma
from numpy.core.multiarray import typeinfo, dtype

######################################
# Building `sctypeDict` and `allTypes`
######################################

sctypeDict = {}
allTypes = {}

_abstract_type_names = {
    "generic", "integer", "inexact", "floating", "number",
    "flexible", "character", "complexfloating", "unsignedinteger",
    "signedinteger"
}

for _abstract_type_name in _abstract_type_names:
    allTypes[_abstract_type_name] = getattr(ma, _abstract_type_name)

for k, v in typeinfo.items():
    concrete_type = v.type
    allTypes[k] = concrete_type
    sctypeDict[k] = concrete_type

_aliases = {
    "double": "float64",
    "cdouble": "complex128",
    "single": "float32",
    "csingle": "complex64",
    "half": "float16"
}

for k, v in _aliases.items():
    sctypeDict[k] = allTypes[v]
    allTypes[k] = allTypes[v]

# extra aliases are added only to `sctypeDict`
# to support dtype name access, such as`np.dtype("float")`
_extra_aliases = {  
    "bool": "bool_",
    "float": "float64",
    "complex": "complex128",
    "object": "object_",
    "bytes": "bytes_",
    "int": "int64",
    "long": "int64",
    "ulong": "uint",
    "str": "str_",
}

for k, v in _extra_aliases.items():
    sctypeDict[k] = allTypes[v]

####################
# Building `sctypes`
####################

sctypes = {"int": [], "uint": [], "float": [], "complex": [], "others": []}

for type_info in set(typeinfo.values()):
    if type_info.kind in ["M", "m"]:  # exclude timedelta and datetime
        continue

    concrete_type = type_info.type

    # find proper group for each concrete type
    for type_group, abstract_type in [
        ("int", ma.signedinteger), ("uint", ma.unsignedinteger), 
        ("float", ma.floating), ("complex", ma.complexfloating), 
        ("others", ma.generic)
    ]:
        if issubclass(concrete_type, abstract_type):
            sctypes[type_group].append(concrete_type)
            break

# sort sctype groups by bitsize
for sctype_list in sctypes.values():
    sctype_list.sort(key=lambda x: dtype(x).itemsize)
