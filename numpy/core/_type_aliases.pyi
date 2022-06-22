from typing import TypedDict

from numpy import complexfloating, floating, generic, signedinteger, unsignedinteger

class _SCTypes(TypedDict):
    int: list[type[signedinteger]]
    uint: list[type[unsignedinteger]]
    float: list[type[floating]]
    complex: list[type[complexfloating]]
    others: list[type]

sctypeDict: dict[int | str, type[generic]]
sctypes: _SCTypes
