from typing import Dict, Type, List, TypedDict

from numpy import generic, signedinteger, unsignedinteger, floating, complexfloating

class _SCTypes(TypedDict):
    int: List[Type[signedinteger]]
    uint: List[Type[unsignedinteger]]
    float: List[Type[floating]]
    complex: List[Type[complexfloating]]
    others: List[type]

sctypeDict: Dict[int | str, Type[generic]]
sctypes: _SCTypes
