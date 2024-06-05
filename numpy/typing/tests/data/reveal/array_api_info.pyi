import sys
from typing import List

import numpy as np

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

array_namespace_info = np.__array_namespace_info__()

assert_type(array_namespace_info.__module__, str)
assert_type(array_namespace_info.capabilities(), np._array_api_info.Capabilities)
assert_type(array_namespace_info.default_device(), str)
assert_type(array_namespace_info.default_dtypes(), np._array_api_info.DefaultDataTypes)
assert_type(array_namespace_info.dtypes(), np._array_api_info.DataTypes)
assert_type(array_namespace_info.devices(), List[str])
