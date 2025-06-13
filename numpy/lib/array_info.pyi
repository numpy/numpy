from typing import Any, Dict, Union
import numpy as np

def array_info(
    arr: Union[np.ndarray, Any],
    *,
    sample_unique_threshold: int = ...,
    example_values: int = ...,
) -> Dict[str, Any]: ...
