import numpy as np

reveal_type(np)  # E: ModuleType

reveal_type(np.char)  # E: ModuleType
reveal_type(np.ctypeslib)  # E: ModuleType
reveal_type(np.emath)  # E: ModuleType
reveal_type(np.fft)  # E: ModuleType
reveal_type(np.lib)  # E: ModuleType
reveal_type(np.linalg)  # E: ModuleType
reveal_type(np.ma)  # E: ModuleType
reveal_type(np.matrixlib)  # E: ModuleType
reveal_type(np.polynomial)  # E: ModuleType
reveal_type(np.random)  # E: ModuleType
reveal_type(np.rec)  # E: ModuleType
reveal_type(np.testing)  # E: ModuleType
reveal_type(np.version)  # E: ModuleType

# TODO: Remove when annotations have been added to `np.testing.assert_equal`
reveal_type(np.testing.assert_equal)  # E: Any

reveal_type(np.__all__)  # E: list[builtins.str]
reveal_type(np.__path__)  # E: list[builtins.str]
reveal_type(np.__version__)  # E: str
reveal_type(np.__git_version__)  # E: str
reveal_type(np.__NUMPY_SETUP__)  # E: bool
reveal_type(np.__deprecated_attrs__)  # E: dict[builtins.str, Tuple[builtins.type, builtins.str]]
reveal_type(np.__expired_functions__)  # E: dict[builtins.str, builtins.str]
