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
