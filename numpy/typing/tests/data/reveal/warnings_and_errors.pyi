import numpy.exceptions as ex

reveal_type(ex.ModuleDeprecationWarning())  # E: ModuleDeprecationWarning
reveal_type(ex.VisibleDeprecationWarning())  # E: VisibleDeprecationWarning
reveal_type(ex.ComplexWarning())  # E: ComplexWarning
reveal_type(ex.RankWarning())  # E: RankWarning
reveal_type(ex.TooHardError())  # E: TooHardError
reveal_type(ex.AxisError("test"))  # E: AxisError
reveal_type(ex.AxisError(5, 1))  # E: AxisError
