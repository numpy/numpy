import enum

__all__ = [
    'CopyMode'
    ]

class CopyMode(enum.Enum):

    ALWAYS = True
    IF_NEEDED = False
    NEVER = 2

    def __bool__(self):
        # For backwards compatiblity
        if self == CopyMode.ALWAYS:
            return True

        if self == CopyMode.IF_NEEDED:
            return False

        raise TypeError(f"{self} is neither True nor False.")


CopyMode.__module__ = 'numpy.array_api'
