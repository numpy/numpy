"""
ISO_FORTRAN_ENV maps for f2py2e
"""
iso_fortran_env_map: dict[str, dict[str, str]] = {
    'integer': {
        'int8': 'signed_char',
        'int16': 'short',
        'int32': 'int',
        'int64': 'long_long',
    },
    'real': {
        'real32': 'float',
        'real64': 'double',
    }
}

isof_kindmap: dict[str, str] = {}
for _fortran_type, _std_type_dict in iso_fortran_env_map.items():
    for _std_type in _std_type_dict.keys():
        isof_kindmap[_std_type] = _fortran_type
