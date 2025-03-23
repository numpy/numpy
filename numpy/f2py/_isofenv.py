"""
ISO_FORTRAN_ENV maps for f2py2e
"""
iso_fortran_env_map = {
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

isof_f2pycode_map = {}
iso_c2py_map = {}

isof_kindmap = {}
for fortran_type, std_type_dict in iso_fortran_env_map.items():
    for std_type in std_type_dict.keys():
        isof_kindmap[std_type] = fortran_type
