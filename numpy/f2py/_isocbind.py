iso_c_binding_map = {
    'integer': {
        'c_int': 'int',
        'c_short': 'short int',
        'c_long': 'long int',
        'c_long_long': 'long long int',
        'c_signed_char': 'signed char',
        'c_size_t': 'size_t',
        'c_int8_t': 'int8_t',
        'c_int16_t': 'int16_t',
        'c_int32_t': 'int32_t',
        'c_int64_t': 'int64_t',
        'c_int_least8_t': 'int_least8_t',
        'c_int_least16_t': 'int_least16_t',
        'c_int_least32_t': 'int_least32_t',
        'c_int_least64_t': 'int_least64_t',
        'c_int_fast8_t': 'int_fast8_t',
        'c_int_fast16_t': 'int_fast16_t',
        'c_int_fast32_t': 'int_fast32_t',
        'c_int_fast64_t': 'int_fast64_t',
        'c_intmax_t': 'intmax_t',
        'c_intptr_t': 'intptr_t',
        'c_ptrdiff_t': 'intptr_t',
    },
    'real': {
        'c_float': 'float',
        'c_double': 'double',
        'c_long_double': 'long double'
    },
    'complex': {
        'c_float_complex': 'float _Complex',
        'c_double_complex': 'double _Complex',
        'c_long_double_complex': 'long double _Complex'
    },
    'logical': {
        'c_bool': '_Bool'
    },
    'character': {
        'c_char': 'char'
    }
}

isoc_kindmap = {}
for fortran_type, c_type_dict in iso_c_binding_map.items():
    for c_type in c_type_dict.keys():
        isoc_kindmap[c_type] = fortran_type
