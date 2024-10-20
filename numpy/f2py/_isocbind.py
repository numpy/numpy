"""
ISO_C_BINDING maps for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# These map to keys in c2py_map, via forced casting for now, see gh-25229
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

# TODO: At some point these should be included, but then they'd need special
# handling in cfuncs.py e.g. needs[int64_t_from_pyobj] These are not very hard
# to add, since they all derive from the base `int_from_pyobj`, e.g. the way
# `short_from_pyobj` and others do
isoc_c2pycode_map = {
    'int': 'i',  # int
    'short int': 'h',  # short int
    'long': 'l',  # long int
    'long long': 'q',  # long long int
    'signed char': 'b',  # signed char
    'size_t': 'I',  # size_t (approx unsigned int)
    'int8_t': 'b',  # int8_t
    'int16_t': 'h',  # int16_t
    'int32_t': 'i',  # int32_t
    'int64_t': 'q',  # int64_t
    'int_least8_t': 'b',  # int_least8_t
    'int_least16_t': 'h',  # int_least16_t
    'int_least32_t': 'i',  # int_least32_t
    'int_least64_t': 'q',  # int_least64_t
    'int_fast8_t': 'b',  # int_fast8_t
    'int_fast16_t': 'h',  # int_fast16_t
    'int_fast32_t': 'i',  # int_fast32_t
    'int_fast64_t': 'q',  # int_fast64_t
    'intmax_t': 'q',  # intmax_t (approx long long)
    'intptr_t': 'q',  # intptr_t (approx long long)
    'ptrdiff_t': 'q',  # intptr_t (approx long long)
    'float': 'f',  # float
    'double': 'd',  # double
    'long double': 'g',  # long double
    'float _Complex': 'F',  # float  _Complex
    'double _Complex': 'D',  # double  _Complex
    'long double _Complex': 'D',  # very approximate complex
    '_Bool': 'i',  #  Bool but not really
    'char': 'c',   # char
}

iso_c2py_map = {
    'int': 'int',
    'short int': 'int',                 # forced casting
    'long': 'int',
    'long long': 'long',
    'signed char': 'int',           # forced casting
    'size_t': 'int',                # approx Python int
    'int8_t': 'int',                # forced casting
    'int16_t': 'int',               # forced casting
    'int32_t': 'int',
    'int64_t': 'long',
    'int_least8_t': 'int',          # forced casting
    'int_least16_t': 'int',         # forced casting
    'int_least32_t': 'int',
    'int_least64_t': 'long',
    'int_fast8_t': 'int',           # forced casting
    'int_fast16_t': 'int',          # forced casting
    'int_fast32_t': 'int',
    'int_fast64_t': 'long',
    'intmax_t': 'long',
    'intptr_t': 'long',
    'ptrdiff_t': 'long',
    'float': 'float',
    'double': 'double',
    'long double': 'float',         # forced casting
    'float _Complex': 'complex',     # forced casting
    'double _Complex': 'complex',
    'long double _Complex': 'complex', # forced casting
    '_Bool': 'bool',
    'char': 'bytes',                  # approx Python bytes
    'short int': 'int',                 # forced casting
    'long int': 'int',
    'long long int': 'long',
}

isoc_kindmap = {}
for fortran_type, c_type_dict in iso_c_binding_map.items():
    for c_type in c_type_dict.keys():
        isoc_kindmap[c_type] = fortran_type
