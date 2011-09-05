"""
Function for importing data and auto-type determination.

New types can be added by adding a regular expression for the type to
dtype_to_re and quote_dtype_to_re, its promotion rules to dtype_promotion,
the relation between its name and the corresponding dtype to dtype_dict,
a converter to dtype_to_conv, and a default missing value to
dtype_default_missing. With some minor tweaking this should allow new
types to be added relatively painlessly.
"""
import re
import numpy as np
import datetime as dt



# Dictionary mapping dtype name to regular expression
# matching that dtype, assuming data isn't quoted
dtype_to_re = {
    'bool': r'(?:True|False|true|false|TRUE|FALSE)',
    'int32': r'[+-]?\d+',
    'int64': r'[+-]?\d+',
    'float32': r'[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:[eE][-+]?\d+)?',
    'float64': r'[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:[eE][-+]?\d+)?',
    'float32InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)',
                              r'(?:[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf|',
                              r'[+-]?Inf|[+-]?INF)']),
    'float64InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)',
                              r'(?:[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf|',
                              r'[+-]?Inf|[+-]?INF)']),
    'commafloat32': r'[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)(?:[eE][-+]?\d+)?',
    'commafloat64': r'[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)(?:[eE][-+]?\d+)?',
    'commafloat32InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)',
                                   r'(?:[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf',
                                   r'|[+-]?Inf|[+-]?INF)']),
    'commafloat64InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)',
                                   r'(?:[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf',
                                   r'|[+-]?Inf|[+-]?INF)']),
   'complex64': ''.join([r'(?:[-+]?(?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee][-+]?',
                         r'\d+)?)?[jJ]|[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                         r'[-+]?\d+)?[-+](?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                         r'[-+]?\d+)?)?[jJ])']),
    'complex128': ''.join([r'(?:[-+]?(?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee][-+]?',
                           r'\d+)?)?[jJ]|[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                           r'[-+]?\d+)?[-+](?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                           r'[-+]?\d+)?)?[jJ])']),
    'datetime64[D]': r'\d{4}-\d{2}-\d{2}',
    '|S1': r'\S+',
    }

# Double check that the regular expressions are non-capturing.
# This is required for loadtable's regular expression
# matching to work correctly.
for key in dtype_to_re.keys():
    if re.compile(dtype_to_re[key]).groups >0:
        s = ''.join(['dtype_to_re: ',
                    'type regular expression for ',
                    key,
                    ' must be non-capturing'])
        raise ValueError(s)

# Dictionary mapping dtype name to regular expression
# matching that dtype, allowing the data to be quoted
quoted_dtype_to_re = {
    'bool': ''.join([r'(?:True|False|true|false|TRUE|FALSE|"True"|"False"|',
                      r'"true"|"false"|"TRUE"|"FALSE")']),
    'int32': r'(?:[+-]?\d+|"[+-]?\d+")',
    'int64': r'(?:[+-]?\d+|"[+-]?\d+")',
    'float32': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:[eE][-+]',
                        r'?\d+)?|"[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:',
                        r'[eE][-+]?\d+)?")']),
    'float64': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:[eE][-+]',
                        r'?\d+)?|"[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:',
                        r'[eE][-+]?\d+)?")']),
    'float32InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:',
                              r'[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf|[+-]?',
                              r'INF|[+-]?Inf|"[+-]?\s*(?:(?:(?:\d+)?\.)?\d',
                              r'+"|"\d+\.)(?:[eE][-+]?\d+)?"|"nan"|"NAN"|',
                              r'"NaN"|"[+-]?inf"|"[+-]?INF"|"[+-]?Inf")']),
    'float64InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?\.)?\d+|\d+\.)(?:',
                              r'[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf|[+-]?',
                              r'INF|[+-]?Inf|"[+-]?\s*(?:(?:(?:\d+)?\.)?\d',
                              r'+"|"\d+\.)(?:[eE][-+]?\d+)?"|"nan"|"NAN"|',
                              r'"NaN"|"[+-]?inf"|"[+-]?INF"|"[+-]?Inf")']),
    'commafloat32': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)(?:[eE]',
                             r'[-+]?\d+)?|"[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d',
                             r'+,)(?:[eE][-+]?\d+)?")']),
    'commafloat64': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)(?:[eE]',
                             r'[-+]?\d+)?|"[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d',
                             r'+,)(?:[eE][-+]?\d+)?")']),
    'commafloat32InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)',
                                   r'(?:[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf|',
                                   r'[+-]?INF|[+-]?Inf|"[+-]?\s*(?:(?:(?:\d',
                                   r'+)?,)?\d+"|"\d+,)(?:[eE][-+]?\d+)?"|',
                                   r'"nan"|"NAN"|"NaN"|"[+-]?inf"|"[+-]?INF"',
                                   r'|"[+-]?Inf")']),
    'commafloat64InfNaN': ''.join([r'(?:[+-]?\s*(?:(?:(?:\d+)?,)?\d+|\d+,)',
                                   r'(?:[eE][-+]?\d+)?|nan|NAN|NaN|[+-]?inf|',
                                   r'[+-]?INF|[+-]?Inf|"[+-]?\s*(?:(?:(?:\d',
                                   r'+)?,)?\d+"|"\d+,)(?:[eE][-+]?\d+)?"|',
                                   r'"nan"|"NAN"|"NaN"|"[+-]?inf"|"[+-]?INF"',
                                   r'|"[+-]?Inf")']),
    'complex64': ''.join([r'(?:[-+]?(?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee][-+]?',
                          r'\d+)?)?[jJ]|[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                          r'[-+]?\d+)?[-+](?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                          r'[-+]?\d+)?)?[jJ]|"[-+]?(?:(?:\d+\.?\d*|\d*\.?\d',
                          r'+)(?:[Ee][-+]?\d+)?)?[j]"|"[-+]?(?:\d+\.?\d*|\d',
                          r'*\.?\d+)(?:[Ee][-+]?\d+)?[-+](?:(?:\d+\.?\d*|\d',
                          r'*\.?\d+)(?:[Ee][-+]?\d+)?)?[jJ]")']),
    'complex128': ''.join([r'(?:[-+]?(?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee][-+]?',
                          r'\d+)?)?[jJ]|[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                          r'[-+]?\d+)?[-+](?:(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee]',
                          r'[-+]?\d+)?)?[jJ]|"[-+]?(?:(?:\d+\.?\d*|\d*\.?\d',
                          r'+)(?:[Ee][-+]?\d+)?)?[j]"|"[-+]?(?:\d+\.?\d*|\d',
                          r'*\.?\d+)(?:[Ee][-+]?\d+)?[-+](?:(?:\d+\.?\d*|\d',
                          r'*\.?\d+)(?:[Ee][-+]?\d+)?)?[jJ]")']),
    'datetime64[D]': r'(?:\d{4}-\d{2}-\d{2}|"\d{4}-\d{2}-\d{2}")',
    '|S1': r'\S+',
    }

# Double check that the regular expressions are non-capturing.
# This is required for loadtable's regular expression
# matching to work correctly.
for key in quoted_dtype_to_re.keys():
    if re.compile(quoted_dtype_to_re[key]).groups >0:
        s = ''.join(['quoted_dtype_to_re: ',
                    'type regular expression for ',
                    key,
                    ' must be non-capturing'])
        raise ValueError(s)


def make_promotion_dict(d):
    """
    Construct the promotion dict to also deal with undetermined types
    and types that include missing values. It also symmetrizes the
    dtype_promotion_dict, which starts out unsymmetric.
    """
    for k1 in d.keys():
        for k2 in d[k1].keys():
                d[k2][k1] = d[k1][k2]
    d['NA'] = {}
    for k1 in d.keys():
      if k1 != 'NA':
        k1_na = 'NA' + k1
        d['NA'][k1] = k1_na
        d['NA'][k1_na] = k1_na
        d[k1]['NA'] = k1_na
        for k2 in d[k1].keys():
          if k2 != 'NA':
            k2_na = 'NA' + k2
            d[k1][k2_na] = 'NA'+d[k1][k2]
      d[k1][None] = k1
    d['NA']['NA'] = 'NA'
    return d

# The promotion relationships between the types.
# This is done via a dictionary of dictionaries.
# make_promotion_dict is called on the initial dictionary, which is
# unsymmetric. The final dictionary is (mostly) symmetric, in that
# dtype_promotion[type1][type2] = dtype_promotion[type2][type1].
# However the original dictionary only has one of those two values
# defined.
dtype_promotion = make_promotion_dict({
    'datetime64[D]': {'datetime64[D]': 'datetime64[D]',
                        'bool': '|S1',
                        'int32': '|S1',
                        'int64': '|S1',
                        'float32': '|S1',
                        'float64': '|S1',
                        'float32InfNaN': '|S1',
                        'float64InfNaN': '|S1',
                        'commafloat32': '|S1',
                        'commafloat64': '|S1',
                        'commafloat32InfNaN': '|S1',
                        'commafloat64InfNaN': '|S1',
                        'complex64': '|S1',
                        'complex128': '|S1',
                        '|S1': '|S1'},
    'bool': {'bool': 'bool',
                'int32': '|S1',
                'int64': '|S1',
                'float32': '|S1',
                'float64': '|S1',
                'float32InfNaN': '|S1',
                'float64InfNaN': '|S1',
                'commafloat32': '|S1',
                'commafloat64': '|S1',
                'commafloat32InfNaN': '|S1',
                'commafloat64InfNaN': '|S1',
                'complex64': '|S1',
                'complex128': '|S1',
                '|S1': '|S1'},
    'int32': {'int32': 'int32',
                'int64': 'int64',
                'float32': 'float32',
                'float64': 'float64',
                'float32InfNaN': 'float32InfNaN',
                'float64InfNaN': 'float64InfNaN',
                'commafloat32': 'commafloat32',
                'commafloat64': 'commafloat64',
                'commafloat32InfNaN': 'commafloat32InfNaN',
                'commafloat64InfNaN': 'commafloat64InfNaN',
                'complex64': 'complex64',
                'complex128': 'complex128',
                '|S1': '|S1'},
    'int64': {'int64': 'int64',
                'float32': 'float32',
                'float64': 'float64',
                'float32InfNaN': 'float32InfNaN',
                'float64InfNaN': 'float64InfNaN',
                'commafloat32': 'commafloat32',
                'commafloat64': 'commafloat64',
                'commafloat32InfNaN': 'commafloat32InfNaN',
                'commafloat64InfNaN': 'commafloat64InfNaN',
                'complex64': 'complex64',
                'complex128': 'complex128',
                '|S1': '|S1'},
    'float32': {'float32': 'float32',
                    'float64': 'float64',
                    'float32InfNaN': 'float32InfNaN',
                    'float64InfNaN': 'float64InfNaN',
                    'commafloat32': 'commafloat32',
                    'commafloat64': 'commafloat64',
                    'commafloat32InfNaN': 'commafloat32InfNaN',
                    'commafloat64InfNaN': 'commafloat64InfNaN',
                    'complex64': 'complex64',
                    'complex128': 'complex128',
                    '|S1': '|S1'},
    'float64': {'float64': 'float64',
                'float32InfNaN': 'float32InfNaN',
                'float64InfNaN': 'float64InfNaN',
                'commafloat32': 'commafloat32',
                'commafloat64': 'commafloat64',
                'commafloat32InfNaN': 'commafloat32InfNaN',
                'commafloat64InfNaN': 'commafloat64InfNaN',
                'complex64': 'complex64',
                'complex128': 'complex128',
                '|S1': '|S1'},
    'float32InfNaN': {'float32InfNaN': 'float32InfNaN',
                        'float64InfNaN': 'float64InfNaN',
                        'commafloat32': 'commafloat32',
                        'commafloat64': 'commafloat64',
                        'commafloat32InfNaN': 'commafloat32InfNaN',
                        'commafloat64InfNaN': 'commafloat64InfNaN',
                        'complex64': '|S1',
                        'complex128': '|S1',
                        '|S1': '|S1'},
    'float64InfNaN': {'float64InfNaN': 'float64InfNaN',
                        'commafloat32': 'commafloat32',
                        'commafloat64': 'commafloat64',
                        'commafloat32InfNaN': 'commafloat32InfNaN',
                        'commafloat64InfNaN': 'commafloat64InfNaN',
                        'complex64': '|S1',
                        'complex128': '|S1',
                        '|S1': '|S1'},
    'commafloat32': {'commafloat32': 'commafloat32',
                        'commafloat64': 'commafloat64',
                        'commafloat32InfNaN': 'commafloat32InfNaN',
                        'commafloat64InfNaN': 'commafloat64InfNaN',
                        'complex64': 'complex64',
                        'complex128': 'complex128',
                        '|S1':'|S1'},
    'commafloat64': {'commafloat64': 'commafloat64',
                        'commafloat32InfNaN': 'commafloat32InfNaN',
                        'commafloat64InfNaN': 'commafloat64InfNaN',
                        'complex64': 'complex64',
                        'complex128': 'complex128',
                        '|S1': '|S1'},
    'commafloat32InfNaN': {'commafloat32InfNaN': 'commafloat32InfNaN',
                            'commafloat64InfNaN': 'commafloat64InfNaN',
                            'complex64': '|S1',
                            'complex128': '|S1',
                            '|S1': '|S1'},
    'commafloat64InfNaN': {'commafloat64InfNaN': 'commafloat64InfNaN',
                            'complex64': '|S1',
                            'complex128': '|S1',
                            '|S1': '|S1'},
    'complex64': {'complex64': 'complex64',
                    'complex128': 'complex128',
                    '|S1': '|S1'},
    'complex128': {'complex128': 'complex128',
                    '|S1': '|S1'},
    '|S1': {'|S1': '|S1'},
})


bool_true_pattern = re.compile('^(True|TRUE|true)$')
bool_false_pattern = re.compile('^(False|FALSE|false)$')
def bool_conv(bool_string):
    """
    Simple type-instensitive conversion from string containing
    'True' to boolean True, and similarly for 'False' to False.
    Returns None otherwise.
    """
    if bool_true_pattern.match(bool_string):
        return True
    elif bool_false_pattern.match(bool_string):
        return False

def comma_float_conv(comma_float_string):
    """
    Simple converter for floats with comma instead of decimal
    point, e.g. 3,1416 instead of 3.1416
    """
    return float(comma_float_string.replace(',','.'))

# Dict relating type string to numpy dtype
dtype_dict = {
    'datetime64[D]': 'datetime64[D]',
    'bool': 'bool',
    'int32': 'int32',
    'int64': 'int64',
    'float32': 'float32',
    'float64': 'float64',
    'commafloat32': 'float32',
    'commafloat64': 'float64',
    'complex64': 'complex64',
    'complex128': 'complex128',
    'float32InfNaN': 'float32',
    'float64InfNaN': 'float64',
    'commafloat32InfNaN': 'float32',
    'commafloat64InfNaN': 'float64',
    '|S1': '|S1'
}

# Dict relating type string to converter
dtype_to_conv = {
    'datetime64[D]': None,
    'bool': bool_conv,
    'int32': int,
    'int64': int,
    'float32': float,
    'float64': float,
    'float32InfNaN': float,
    'float64InfNaN': float,
    'commafloat32': comma_float_conv,
    'commafloat64': comma_float_conv,
    'commafloat32InfNaN': comma_float_conv,
    'commafloat64InfNaN': comma_float_conv,
    'complex64': complex,
    'complex128': complex,
    'datetime64': None,
    '|S1': str
    }

# The following for loops add NA types to the previous
# 4 dicts.
# The NA types have the same data as the non-NA dtypes.

for key in dtype_dict.keys():
    NAkey = 'NA' + key
    dtype_dict[NAkey] = dtype_dict[key]

for key in dtype_to_conv.keys():
    NAkey = 'NA' + key
    dtype_to_conv[NAkey] = dtype_to_conv[key]

for key in dtype_to_re.keys():
    NAkey = 'NA' + key
    dtype_to_re[NAkey] = dtype_to_re[key]

for key in quoted_dtype_to_re.keys():
    NAkey = 'NA' + key
    quoted_dtype_to_re[NAkey] = quoted_dtype_to_re[key]

# Dict for default missing values for each dtype, to use in masked arrays
dtype_default_missing = {
    'bool': True,
    'int32': 999999,
    'int64': 999999,
    'float32': 1.e20,
    'float64': 1.e20,
    'complex64': 1.e20+0j,
    'complex128': 1.e20+0j,
    'datetime64[D]': 'NaT',
    '|S1': 'N/A'
    }

# For easy in the loadtable function, so the re's aren't recompiled
# every time loadtable is called.
quoted_entry_pattern = re.compile(r'"[^"]*"')
entry_re = r'"[^"]*"|[^"]*?'



def loadtable(fname,
        delimiter=' ',
        comments='#',
        header=False,
        type_search_order=['b1', 'i8', 'f8','M8[D]'],
        skip_lines=0,
        num_lines_search=0,
        string_sizes=1,
        check_sizes=True,
        is_Inf_NaN=True,
        NA_re='NA|',
        quoted=False,
        comma_decimals=False,
        force_mask=False,
        date_re=r'\d{4}-\d{2}-\d{2}',
        date_strp='%Y-%m-%d',
        default_missing_dtype='|S1'):
    """
    Load a text file with rows of data into a numpy record array.
    This function will automatically detect the types of each
    column, as well as the presense of missing values. If there are
    missing values a masked array is returned, otherwise a numpy
    array is returned.

    It will also automatically detect the prescense of unlabeled row
    names, but only if there is one column of them. This is done
    to make loading data saved from some other systems, such as R,
    easy to load.

    For most users, the only parameters of interest are fname, delimiter,
    header, and (possibly) type_search_order. The rest are for various
    more specialzed/unusual data formats.

    See Notes for performance tips.

    Parameters
    ----------
    fname: string or iterable (see description)
        Either the filename of the file containing the data or an iterable
        (such as a file object). If an iterable, must have __iter__, next,
        and seek methods.
    delimiter: string
        Regular expression for the delimeter between data. The regular
        expression must be non-capturing. (i.e. r'(?:3.14)' instead of
        r'(3.14)')
    comments: string
        Regular expression for the symbol(s) indicating the start
        of a comment.
    header: bool
        Flag indicating whether the data contains a row of column names.
    type_search_order: list of strings/dtypes
        List of objects which np.dtype will recognize as dtypes.
    skip_lines: int
        Number of lines in the beginning of the text file to skip before
        reading the data
    num_lines_search: int
        Number of lines, not including comments and header, to search
        for type and size information. Done to decrease the time required
        to determine the type and size information for data that is very
        homogenous.
    string_sizes: int or list of ints
        If a single int, interpreted as a minimum string size for all entries.
        If a list of ints, interpreted as the minimum string size for each
        individual entry. An error is thrown if the lenght of this list
        differs from the number of entries per row found. If check_sizes is
        False, then these minimum sizes are never changed.
   check_sizes: boolean or int
        Whether to check string sizes in each row for determining the size of
        string dtypes. This is an expensive option.
        If true it will check all lines for sizes. If an integer, it will
        check up to that number of rows from the beginning. And if false
        it will check no rows and use the defaults given from string_size.
    is_Inf_NaN: bool
        Whether to allow floats that are Inf and NaN
    NA_re: string
        Regular expression for missing data The regular
        expression must be non-capturing. (i.e. r'(?:3.14)' instead of
        r'(3.14)')
    quoted: bool
        Whether to allow the data to contain quotes, such as "3.14" or
        "North America"
    comma_decimals: bool
        Whether floats may use commas instead of decimals, e.g. "3,14"
        instead of "3.14".
    force_mask: bool
        Whether to force the returned array to be a masked array
    date_re: string
        The regular expression for dates. This assumes that all dates
        follow the same format. Defaults to the ISO standard. The regular
        expression must be non-capturing. (i.e. r'(?:3.14)' instead of
        r'(3.14)')
    date_strp: string
        The format to use for converting a date string to a date. Uses
        the format from datetime.datetime.strptime in the Python Standard
        Library datetime module.
    default_missing_dtype: string
        String representation for the default dtype for columns all of whose
        entries are missing data.

    Returns
    -------
    result: Numpy record array or masked record array
        The data stored in the file, as a numpy record array. The field
        names default to 'f'+i for field i if header is False, else the
        names from the first row of non-whitespace, non-comments.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.
    ValueError
        If the input file does not contain any data.
    RuntumeError
        If any regular expression matching fails

    See Also
    --------
    loadtxt, genfromtxt, dtype

    Notes
    -----
    This function operates by making two passes through the text file given.
    In the first pass it determines the dtypes based on regular expressions
    for the dtypes and custom promotion rules between dtypes. (The promotion
    rules are used, for example, if a column appears to be integer and then
    a float is seen.) In the first pass it can also determine the sizes of
    strings, if that option is enabled. After determining the dtypes and
    string sizes, it pre-allocates a numpy array of the appropriate size
    (or masked array in the prescense of missing data) and fills it line
    by line.

    The methods within this function are fairly modular, and it requires
    little difficulty to extract, for example, the method that determines
    dtypes or change the method for reading in data.

    Performance Tips:

     * Determning the sizes is expensive. For large arrays containing no
     string data it is best to set check_sizes to False.
     * Similarly, determining the dtypes can be expensive. If the text
     file is very large but has very homogeneous data (i.e. the dtypes are
     easily determined), then it is best to only check the first k lines
     for some reasonable value of k.
     * This method defaults to 64-bit ints and floats. If these sizes are
     unnecessary they should be reduced to 32-bit ints and floats to
     conserve space.
     * Converting comma float strings (i.e. '3,24') is about twice as
     expensive as converting decimal float strings

    Examples
    --------
    First create simply data file and then load it.
    >>> from StringIO import StringIO #StringIO behaves like a file object
    >>> s = ''.join(['#Comment \n \n TempRange, Cloudy, AvgInchesRain,',
    ...             'Count\n 60to70, True, 0.3, 5\n 60to70, True, 4.3,',
    ...             '3 \n 80to90, False, 0, 20'])
    >>> f = StringIO(s)

    >>> output = loadtable(f, header=True)
    >>> output
   array([('60to70', True, 0.29999999999999999, 5L),
       ('60to70', True, 4.2999999999999998, 3L), ('80to90', False, 0.0, 20L)],
      dtype=[('TempRange', '|S6'), ('Cloudy', '|b1'),
             ('AvgInchesRain', '<f8'),
             ('Count', '<i8')])

    The following is slightly more involved example.
    >>> from StringIO import StringIO #StringIO behaves like a file object
    >>> s = ''.join(['//Comment \n \n TempRange Cloudy AvgInchesRain',
    ...         ' Count\n #NA True "0,3" 5\n "60to70" True "4,3" 3 \n',
    ...         '"80to90" False #NA 20'])
    >>> f = StringIO(s)

    >>> output = loadtable(f, header=True,
    ...             delimiter=' ', comments='//', quoted=True,
    ...             comma_decimals=True, NA_re='#NA',
    ...             type_search_order=['b1','i4','f4'])
    >>> output #Reformated for readability
    masked_array(data = [(--, True, 0.30000001192092896, 5)
        ('60to70', True, 4.3000001907348633, 3)
        ('80to90', False, --, 20)],
    mask = [(True, False, False, False)
        (False, False, False, False)
        (False, False, True, False)],
    fill_value = ('N/A', True, 1.0000000200408773e+20, 999999),
    dtype = [('TempRange', 'S8'),
            ('Cloudy', '?'),
            ('AvgInchesRain', '<f4'),
            ('Count', '<i4')])

    For more examples see the test files for load_table in numpy/lib/tests
    """

    # Initialize various variables and sanitize inputs variables
    f = init_file(fname)
    re_dict = init_re_dict(quoted)
    type_search_order = init_type_search_order(type_search_order,
                                                is_Inf_NaN,
                                                comma_decimals)
    init_datetime(date_re, date_strp, re_dict, quoted)
    delimiter, NA_re = init_delimiter_and_NA(delimiter,
                                                NA_re,
                                                type_search_order,
                                                re_dict)
    ignore_pattern, delimiter_pattern = init_patterns(comments,
                                                        delimiter)
    if num_lines_search < 0:
        raise ValueError('num_lines_search must be non-negative')

    col_names = None

    # Find column titles, raises error if none found
    if header:
        col_names = get_col_names(f,
                                    ignore_pattern,
                                    delimiter,
                                    delimiter_pattern,
                                    skip_lines)
        if col_names == None:
            raise ValueError('File has no column names')

    nrows_data, sizes,coltypes,entry_pattern = get_nrows_sizes_coltypes(f,
                                                ignore_pattern,
                                                delimiter,
                                                delimiter_pattern,
                                                type_search_order,
                                                num_lines_search,
                                                string_sizes,
                                                check_sizes,
                                                col_names,
                                                NA_re,
                                                skip_lines,
                                                re_dict)
    # Raise error if no data
    if not nrows_data:
       raise ValueError('File has no data')

    # Determine if there are any missing values
    exists_NA = any([re.match('NA', coltype) for coltype in coltypes])

    # Read data into array
    if force_mask or exists_NA:
        dtype_dict['NA'] = np.dtype(default_missing_dtype).str
        data = get_data_missing(f,
                                ignore_pattern,
                                entry_pattern,
                                delimiter_pattern,
                                nrows_data,
                                sizes,
                                coltypes,
                                header,
                                col_names,
                                skip_lines,
                                re_dict,
                                quoted)
    else:
        data = get_data_no_missing(f,
                                    ignore_pattern,
                                    entry_pattern,
                                    delimiter_pattern,
                                    nrows_data,
                                    sizes,
                                    coltypes,
                                    header,
                                    col_names,
                                    skip_lines,
                                    quoted)

    return data

def init_file(fname):
    """
    initiate the file variable

    Parameters
    ----------
    fname: file or string
        The file to read data from, or the file's path (absolute or
        relative)

    Returns
    -------
    file
        The file object for the text file containing the data
    """
    if isinstance(fname, basestring):
        f = open(fname, 'U')
    elif hasattr(fname, '__iter__') and hasattr(fname, 'seek') and\
            hasattr(fname, 'next'):
        f = fname
    else:
        raise ValueError('fname must be string or file type')
    return f

def init_re_dict(quoted):
    """
    Determines whether to use quoted or unquoted regular expressions
    for the data

    Parameters
    ----------
    quoted: boolean
        True of the data can be quoted, False else

    Returns
    -------
    dictionary
        The dictionary with the types as keys and the regular expressions
        as values
    """
    if quoted:
        return quoted_dtype_to_re
    else:
        return dtype_to_re

def init_type_search_order(type_search_order, is_Inf_NaN, comma_decimals):
    """
    Initialize the type search order. This adds Inf/NaN types and the
    possibility of comma decimals if desired.

    Parameters
    ----------
    type_search_order: list of strings
        List of strings giving dtype names for the order in which
        type should be checked, and which types to check for.
    is_Inf_NaN: bool
        Whether to allow floats that are Inf and NaN
    comma_decimals: bool
        Whether floats may use commas instead of decimals

    Returns
    -------
    list of strings
        The updated type_search_order
"""
    # Obtain np.dtype.name for each type name. Due to quirk in dtype,
    # output from dtype.name can be different from the string used to
    # create the dtype.
    type_search_order = [np.dtype(s).name for s in type_search_order]
    if 'float32' in type_search_order:
        float_index = type_search_order.index('float32')
        if is_Inf_NaN and comma_decimals:
            type_search_order.insert(float_index+1, 'float32InfNaN')
            type_search_order.insert(float_index+2, 'commafloat32')
            type_search_order.insert(float_index+3, 'commafloat32InfNaN')
        if is_Inf_NaN and not comma_decimals:
            type_search_order.insert(float_index+1, 'float32InfNaN')
        if not is_Inf_NaN and comma_decimals:
            type_search_order.insert(float_index+1, 'commafloat32')
            type_search_order.insert(float_index+2, 'commafloat32InfNaN')

    if 'float64' in type_search_order:
        float_index = type_search_order.index('float64')
        if is_Inf_NaN and comma_decimals:
            type_search_order.insert(float_index+1, 'float64InfNaN')
            type_search_order.insert(float_index+2, 'commafloat64')
            type_search_order.insert(float_index+3, 'commafloat64InfNaN')
        if is_Inf_NaN and not comma_decimals:
            type_search_order.insert(float_index+1, 'float64InfNaN')
        if not is_Inf_NaN and comma_decimals:
            type_search_order.insert(float_index+1, 'commafloat64')
            type_search_order.insert(float_index+2, 'commafloat64InfNaN')

    return type_search_order

def init_datetime(date_re, date_strp, re_dict, quoted):
    """
    Initialize the datetime regular expression and converter

    Parameters
    ----------
    date_re: string
        The regular expression for dates.
    date_strp: string
        The format to use for converting a date string to a date. Uses
        the format from datetime.datetime.strptime in the Python Standard
        Library datetime module.
    re_dict: dictionary
        Dictionary mapping the types to their associated regular expressions
    quoted: bool
        Whether to allow the data to contain quotes

    Returns
    -------
    Nothing
        All changes are side-effects, made to global variables in this file
    """
    # Set date regular expression
    # Since this is a global variable it must be reset to default
    # in case previous calls set it to a non-default regular expression
    if re.compile(date_re).groups>0:
        raise ValueError("Date regular expression must be non-capturing.")
    re_dict['datetime64[D]'] = date_re
    if quoted and date_re==r'\d{4}-\d{2}-\d{2}':
        re_dict['datetime64[D]'] = ''.join(['(?:\d{4}-\d{2}-\d{2}|"',
                                            '\d{4}-\d{2}-\d{2}")'])
    # Create date converter
    f = lambda date_str : dt.datetime.strptime(date_str, date_strp).date()
    dtype_to_conv['datetime64[D]'] = f
    dtype_to_conv['NAdatetime64[D]'] =  dtype_to_conv['datetime64[D]']

def init_delimiter_and_NA(delimiter,
                            NA_re,
                            type_search_order,
                            re_dict):
    """
    Initialize the delimiter and NA_re

    Parameters
    ----------
    delimiter: string
        Regular expression for the delimeter between data.
    NA_re: string
        The the regular expression for missing data
    type_search_order: list of strings
        List of strings giving dtype names for the order in which
        type should be checked, and which types to check for.
    re_dict: dictionary
        Dictionary mapping the types to their associated regular expressions

    Returns
    -------
    string, string
        Sanitized versions of delimiter and NA_re
    """
    delimiter = delimiter.strip()
    # Space delimiter is a special case
    if delimiter == '':
        delimiter = r'\s+'
        if NA_re == 'NA|':
            NA_re = 'NA'
    else:
        delimiter = ''.join(['\s*(?:',
                            delimiter,
                            ')\s*'])
    if NA_re:
        type_search_order.append('NA')
        if re.compile(NA_re).groups>0:
            raise ValueError("NA regular expression must be non-capturing")
        NA_re  = ''.join(['(?:', NA_re,')'])
        re_dict['NA'] = NA_re
    return delimiter, NA_re

def init_patterns(comments, delimiter):
    """
    Initialize the ignore and delimiter compiled regular expressions

    Parameters
    ----------
    delimiter: string
        Regular expression for the delimeter between data.
    comments: string
        Regular expression for the symbol(s) indicating the start
        of a comment.

    Returns
    -------
    compiled regular expression, compiled regular expression
        The compiled regular expressions for a line to ignore
        and a delimiter
    """
    commentstr = ''.join(['^\s*',comments])
    # RE for comments and whitespace lines
    ignore_pattern = re.compile(''.join(['(',
                                commentstr,')|(^\s*$)']))
    # RE for the delimiter including white space
    delimiter_pattern = re.compile(''.join(['\s*(?:',
                                    delimiter,')\s*']))
    if delimiter_pattern.groups>0:
        raise ValueError("Delimiter regular expression must be non-capturing")
    return ignore_pattern, delimiter_pattern

def get_data_missing(f,
            ignore_pattern,
            entry_pattern,
            delimiter_pattern,
            nrows_data,
            sizes,
            coltypes,
            header,
            col_names,
            skip_lines,
            re_dict,
            quoted):
    """
    Function that actually loads the data into a masked array.

    Parameters
    ----------
    f: file
        The text file containing the data
    ignore_pattern: compiled regular expression
        The compiled regular expression which matches lines to ignore
    entry_pattern: compiled regular expression
        The compiled regular expression used to capture entries on a
        line containing data
    delimiter_pattern: compiled regular expression
        Compiled regular expression matching missing data
    nrows_data: integer
        Number of rows containing data
    sizes: list of integers
        The maximum size of any entry in each column in the data
    coltypes: list of strings
        String representations of the column types
    header: boolean
        Whether the data contains column names
    col_names: list of strings
        The column names, if header is True
    skip_lines: integer
        The number of lines to skip before reading the text file for data
    re_dict: dictionary
        Dictionary mapping the types to their associated regular expressions
    quoted: bool
        Whether to allow the data to contain quotes, such as "3.14" or
        "North America"

    Returns
    -------
    numpy masked array
        numpy masked array containing the data in f
    """

    NA_pattern = re.compile(''.join(['^\s*',
                                re_dict['NA'],
                                '\s*$']))
    # Construct the array of dtypes for each row entry
    coltypes_input = [dtype_dict[t] if t not in
                        ['|S1','NA|S1'] else '|S%d' %sizes[i]
                        for i,t in enumerate(coltypes)]
    # Constuct column names/dtype format dictionary if there is header
    if not header:
        col_names = []
        for i in xrange(len(coltypes_input)):
            col_names.append('f'+str(i))
    data_dtype = np.dtype(zip(col_names, coltypes_input))
    data = np.ma.zeros((nrows_data,), dtype = data_dtype)
    data.mask = np.zeros((nrows_data,), dtype='bool')
    f.seek(0)
    for count in xrange(skip_lines):
        next(f)
    seen_header = False
    i = 0

    # Special case when there is only one column of data
    # In that case can't enter data using tuple
    if len(coltypes_input) == 1:
        for line in f:
            if ignore_pattern.match(line):
                pass
            elif header and not seen_header:
                seen_header = True
            else:
                if NA_pattern.match(line):
                    data[i] = dtype_default_missing[coltypes[0]]
                    data.mask[i] = True
                else:
                    rowelem = line.strip()
                    if quoted:
                        rowelem = rowelem.replace('"','')
                    rowelem = dtype_to_conv[coltypes[0]](rowelem)
                    data[i] = rowelem
                i = i+1
    else:
        num_columns = len(coltypes_input)
        tmpline = [0]*num_columns
        for line in f:
            if(ignore_pattern.match(line)):
                pass
            elif(header and not seen_header):
                seen_header = True
            else:
                tmpmask = [False] * num_columns
                matches = entry_pattern.match(line.strip())
                if not matches:
                    raise RuntimeError('Cannot parse column data')
                rowelems = matches.groups()
                for j,rowelem in enumerate(rowelems):
                    if NA_pattern.match(rowelem):
                        tmpline[j] = dtype_default_missing[
                                          dtype_dict[coltypes[j]]]
                        tmpmask[j] = True
                    else:
                        if quoted:
                            rowelem = rowelem.replace('"','')
                        tmpline[j] = dtype_to_conv[coltypes[j]](rowelem)
                data[i] = tuple(tmpline)
                data.mask[i] = tuple(tmpmask)
                i += 1
    return data





def get_data_no_missing(f,
            ignore_pattern,
            entry_pattern,
            delimiter_pattern,
            nrows_data,
            sizes,
            coltypes,
            header,
            col_names,
            skip_lines,
            quoted):
    """
    Function that actually loads the data into a numpy array

    Parameters
    ----------
    f: file
        The text file containing the data
    ignore_pattern: compiled regular expression
        The compiled regular expression which matches lines to ignore
    entry_pattern: compiled regular expression
        The compiled regular expression used to capture entries on a
        line containing data
    delimiter_pattern: compiled regular expression
        Compiled regular expression matching missing data
    nrows_data: integer
        Number of rows containing data
    sizes: list of integers
        The maximum size of any entry in each column in the data
    coltypes: list of strings
        String representations of the column types
    header: boolean
        Whether the data contains column names
    col_names: list of strings
        The column names, if header is True
    skip_lines: integer
        The number of lines to skip before reading the text file for data
    quoted: bool
        Whether to allow the data to contain quotes, such as "3.14" or
        "North America"

    Returns
    -------
    numpy array
        numpy array containing the data in f
    """


    # Construct the array of dtypes for each row entry
    coltypes_input = [dtype_dict[t] if t != '|S1' else '|S%d' %sizes[i]
                        for i,t in enumerate(coltypes)]
    # Constuct column names/dtype format dictionary if there is header
    if not header:
        col_names = []
        for i in xrange(len(coltypes_input)):
            col_names.append('f'+str(i))
    data_dtype = np.dtype(zip(col_names, coltypes_input))
    data = np.zeros((nrows_data,), dtype = data_dtype)
    f.seek(0)
    for count in xrange(skip_lines):
        next(f)
    seen_header = False
    i = 0

    # Special case when there is only one column of data
    # In that case can't enter data using tuple
    if len(coltypes_input) == 1:
        for line in f:
            if ignore_pattern.match(line):
                pass
            elif header and not seen_header:
                seen_header = True
            else:
                if quoted:
                    rowelem = line.strip().replace('"','')
                    rowelem = dtype_to_conv[coltypes[0]](rowelem)
                else:
                    rowelem = dtype_to_conv[coltypes[0]](line.strip())
                data[i] = rowelem
                i = i+1
    else:
        for line in f:
            if(ignore_pattern.match(line)):
                pass
            elif(header and not seen_header):
                seen_header = True
            else:
                matches = entry_pattern.match(line.strip())
                if not matches:
                    raise RuntimeError('Cannot parse column data')
                rowelems = matches.groups()
                if quoted:
                    rowelems = [dtype_to_conv[coltypes[j]](
                                    rowelem.replace('"',''))
                                    for j,rowelem in
                                    enumerate(rowelems)]
                else:
                    rowelems = [dtype_to_conv[coltypes[j]](
                                    rowelem)
                                    for j,rowelem in
                                    enumerate(rowelems)]
                data[i] = tuple(rowelems)
                i += 1
    return data

def get_col_names(f,
                  ignore_pattern,
                  delimiter,
                  delimiter_pattern,
                  skip_lines):
    """
    Obtain the column names from a file of data.

    Parameters
    ----------
    f: file
        The text file containing the data
    ignore_pattern: compiled regular expression
        Compiled regular expression matching lines to be ignored
    delimiter: string
        Regular expression for missing data
    delimiter_pattern: compiled regular expression
        Compiled regular expression matching missing data
    skip_lines: int
        The number of lines to skip before looking for column names

    Returns
    -------
    None or list of strings
        Either returns a list of column names or None if no line
        containing data is found
    """

    for count in xrange(skip_lines):
        next(f)

    for line in f:
        if(ignore_pattern.match(line)):
            # line is comment or whitespace
            pass
        else:
            # Find column names. Eliminate double quotes around
            # column names, if they exist in data file
            quotes_gone = quoted_entry_pattern.sub('0',line)
            n = len(delimiter_pattern.split(quotes_gone.strip()))
            m = build_entry_pattern(n, delimiter)
            matches = m.match(line.strip())
            if not matches:
                raise RuntimeError('Cannot parse column names')
            col_names = [rowelem.replace('"','')
                            for rowelem in matches.groups()]
            return col_names
    return None


def build_entry_pattern(n, delimiter):
    """
    Build a compiled regular expression to capture the entries in
    a row of data

    Parameters
    ----------
    n: integer
        The number of columns in a row of data
    delimiter: string
        The regular expression for the delimiter between data entries

    Returns
    -------
    compiled regular expression
        A compiled regular expression to capture the entries in a row
        of data from the text file
    """
    if n==1:
        return re.compile(''.join(['^(',
                                    entry_re,
                                    ')$']))
    else:
        re_list = ['^(',entry_re,')']
        typical_entry = [delimiter, '(', entry_re,')']
        re_list.extend(typical_entry*(n-1))
        re_list.append('$')
        return re.compile(''.join(re_list))

def get_nrows_sizes_coltypes(f,
                            ignore_pattern,
                            delimiter,
                            delimiter_pattern,
                            type_search_order,
                            num_lines_search,
                            string_sizes,
                            check_sizes,
                            col_names,
                            NA_re,
                            skip_lines,
                            re_dict):
    """
    Obtain the number of rows of data, the
    string sizes for the data, and the column types
    from a file f. Assumes the header has already been
    read if there is a header.

    Parameters
    ----------
    f: file
        The text file containing the data
    ignore_pattern: compiled regular expression
        The compiled regular expression for lines to be ignored
    delimiter: string
        Regular expression for the delimieter between data
    delimiter_pattern: compiled regular expression
        Compiled regular expression for the delimiter between data
    type_search_order: list of strings
        List of string representations for the types to be checked for,
        in the order in the list
    num_lines_search: int
        The number of lines to use when determining the dtype for each
        column
    string_sizes: int or list of ints
        If a single int, interpreted as a minimum string size for all entries.
        If a list of ints, interpreted as the minimum string size for each
        individual entry. An error is thrown if the lenght of this list
        differs from the number of entries per row found. If check_sizes is
        False, then these minimum sizes are never changed.
    check_sizes: int
        Number of lines of data to use for baseline size estimates before
        checking via regular expressions. This is included because for
        complicated data files it can be extremely expensive to discover
        that the current size estimates are wrong, and it is much cheaper
        to simply spend a longer time determining sizes before assuming
        you have the correct sizes.
    col_names:
        The names for each column
    NA_re: string
        The the regular expression for missing data
    skip_lines: int
        The number of lines to skip before reading the text file for data
    re_dict: dictionary
        Dictionary associating types in type_search_order with regular
        expressions for each type

    Returns
    -------
    integer, list of integers, list of strings, compiled regular expression
        The number of rows of data, the maximum string size of each column's
        data, the types of each column, and a compiled regular expression
        to capture the entries in a row of data
    """

    nrows_data = 0
    coltypes = None
    sizes = None
    if check_sizes and isinstance(check_sizes,type(True)):
        check_sizes = np.Inf

    # RE of types in search order, to discover what type input matches
    # according to precedence.
    type_re = build_type_re(type_search_order, re_dict)
    white = r'\s*'

    if not col_names:
        for count in xrange(skip_lines):
            next(f)
    # Find the first non-empty line
    for line in f:
        # If not comment/whitespace, do initializations
        if not ignore_pattern.match(line):
            quotes_gone = quoted_entry_pattern.sub('0',line)
            ncol = len(delimiter_pattern.split(quotes_gone.strip()))
            if (col_names) and (len(col_names) != ncol):
                if ncol == len(col_names)+1:
                    col_names.insert(0,'row_names')
                else:
                    raise ValueError(''.join(['Number of columns and',
                                            ' column titles differ.']))
            coltypes = [None]*ncol
            entry_pattern = build_entry_pattern(ncol, delimiter)
            if isinstance(string_sizes, type(1)):
                sizes = [string_sizes]*ncol
            elif isinstance(string_sizes, type([1])):
                if len(string_sizes) != ncol:
                    raise ValueError(''.join(['string_sizes has wrong ',
                                              'number of elements']))
                sizes = string_sizes
            sizes = np.array(sizes)
            coltypes = update_coltypes(type_search_order,
                                    type_re, line, coltypes,
                                    delimiter_pattern,
                                    entry_pattern)
            if 0<check_sizes:
                update_sizes(sizes, line, entry_pattern)
            row_re = make_row_re(coltypes,
                                    white,
                                    delimiter,
                                    NA_re,
                                    re_dict)
            row_re_pattern = re.compile(row_re)
            nrows_data = 1
            break
    # Process the rest of the lines
    for line in f:
        if not ignore_pattern.match(line):
            nrows_data += 1
            if num_lines_search and (num_lines_search < nrows_data):
                continue
            # if not whitespace, check if line has predicted type pattern
            # if not, update type pattern
            if not row_re_pattern.match(line):
                coltypes = update_coltypes(type_search_order,
                                            type_re, line,
                                            coltypes,
                                            delimiter_pattern,
                                            entry_pattern)
                row_re = make_row_re(coltypes,
                                        white,
                                        delimiter,
                                        NA_re,
                                        re_dict)
                row_re_pattern = re.compile(row_re)
            if nrows_data<=check_sizes:
                update_sizes(sizes, line, entry_pattern)
    return nrows_data, sizes, coltypes, entry_pattern

def update_sizes(sizes, line, entry_pattern):
    """
    Update the sizes for a row of data. Checks a re of the
    current upper limit for the sizes against the line. If it
    doesn't match, the elements are seperated by delimiter, taking
    out the whitespace, and the sizes are increased to the sizes
    in the current row.

    Parameters
    ----------
    sizes: list of integers
        The current maximum size of each entry in the data
    line: string
        The current line of data
    delimiter_pattern: compiled regular expression
        The compiled regular expression for the delimiter separating entries

    Returns
    -------
    list of ints, compiled regular expression
        The new sizes and compiled regular expression for those sizes
    """

    matches = entry_pattern.match(line.strip())
    if not matches:
        raise RuntimeError('Cannot parse column data')
    rowelems = map(len, matches.groups())
    rowelems = np.array(rowelems)
    sizes = np.maximum(rowelems, sizes, out=sizes)

def build_type_re(type_search_order, re_dict):
    """
    Builds a regular expression for testing for types, using the
    types and search order specified in type_search_order.

    Parameters
    ----------
    type_search_order: list of strings
        List of the dtypes allowable in the order they're checked for.
        Dtypes are represented with their string representation.
    re_dict: dictionary
        Dictionary mapping each dtype string representation to the
        corresponding regular expression for that dtype

    Returns
    -------
    Compiled regular expresision
        Compiled regular expression for the dtypes in the order specified
        by type_search_order.
    """
    res = []
    for t in type_search_order:
        res.append(''.join(['(^', re_dict[t],'$)']))
    return re.compile('|'.join(res))


def update_coltypes(type_search_order,
                    type_re,
                    line,
                    coltypes,
                    delimiter_pattern,
                    entry_pattern):
    """
    Update the current gueses for dtypes for each column.
    Is only called if there is a mismatch between the current guess
    and the data in the row line, according to the current
    regular expression.

    Parameters
    ----------
    type_search_order: list of strings
        List of the dtypes to be searched for, in order
    type_re: compiled regular expression
        Compiled regular expression for the types being checked for
        in type_search_order.
    line: string
        The current line from the data file
    coltypes: list of strings
        String representations of the current guesses for column types
    delimiter_pattern: compiled regular expression
        Compiled regular expression for delimiter between data
    entry_pattern: compiled regular expression
        Compiled regular expression for what the entries in the row will
        look like.
    Returns
    -------
    list of strings
        List of the new column types, represented by the string for that
        type name.

    Notes
    -----
    entry_pattern is used to capture the entries in line and then type_re
    determines the type of each entry in the line.
    """
    newcoltypes = [None] * len(coltypes)
    matches = entry_pattern.match(line.strip())
    if not matches:
        raise RuntimeError('Cannot parse column data')
    rowelems = matches.groups()
    for i,elem  in enumerate(rowelems):
        m = type_re.match(elem)
        #m = re.match(type_re, elem, re.I)
        if m:
            ind = next((j for j,v in enumerate(m.groups()) if v!=None))
            # Use promote types. Depends on name in type_search_order
            # being a valid dtype name
            newtype = dtype_promotion[type_search_order[ind]][coltypes[i]]
            newcoltypes[i] = str(newtype)
        else:
            # All strings recorded as string size of length 1 to allow
            # consistent use in np.promote_types, and since
            # sizes recorded seperately
            newtype = dtype_promotion['|S1'][coltypes[i]]
            newcoltypes[i] = str(newtype)
    return newcoltypes

def make_row_re(coltypes, white, delimiter, NA_re, re_dict):
    """
    Make an re for current column type guesses.

    Parameters
    ----------
    coltypes: list of strings
        String representation of the current guesses for the types of
        each column
    white: string
        Regular expression for white space
    delimiter: string
        Regular expression for delimiter between data
    NA_re: string
        Regular expression for missing entries
    re_dict: dictionary
        Dictionary giving regular expression for each type

    Returns
    -------
    string
        Regular expression each succeeding row of the data file should match
    """
    na_type = re.compile('NA')
    if na_type.match(coltypes[0]):
        pieces = ['^', white, '(?:',re_dict[coltypes[0]],'|',
                        NA_re, ')']
    else:
        pieces = ['^', white, re_dict[coltypes[0]]]
    for ct in coltypes[1:]:
        if na_type.match(ct):
            pieces += [ delimiter, '(?:', re_dict[ct],
                       '|', NA_re,')']
        else:
            pieces += [delimiter, re_dict[ct]]
    pieces += [white, '$']
    return ''.join(pieces)








