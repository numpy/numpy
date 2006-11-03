"""
Tools for constructing patterns.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

import re

class Pattern:
    """
    p1 | p2    -> <p1> | <p2>
    p1 + p2    -> <p1> <p2>
    p1 & p2    -> <p1><p2>
    ~p1        -> [ <p1> ]
    ~~p1       -> [ <p1> ]...
    ~~~p1      -> <p1> [ <p1> ]...
    ~~~~p1     -> ~~~p1
    abs(p1)    -> whole string match of <p1>
    p1.named(name) -> match of <p1> has name
    p1.match(string) -> return string match with <p1>
    p1.flags(<re.I,..>)
    p1.rsplit(..) -> split a string from the rightmost p1 occurrence
    p1.lsplit(..) -> split a string from the leftmost p1 occurrence
    """
    _special_symbol_map = {'.': '[.]',
                           '*': '[*]',
                           '+': '[+]',
                           '|': '[|]',
                           '(': r'\(',
                           ')': r'\)',
                           '[': r'\[',
                           ']': r'\]',
                           '^': '[^]',
                           '$': '[$]',
                           '?': '[?]',
                           '{': '\{',
                           '}': '\}',
                           '>': '[>]',
                           '<': '[<]',
                           '=': '[=]'
                           }

    def __init__(self, label, pattern, optional=0, flags=0, value=None):
        self.label = label
        self.pattern = pattern
        self.optional = optional
        self._flags = flags
        self.value = value
        return

    def flags(self, *flags):
        f = self._flags
        for f1 in flags:
            f = f | f1
        return Pattern(self.label, self.pattern, optional=self.optional, flags=f, value=self.value)

    def get_compiled(self):
        try:
            return self._compiled_pattern
        except AttributeError:
            self._compiled_pattern = compiled = re.compile(self.pattern, self._flags)
            return compiled

    def match(self, string):
        return self.get_compiled().match(string)

    def search(self, string):
        return self.get_compiled().search(string)

    def rsplit(self, string):
        """
        Return (<lhs>, <pattern_match>, <rhs>) where
          string = lhs + pattern_match + rhs
        and rhs does not contain pattern_match.
        If no pattern_match is found in string, return None.
        """
        compiled = self.get_compiled()
        t = compiled.split(string)
        if len(t) < 3: return
        if '' in t[1:-1]: return
        rhs = t[-1].strip()
        pattern_match = t[-2].strip()
        assert abs(self).match(pattern_match),`self,string,t,pattern_match`
        lhs = (''.join(t[:-2])).strip()
        return lhs, pattern_match, rhs

    def lsplit(self, string):
        """
        Return (<lhs>, <pattern_match>, <rhs>) where
          string = lhs + pattern_match + rhs
        and rhs does not contain pattern_match.
        If no pattern_match is found in string, return None.
        """
        compiled = self.get_compiled()
        t = compiled.split(string) # can be optimized
        if len(t) < 3: return
        lhs = t[0].strip()
        pattern_match = t[1].strip()
        rhs = (''.join(t[2:])).strip()
        assert abs(self).match(pattern_match),`pattern_match`
        return lhs, pattern_match, rhs

    def __abs__(self):
        return Pattern(self.label, r'\A' + self.pattern+ r'\Z',flags=self._flags, value=self.value)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.label, self.pattern)

    def __or__(self, other):
        label = '( %s OR %s )' % (self.label, other.label)
        if self.pattern==other.pattern:
            pattern = self.pattern
            flags = self._flags
        else:
            pattern = '(%s|%s)' % (self.pattern, other.pattern)
            flags = self._flags | other._flags
        return Pattern(label, pattern, flags=flags)

    def __and__(self, other):
        if isinstance(other, Pattern):
            label = '%s%s' % (self.label, other.label)
            pattern = self.pattern + other.pattern
            flags = self._flags | other._flags
        else:
            assert isinstance(other,str),`other`
            label = '%s%s' % (self.label, other)
            pattern = self.pattern + other
            flags = self._flags
        return Pattern(label, pattern, flags=flags)

    def __rand__(self, other):
        assert isinstance(other,str),`other`
        label = '%s%s' % (other, self.label)
        pattern = other + self.pattern
        return Pattern(label, pattern, flags=self._flags)

    def __invert__(self):
        if self.optional:
            if self.optional==1:
                return Pattern(self.label + '...', self.pattern[:-1] + '*', optional=2,flags=self._flags)
            if self.optional==2:
                return Pattern('%s %s' % (self.label[1:-4].strip(), self.label), self.pattern[:-1] + '+',
                               optional=3, flags=self._flags)
            return self
        label = '[ %s ]' % (self.label)
        pattern = '(%s)?' % (self.pattern)
        return Pattern(label, pattern, optional=1, flags=self._flags)

    def __add__(self, other):
        if isinstance(other, Pattern):
            label = '%s %s' % (self.label, other.label)
            pattern = self.pattern + r'\s*' + other.pattern
            flags = self._flags | other._flags
        else:
            assert isinstance(other,str),`other`
            label = '%s %s' % (self.label, other)
            other = self._special_symbol_map.get(other, other)
            pattern = self.pattern + r'\s*' + other
            flags = self._flags
        return Pattern(label, pattern, flags = flags)

    def __radd__(self, other):
        assert isinstance(other,str),`other`
        label = '%s %s' % (other, self.label)
        other = self._special_symbol_map.get(other, other)
        pattern = other + r'\s*' + self.pattern
        return Pattern(label, pattern, flags=self._flags)

    def named(self, name = None):
        if name is None:
            label = self.label
            assert label[0]+label[-1]=='<>' and ' ' not in label,`label`
        else:
            label = '<%s>' % (name)
        pattern = '(?P%s%s)' % (label.replace('-','_'), self.pattern)
        return Pattern(label, pattern, flags=self._flags, value= self.value)

    def rename(self, label):
        if label[0]+label[-1]!='<>':
            label = '<%s>' % (label)
        return Pattern(label, self.pattern, optional=self.optional, flags=self._flags, value=self.value)

    def __call__(self, string):
        m = self.match(string)
        if m is None: return
        if self.value is not None: return self.value
        return m.group()

# Predefined patterns

letter = Pattern('<letter>','[A-Z]',flags=re.I)
name = Pattern('<name>', r'[A-Z]\w*',flags=re.I)
digit = Pattern('<digit>',r'\d')
underscore = Pattern('<underscore>', '_')
binary_digit = Pattern('<binary-digit>',r'[01]')
octal_digit = Pattern('<octal-digit>',r'[0-7]')
hex_digit = Pattern('<hex-digit>',r'[\dA-F]',flags=re.I)

digit_string = Pattern('<digit-string>',r'\d+')
binary_digit_string = Pattern('<binary-digit-string>',r'[01]+')
octal_digit_string = Pattern('<octal-digit-string>',r'[0-7]+')
hex_digit_string = Pattern('<hex-digit-string>',r'[\dA-F]+',flags=re.I)

sign = Pattern('<sign>',r'[+-]')
exponent_letter = Pattern('<exponent-letter>',r'[ED]',flags=re.I)

alphanumeric_character = Pattern('<alphanumeric-character>',r'\w') # [A-Z0-9_]
special_character = Pattern('<special-character>',r'[ =+-*/\()[\]{},.:;!"%&~<>?,\'`^|$#@]')
character = alphanumeric_character | special_character

kind_param = digit_string | name
kind_param_named = kind_param.named('kind-param')
signed_digit_string = ~sign + digit_string
int_literal_constant = digit_string + ~('_' + kind_param)
signed_int_literal_constant = ~sign + int_literal_constant
int_literal_constant_named = digit_string.named('value') + ~ ('_' + kind_param_named)
signed_int_literal_constant_named = (~sign + digit_string).named('value') + ~ ('_' + kind_param_named)

binary_constant = ('B' + ("'" & binary_digit_string & "'" | '"' & binary_digit_string & '"')).flags(re.I)
octal_constant = ('O' + ("'" & octal_digit_string & "'" | '"' & octal_digit_string & '"')).flags(re.I)
hex_constant = ('Z' + ("'" & hex_digit_string & "'" | '"' & hex_digit_string & '"')).flags(re.I)
boz_literal_constant = binary_constant | octal_constant | hex_constant

exponent = signed_digit_string
significand = digit_string + '.' + ~digit_string | '.' + digit_string
real_literal_constant = significand + ~(exponent_letter + exponent) + ~ ('_' + kind_param) | \
                        digit_string + exponent_letter + exponent + ~ ('_' + kind_param)
real_literal_constant_named = (significand + ~(exponent_letter + exponent) |\
                               digit_string + exponent_letter + exponent).named('value') +  ~ ('_' + kind_param_named)
signed_real_literal_constant_named = (~sign + (significand + ~(exponent_letter + exponent) |\
                               digit_string + exponent_letter + exponent)).named('value') +  ~ ('_' + kind_param_named)
signed_real_literal_constant = ~sign + real_literal_constant

named_constant = name
real_part = signed_int_literal_constant | signed_real_literal_constant | named_constant
imag_part = real_part
complex_literal_constant = '(' + real_part + ',' + imag_part + ')'

a_n_rep_char = Pattern('<alpha-numeric-rep-char>',r'\w')
rep_char = Pattern('<rep-char>',r'.')
char_literal_constant = ~( kind_param + '_') + ("'" + ~~rep_char + "'" | '"' + ~~rep_char + '"' )
a_n_char_literal_constant_named1 = ~( kind_param_named + '_') + (~~~("'" + ~~a_n_rep_char + "'" )).named('value')
a_n_char_literal_constant_named2 = ~( kind_param_named + '_') + (~~~('"' + ~~a_n_rep_char + '"' )).named('value')

logical_literal_constant = ('[.](TRUE|FALSE)[.]' + ~ ('_' + kind_param)).flags(re.I)
logical_literal_constant_named = Pattern('<value>',r'[.](TRUE|FALSE)[.]',flags=re.I).named() + ~ ('_' + kind_param_named)
literal_constant = int_literal_constant | real_literal_constant | complex_literal_constant | logical_literal_constant | char_literal_constant | boz_literal_constant
constant = literal_constant | named_constant
int_constant = int_literal_constant | boz_literal_constant | named_constant
char_constant = char_literal_constant | named_constant

# assume that replace_string_map is applied:
part_ref = name + ~((r'[(]' + name + r'[)]'))
data_ref = part_ref + ~~~(r'[%]' + part_ref)
primary = constant | name | data_ref | (r'[(]' + name + r'[)]')

power_op = Pattern('<power-op>','[*]{2}')
mult_op = Pattern('<mult-op>',r'[/*]')
add_op = Pattern('<add-op>',r'[+-]')
concat_op = Pattern('<concat-op>',r'[/]{2}')
rel_op = Pattern('<rel-op>','[.]EQ[.]|[.]NE[.]|[.]LT[.]|[.]LE[.]|[.]GT[.]|[.]GE[.]|[=]{2}|/[=]|[<][=]|[<]|[>][=]|[>]',flags=re.I)
not_op = Pattern('<not-op>','[.]NOT[.]',flags=re.I)
and_op = Pattern('<and-op>','[.]AND[.]',flags=re.I)
or_op = Pattern('<or-op>','[.]OR[.]',flags=re.I)
equiv_op = Pattern('<equiv-op>','[.]EQV[.]|[.]NEQV[.]',flags=re.I)
percent_op = Pattern('<percent-op>',r'%',flags=re.I)
intrinsic_operator = power_op | mult_op | add_op | concat_op | rel_op | not_op | and_op | or_op | equiv_op
extended_intrinsic_operator = intrinsic_operator

defined_unary_op = Pattern('<defined-unary-op>','[.][A-Z]+[.]',flags=re.I)
defined_binary_op = Pattern('<defined-binary-op>','[.][A-Z]+[.]',flags=re.I)
defined_operator = defined_unary_op | defined_binary_op | extended_intrinsic_operator
abs_defined_operator = abs(defined_operator)

non_defined_binary_op = intrinsic_operator | logical_literal_constant

label = Pattern('<label>','\d{1,5}')
abs_label = abs(label)

keyword = name
keyword_equal = keyword + '='




abs_constant = abs(constant)
abs_literal_constant = abs(literal_constant)
abs_int_literal_constant = abs(int_literal_constant)
abs_signed_int_literal_constant = abs(signed_int_literal_constant)
abs_signed_int_literal_constant_named = abs(signed_int_literal_constant_named)
abs_int_literal_constant_named = abs(int_literal_constant_named)
abs_real_literal_constant = abs(real_literal_constant)
abs_signed_real_literal_constant = abs(signed_real_literal_constant)
abs_signed_real_literal_constant_named = abs(signed_real_literal_constant_named)
abs_real_literal_constant_named = abs(real_literal_constant_named)
abs_complex_literal_constant = abs(complex_literal_constant)
abs_logical_literal_constant = abs(logical_literal_constant)
abs_char_literal_constant = abs(char_literal_constant)
abs_boz_literal_constant = abs(boz_literal_constant)
abs_name = abs(name)
abs_a_n_char_literal_constant_named1 = abs(a_n_char_literal_constant_named1)
abs_a_n_char_literal_constant_named2 = abs(a_n_char_literal_constant_named2)
abs_logical_literal_constant_named = abs(logical_literal_constant_named)
abs_binary_constant = abs(binary_constant)
abs_octal_constant = abs(octal_constant)
abs_hex_constant = abs(hex_constant)

intrinsic_type_name = Pattern('<intrinsic-type-name>',r'(INTEGER|REAL|COMPLEX|LOGICAL|CHARACTER|DOUBLE\s*COMPLEX|DOUBLE\s*PRECISION|BYTE)',flags=re.I)
abs_intrinsic_type_name = abs(intrinsic_type_name)
double_complex_name = Pattern('<double-complex-name>','DOUBLE\s*COMPLEX', flags=re.I, value='DOUBLE COMPLEX')
double_precision_name = Pattern('<double-precision-name>','DOUBLE\s*PRECISION', flags=re.I, value='DOUBLE PRECISION')
abs_double_complex_name = abs(double_complex_name)
abs_double_precision_name = abs(double_precision_name)

access_spec = Pattern('<access-spec>',r'PUBLIC|PRIVATE',flags=re.I)
abs_access_spec = abs(access_spec)

implicit_none = Pattern('<implicit-none>',r'IMPLICIT\s*NONE',flags=re.I, value='IMPLICIT NONE')
abs_implicit_none = abs(implicit_none)

attr_spec = Pattern('<attr-spec>',r'ALLOCATABLE|ASYNCHRONOUS|EXTERNAL|INTENT|INTRINSIC|OPTIONAL|PARAMETER|POINTER|PROTECTED|SAVE|TARGET|VALUE|VOLATILE',flags=re.I)
abs_attr_spec = abs(attr_spec)

dimension = Pattern('<dimension>',r'DIMENSION', flags=re.I)
abs_dimension = abs(dimension)

intent = Pattern('<intent>', r'INTENT', flags=re.I)
abs_intent = abs(intent)

intent_spec = Pattern('<intent-spec>', r'INOUT|IN|OUT', flags=re.I)
abs_intent_spec = abs(intent_spec)

subroutine = Pattern('<subroutine>', r'SUBROUTINE', flags=re.I)

def _test():
    assert name.match('a1_a')
    assert abs(name).match('a1_a')
    assert not abs(name).match('a1_a[]')

    m = abs(kind_param)
    assert m.match('23')
    assert m.match('SHORT')

    m = abs(signed_digit_string)
    assert m.match('23')
    assert m.match('+ 23')
    assert m.match('- 23')
    assert m.match('-23')
    assert not m.match('+n')

    m = ~sign.named() + digit_string.named('number')
    r = m.match('23')
    assert r.groupdict()=={'number': '23', 'sign': None}
    r = m.match('- 23')
    assert r.groupdict()=={'number': '23', 'sign': '-'}

    m = abs(char_literal_constant)
    assert m.match('"adadfa"')
    assert m.match('"adadfa""adad"')
    assert m.match('HEY_"adadfa"')
    assert m.match('HEY _ "ad\tadfa"')
    assert not m.match('adadfa')
    print 'ok'

if __name__ == '__main__':
    _test()
