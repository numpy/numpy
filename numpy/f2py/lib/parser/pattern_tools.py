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
    ~~~~p1     -> <p1> [ <p1> ]...
    abs(p1)    -> whole string match of <p1>
    p1.named(name) -> match of <p1> has name
    p1.match(string) -> return string match with <p1>
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
                           }

    def __init__(self, label, pattern, optional=0):
        self.label = label
        self.pattern = pattern
        self.optional = optional
        return

    def get_compiled(self):
        try:
            return self._compiled_pattern
        except AttributeError:
            self._compiled_pattern = compiled = re.compile(self.pattern)
            return compiled

    def match(self, string):
        return self.get_compiled().match(string)

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
        rhs = t[-1]
        pattern_match = t[-2]
        assert abs(self).match(pattern_match),`pattern_match`
        lhs = ''.join(t[:-2])
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
        lhs = t[0]
        pattern_match = t[1]
        rhs = ''.join(t[2:])
        assert abs(self).match(pattern_match),`pattern_match`
        return lhs, pattern_match, rhs

    def __abs__(self):
        return Pattern(self.label, r'\A' + self.pattern+ r'\Z')

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.label, self.pattern)

    def __or__(self, other):
        label = '( %s OR %s )' % (self.label, other.label)
        if self.pattern==other.pattern:
            pattern = self.pattern
        else:
            pattern = '(%s|%s)' % (self.pattern, other.pattern)
        return Pattern(label, pattern)

    def __and__(self, other):
        if isinstance(other, Pattern):
            label = '%s%s' % (self.label, other.label)
            pattern = self.pattern + other.pattern
        else:
            assert isinstance(other,str),`other`
            label = '%s%s' % (self.label, other)
            pattern = self.pattern + other
        return Pattern(label, pattern)

    def __rand__(self, other):
        assert isinstance(other,str),`other`
        label = '%s%s' % (other, self.label)
        pattern = other + self.pattern
        return Pattern(label, pattern)

    def __invert__(self):
        if self.optional:
            if self.optional==1:
                return Pattern(self.label + '...', self.pattern[:-1] + '*', 2)
            if self.optional==2:
                return Pattern('%s %s' % (self.label[1:-4].strip(), self.label), self.pattern[:-1] + '+', 3)
            return self
        label = '[ %s ]' % (self.label)
        pattern = '(%s)?' % (self.pattern)
        return Pattern(label, pattern, 1)

    def __add__(self, other):
        if isinstance(other, Pattern):
            label = '%s %s' % (self.label, other.label)
            pattern = self.pattern + r'\s*' + other.pattern
        else:
            assert isinstance(other,str),`other`
            label = '%s %s' % (self.label, other)
            other = self._special_symbol_map.get(other, other)
            pattern = self.pattern + r'\s*' + other
        return Pattern(label, pattern)

    def __radd__(self, other):
        assert isinstance(other,str),`other`
        label = '%s %s' % (other, self.label)
        other = self._special_symbol_map.get(other, other)
        pattern = other + r'\s*' + self.pattern
        return Pattern(label, pattern)

    def named(self, name = None):
        if name is None:
            label = self.label
            assert label[0]+label[-1]=='<>' and ' ' not in label,`label`
        else:
            label = '<%s>' % (name)
        pattern = '(?P%s%s)' % (label.replace('-','_'), self.pattern)
        return Pattern(label, pattern)

    def rename(self, label):
        if label[0]+label[-1]!='<>':
            label = '<%s>' % (label)
        return Pattern(label, self.pattern, self.optional)

# Predefined patterns

letter = Pattern('<letter>','[a-zA-Z]')
name = Pattern('<name>', r'[a-zA-Z]\w*')
digit = Pattern('<digit>',r'\d')
underscore = Pattern('<underscore>', '_')
hex_digit = Pattern('<hex-digit>',r'[\da-fA-F]')

digit_string = Pattern('<digit-string>',r'\d+')
hex_digit_string = Pattern('<hex-digit-string>',r'[\da-fA-F]+')

sign = Pattern('<sign>',r'[+-]')
exponent_letter = Pattern('<exponent-letter>',r'[ED]')

alphanumeric_character = Pattern('<alphanumeric-character>','\w') # [a-z0-9_]
special_character = Pattern('<special-character>',r'[ =+-*/\()[\]{},.:;!"%&~<>?,\'`^|$#@]')
character = alphanumeric_character | special_character

kind_param = digit_string | name
signed_digit_string = ~sign + digit_string
int_literal_constant = digit_string + ~('_' + kind_param)
signed_int_literal_constant = ~sign + int_literal_constant

binary_constant = '[Bb]' + ("'" & digit_string & "'" | '"' & digit_string & '"')
octal_constant = '[Oo]' + ("'" & digit_string & "'" | '"' & digit_string & '"')
hex_constant = '[Zz]' + ("'" & hex_digit_string & "'" | '"' & hex_digit_string & '"')
boz_literal_constant = binary_constant | octal_constant | hex_constant

exponent = signed_digit_string
significand = digit_string + '.' + ~digit_string | '.' + digit_string
real_literal_constant = significand + ~(exponent_letter + exponent) + ~ ('_' + kind_param) | \
                        digit_string + exponent_letter + exponent + ~ ('_' + kind_param)
signed_real_literal_constant = ~sign + real_literal_constant

named_constant = name
real_part = signed_int_literal_constant | signed_real_literal_constant | named_constant
imag_part = real_part
complex_literal_constant = '(' + real_part + ',' + imag_part + ')'

char_literal_constant = ~( kind_param + '_') + "'.*'" | ~( kind_param + '_') + '".*"'

logical_literal_constant = '[.](TRUE|FALSE)[.]' + ~ ('_' + kind_param)
literal_constant = int_literal_constant | real_literal_constant | complex_literal_constant | logical_literal_constant | char_literal_constant | boz_literal_constant
constant = literal_constant | named_constant
int_constant = int_literal_constant | boz_literal_constant | named_constant
char_constant = char_literal_constant | named_constant
abs_constant = abs(constant)

power_op = Pattern('<power-op>','[*]{2}')
mult_op = Pattern('<mult-op>','[*/]')
add_op = Pattern('<add-op>','[+-]')
concat_op = Pattern('<concat-op>','[/]{}')
rel_op = Pattern('<rel-op>','([.](EQ|NE|LT|LE|GT|GE)[.])|[=]{2}|/[=]|[<]|[<][=]|[>]|[=][>]')
not_op = Pattern('<not-op>','[.]NOT[.]')
and_op = Pattern('<and-op>','[.]AND[.]')
or_op = Pattern('<or-op>','[.]OR[.]')
equiv_op = Pattern('<equiv-op>','[.](EQV|NEQV)[.]')
intrinsic_operator = power_op | mult_op | add_op | concat_op | rel_op | not_op | and_op | or_op | equiv_op
extended_intrinsic_operator = intrinsic_operator

defined_unary_op = Pattern('<defined-unary-op>','[.][a-zA-Z]+[.]')
defined_binary_op = Pattern('<defined-binary-op>','[.][a-zA-Z]+[.]')
defined_operator = defined_unary_op | defined_binary_op | extended_intrinsic_operator

label = Pattern('<label>','\d{1,5}')

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
