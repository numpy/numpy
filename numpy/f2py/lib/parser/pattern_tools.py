
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
                           }

    def __init__(self, label, pattern, optional=0):
        self.label = label
        self.pattern = pattern
        self.optional = optional
        return

    def match(self, string):
        if hasattr(self, '_compiled_match'):
            return self._compiled.match(string)
        self._compiled = compiled = re.compile(self.pattern)
        return compiled.match(string)

    def __abs__(self):
        return Pattern(self.label, r'\A' + self.pattern+ r'\Z')

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.label, self.pattern)

    def __or__(self, other):
        label = '( %s OR %s )' % (self.label, other.label)
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

name = Pattern('<name>', r'[a-z]\w*')
digit_string = Pattern('<digit-string>',r'\d+')
sign = Pattern('<sign>',r'[+-]')
exponent_letter = Pattern('<exponent-letter>',r'[ED]')

kind_param = digit_string | name
signed_digit_string = ~sign + digit_string
int_literal_constant = digit_string + ~('_' + kind_param)
signed_int_literal_constant = ~sign + int_literal_constant

exponent = signed_digit_string
significand = digit_string + '.' + ~digit_string | '.' + digit_string
real_literal_constant = significand + ~(exponent_letter + exponent) + ~ ('_' + kind_param) | \
                        digit_string + exponent_letter + exponent + ~ ('_' + kind_param)
signed_real_literal_constant = ~sign + real_literal_constant


print signed_real_literal_constant
