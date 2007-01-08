from tokenize import  generate_tokens
import token
from StringIO import StringIO
import sys
def insert(s1, s2, posn):
    """insert s1 into s2 at positions posn

    >>> insert("XX", "abcdef", [2, 4])
    'abXXcdXXef'
    """
    pieces = []
    start = 0
    for end in posn + [len(s2)]:
        pieces.append(s2[start:end])
        start = end
    return s1.join(pieces)

def insert_dtype(readline, output=None):
    """
    >>> src = "zeros((2,3), dtype=float); zeros((2,3));"
    >>> insert_dtype(StringIO(src).readline)
    zeros((2,3), dtype=float); zeros((2,3), dtype=int);
    """
    if output is None:
        output = sys.stdout
    tokens = generate_tokens(readline)
    flag = 0
    parens = 0
    argno = 0
    posn = []
    nodtype = True
    prevtok = None
    kwarg = 0
    for (tok_type, tok, (srow, scol), (erow, ecol), line) in tokens:
        if not flag and tok_type == token.NAME and tok in ('zeros', 'ones', 'empty'):
            flag = 1
        else:
            if tok == '(':
                parens += 1
            elif tok == ')':
                parens -= 1
                if parens == 0:
                    if nodtype and argno < 1:
                        posn.append(scol)
                    argno = 0
                    flag = 0
                    nodtype = True
                    argno = 0
            elif tok == '=':
                kwarg = 1
                if prevtok == 'dtype':
                    nodtype = False
            elif tok == ',':
                argno += (parens == 1)
        if len(line) == ecol:
            output.write(insert(', dtype=int', line, posn))
            posn = []
        prevtok = tok

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
