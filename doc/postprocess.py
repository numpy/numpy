#!/usr/bin/env python
"""
%prog MODE FILES...

Post-processes HTML and Latex files output by Sphinx.
MODE is either 'html' or 'tex'.

"""
import re, optparse

def main():
    p = optparse.OptionParser(__doc__)
    options, args = p.parse_args()

    if len(args) < 1:
        p.error('no mode given')

    mode = args.pop(0)

    if mode not in ('html', 'tex'):
        p.error('unknown mode %s' % mode)

    for fn in args:
        f = open(fn, 'r')
        try:
            if mode == 'html':
                lines = process_html(fn, f.readlines())
            elif mode == 'tex':
                lines = process_tex(f.readlines())
        finally:
            f.close()

        f = open(fn, 'w')
        f.write("".join(lines))
        f.close()

def process_html(fn, lines):
    return lines

def process_tex(lines):
    """
    Remove unnecessary section titles from the LaTeX file.
    
    """
    new_lines = []
    for line in lines:
        line = unicode(line, 'utf-8').replace(u"\00a0", u"~").encode('utf-8')
        if (line.startswith(r'\section{numpy.')
            or line.startswith(r'\subsection{numpy.')
            or line.startswith(r'\subsubsection{numpy.')
            or line.startswith(r'\paragraph{numpy.')
            or line.startswith(r'\subparagraph{numpy.')
            ):
            pass # skip!
        else:
            new_lines.append(line)
    return new_lines

if __name__ == "__main__":
    main()
