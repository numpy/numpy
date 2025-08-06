#!/usr/bin/env python3
"""
Post-processes HTML and Latex files output by Sphinx.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', help='file mode', choices=('html', 'tex'))
    parser.add_argument('file', nargs='+', help='input file(s)')
    args = parser.parse_args()

    mode = args.mode

    for fn in args.file:
        with open(fn, encoding="utf-8") as f:
            if mode == 'html':
                lines = process_html(fn, f.readlines())
            elif mode == 'tex':
                lines = process_tex(f.readlines())

        with open(fn, 'w', encoding="utf-8") as f:
            f.write("".join(lines))

def process_html(fn, lines):
    return lines

def process_tex(lines):
    """
    Remove unnecessary section titles from the LaTeX file.

    """
    new_lines = []
    for line in lines:
        if line.startswith(("\\section{numpy.",
                            "\\subsection{numpy.",
                            "\\subsubsection{numpy.",
                            "\\paragraph{numpy.",
                            "\\subparagraph{numpy.",
                            )):
            pass
        else:
            new_lines.append(line)
    return new_lines


if __name__ == "__main__":
    main()
