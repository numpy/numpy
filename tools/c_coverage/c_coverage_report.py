#!/usr/bin/env python3
"""
A script to create C code-coverage reports based on the output of
valgrind's callgrind tool.

"""
import os
import re
import sys
from xml.sax.saxutils import quoteattr, escape

try:
    import pygments
    if tuple(int(x) for x in pygments.__version__.split('.')) < (0, 11):
        raise ImportError
    from pygments import highlight
    from pygments.lexers import CLexer
    from pygments.formatters import HtmlFormatter
    has_pygments = True
except ImportError:
    print("This script requires pygments 0.11 or greater to generate HTML")
    has_pygments = False


class FunctionHtmlFormatter(HtmlFormatter):
    """Custom HTML formatter to insert extra information with the lines."""
    def __init__(self, lines, **kwargs):
        HtmlFormatter.__init__(self, **kwargs)
        self.lines = lines

    def wrap(self, source, outfile):
        for i, (c, t) in enumerate(HtmlFormatter.wrap(self, source, outfile)):
            as_functions = self.lines.get(i - 1, None)
            if as_functions is not None:
                yield 0, ('<div title=%s style="background: #ccffcc">[%2d]' %
                          (quoteattr('as ' + ', '.join(as_functions)),
                           len(as_functions)))
            else:
                yield 0, '    '
            yield c, t
            if as_functions is not None:
                yield 0, '</div>'


class SourceFile:
    def __init__(self, path):
        self.path = path
        self.lines = {}

    def mark_line(self, lineno, as_func=None):
        line = self.lines.setdefault(lineno, set())
        if as_func is not None:
            as_func = as_func.split("'", 1)[0]
            line.add(as_func)

    def write_text(self, fd):
        with open(self.path, "r") as source:
            for i, line in enumerate(source):
                if i + 1 in self.lines:
                    fd.write("> ")
                else:
                    fd.write("! ")
                fd.write(line)

    def write_html(self, fd):
        with open(self.path, 'r') as source:
            code = source.read()
            lexer = CLexer()
            formatter = FunctionHtmlFormatter(
                self.lines,
                full=True,
                linenos='inline')
            fd.write(highlight(code, lexer, formatter))


class SourceFiles:
    def __init__(self):
        self.files = {}
        self.prefix = None

    def get_file(self, path):
        if path not in self.files:
            self.files[path] = SourceFile(path)
            if self.prefix is None:
                self.prefix = path
            else:
                self.prefix = os.path.commonprefix([self.prefix, path])
        return self.files[path]

    def clean_path(self, path):
        path = path[len(self.prefix):]
        return re.sub(r"[^A-Za-z0-9\.]", '_', path)

    def write_text(self, root):
        for path, source in self.files.items():
            with open(os.path.join(root, self.clean_path(path)), "w") as fd:
                source.write_text(fd)

    def write_html(self, root):
        for path, source in self.files.items():
            with open(
                os.path.join(root, self.clean_path(path) + ".html"), "w"
            ) as fd:
                source.write_html(fd)

        with open(os.path.join(root, 'index.html'), 'w') as fd:
            fd.write("<html>")
            paths = sorted(self.files.keys())
            for path in paths:
                fd.write('<p><a href="%s.html">%s</a></p>' %
                         (self.clean_path(path),
                          escape(path[len(self.prefix):])))
            fd.write("</html>")


def collect_stats(files, fd, pattern):
    # TODO: Handle compressed callgrind files
    line_regexs = [
        re.compile(r"(?P<lineno>[0-9]+)(\s[0-9]+)+"),
        re.compile(r"((jump)|(jcnd))=([0-9]+)\s(?P<lineno>[0-9]+)")
        ]

    current_file = None
    current_function = None
    for line in fd:
        if re.match(r"f[lie]=.+", line):
            path = line.split('=', 2)[1].strip()
            if os.path.exists(path) and re.search(pattern, path):
                current_file = files.get_file(path)
            else:
                current_file = None
        elif re.match(r"fn=.+", line):
            current_function = line.split('=', 2)[1].strip()
        elif current_file is not None:
            for regex in line_regexs:
                match = regex.match(line)
                if match:
                    lineno = int(match.group('lineno'))
                    current_file.mark_line(lineno, current_function)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'callgrind_file', nargs='+',
        help='One or more callgrind files')
    parser.add_argument(
        '-d', '--directory', default='coverage',
        help='Destination directory for output (default: %(default)s)')
    parser.add_argument(
        '-p', '--pattern', default='numpy',
        help='Regex pattern to match against source file paths '
             '(default: %(default)s)')
    parser.add_argument(
        '-f', '--format', action='append', default=[],
        choices=['text', 'html'],
        help="Output format(s) to generate. "
             "If option not provided, both will be generated.")
    args = parser.parse_args()

    files = SourceFiles()
    for log_file in args.callgrind_file:
        with open(log_file, 'r') as log_fd:
            collect_stats(files, log_fd, args.pattern)

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    if args.format == []:
        formats = ['text', 'html']
    else:
        formats = args.format
    if 'text' in formats:
        files.write_text(args.directory)
    if 'html' in formats:
        if not has_pygments:
            print("Pygments 0.11 or later is required to generate HTML")
            sys.exit(1)
        files.write_html(args.directory)
