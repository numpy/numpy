"""
Scan the directory of nep files and extract their metadata.  The
metadata is passed to Jinja for filling out `index.rst.tmpl`.
"""

import os
import sys
import jinja2
import glob
import re


def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)

def nep_metadata():
    ignore = ('nep-template.rst')
    sources = sorted(glob.glob(r'nep-*.rst'))
    sources = [s for s in sources if not s in ignore]

    meta_re = r':([a-zA-Z]*): (.*)'

    neps = {}
    print('Loading metadata for:')
    for source in sources:
        print(f' - {source}')
        nr = int(re.match(r'nep-([0-9]{4}).*\.rst', source).group(1))

        with open(source) as f:
            lines = f.readlines()
            tags = [re.match(meta_re, line) for line in lines]
            tags = [match.groups() for match in tags if match is not None]
            tags = {tag[0]: tag[1] for tag in tags}

            # We could do a clever regexp, but for now just assume the title is
            # the second line of the document
            tags['Title'] = lines[1].strip()
            tags['Filename'] = source

        neps[nr] = tags

    return {'neps': neps}


infile = 'index.rst.tmpl'
outfile = 'index.rst'

meta = nep_metadata()

print(f'Compiling {infile} -> {outfile}')
index = render(infile, meta)

with open(outfile, 'w') as f:
    f.write(index)
