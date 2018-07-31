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

    meta_re = r':([a-zA-Z\-]*): (.*)'

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

        if not tags['Title'].startswith(f'NEP {nr} — '):
            raise RuntimeError(
                f'Title for NEP {nr} does not start with "NEP {nr} — " '
                '(note that — here is a special, enlongated dash)')

        if tags['Status'] in ('Accepted', 'Rejected', 'Withdrawn'):
            if not 'Resolution' in tags:
                raise RuntimeError(
                    f'NEP {nr} is Accepted/Rejected/Withdrawn but '
                    'has no Resolution tag'
                )

        neps[nr] = tags

    # Now that we have all of the NEP metadata, do some global consistency
    # checks

    for nr, tags in neps.items():
        if tags['Status'] == 'Superseded':
            if not 'Replaced-By' in tags:
                raise RuntimeError(
                    f'NEP {nr} has been Superseded, but has no Replaced-By tag'
                )

            replaced_by = int(tags['Replaced-By'])
            replacement_nep = neps[replaced_by]

            if not 'Replaces' in replacement_nep:
                raise RuntimeError(
                    f'NEP {nr} is superseded by {replaced_by}, but that NEP has '
                    f"no Replaces tag."
                )

            if not int(replacement_nep['Replaces']) == nr:
                raise RuntimeError(
                    f'NEP {nr} is superseded by {replaced_by}, but that NEP has a '
                    f"Replaces tag of `{replacement_nep['Replaces']}`."
                )

        if 'Replaces' in tags:
            replaced_nep = int(tags['Replaces'])
            replaced_nep_tags = neps[replaced_nep]
            if not replaced_nep_tags['Status'] == 'Superseded':
                raise RuntimeError(
                    f'NEP {nr} replaces {replaced_nep}, but that NEP has not '
                    f'been set to Superseded'
                )

    return {'neps': neps}


infile = 'index.rst.tmpl'
outfile = 'index.rst'

meta = nep_metadata()

print(f'Compiling {infile} -> {outfile}')
index = render(infile, meta)

with open(outfile, 'w') as f:
    f.write(index)
