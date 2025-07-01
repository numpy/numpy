"""
Scan the directory of nep files and extract their metadata.  The
metadata is passed to Jinja for filling out the toctrees for various NEP
categories.
"""

import glob
import os
import re

import jinja2


def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)

def nep_metadata():
    ignore = ('nep-template.rst')
    sources = sorted(glob.glob(r'nep-*.rst'))
    sources = [s for s in sources if s not in ignore]

    meta_re = r':([a-zA-Z\-]*): (.*)'

    has_provisional = False
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

            # The title should be the first line after a line containing only
            # * or = signs.
            for i, line in enumerate(lines[:-1]):
                chars = set(line.rstrip())
                if len(chars) == 1 and ("=" in chars or "*" in chars):
                    break
            else:
                raise RuntimeError("Unable to find NEP title.")

            tags['Title'] = lines[i + 1].strip()
            tags['Filename'] = source

        if not tags['Title'].startswith(f'NEP {nr} — '):
            raise RuntimeError(
                f'Title for NEP {nr} does not start with "NEP {nr} — " '
                '(note that — here is a special, elongated dash). Got: '
                f'    {tags["Title"]!r}')

        if tags['Status'] in ('Accepted', 'Rejected', 'Withdrawn'):
            if 'Resolution' not in tags:
                raise RuntimeError(
                    f'NEP {nr} is Accepted/Rejected/Withdrawn but '
                    'has no Resolution tag'
                )
        if tags['Status'] == 'Provisional':
            has_provisional = True

        neps[nr] = tags

    # Now that we have all of the NEP metadata, do some global consistency
    # checks

    for nr, tags in neps.items():
        if tags['Status'] == 'Superseded':
            if 'Replaced-By' not in tags:
                raise RuntimeError(
                    f'NEP {nr} has been Superseded, but has no Replaced-By tag'
                )

            replaced_by = int(re.findall(r'\d+', tags['Replaced-By'])[0])
            replacement_nep = neps[replaced_by]

            if 'Replaces' not in replacement_nep:
                raise RuntimeError(
                    f'NEP {nr} is superseded by {replaced_by}, but that NEP has '
                    f"no Replaces tag."
                )

            if nr not in parse_replaces_metadata(replacement_nep):
                raise RuntimeError(
                    f'NEP {nr} is superseded by {replaced_by}, but that NEP has a '
                    f"Replaces tag of `{replacement_nep['Replaces']}`."
                )

        if 'Replaces' in tags:
            replaced_neps = parse_replaces_metadata(tags)
            for nr_replaced in replaced_neps:
                replaced_nep_tags = neps[nr_replaced]
                if not replaced_nep_tags['Status'] == 'Superseded':
                    raise RuntimeError(
                        f'NEP {nr} replaces NEP {nr_replaced}, but that NEP '
                        'has not been set to Superseded'
                    )

    return {'neps': neps, 'has_provisional': has_provisional}


def parse_replaces_metadata(replacement_nep):
    """Handle :Replaces: as integer or list of integers"""
    replaces = re.findall(r'\d+', replacement_nep['Replaces'])
    replaced_neps = [int(s) for s in replaces]
    return replaced_neps


meta = nep_metadata()

for nepcat in (
    "provisional", "accepted", "deferred", "finished", "meta",
    "open", "rejected",
):
    infile = f"{nepcat}.rst.tmpl"
    outfile = f"{nepcat}.rst"

    print(f'Compiling {infile} -> {outfile}')
    genf = render(infile, meta)
    with open(outfile, 'w') as f:
        f.write(genf)
