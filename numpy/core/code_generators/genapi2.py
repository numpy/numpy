import sys

if sys.version_info[:2] < (2, 6):
    from sets import Set as set

from genapi import API_FILES, find_functions

def order_dict(d):
    """Order dict by its values."""
    o = d.items()
    def cmp(x, y):
        return x[1] - y[1]
    return sorted(o, cmp=cmp)

def merge_api_dicts(dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            ret[k] = v

    return ret

def check_api_dict(d):
    """Check that an api dict is valid (does not use the same index twice)."""
    # We have if a same index is used twice: we 'revert' the dict so that index
    # become keys. If the length is different, it means one index has been used
    # at least twice
    revert_dict = dict([(v, k) for k, v in d.items()])
    if not len(revert_dict) == len(d):
        # We compute a dict index -> list of associated items
        doubled = {}
        for name, index in d.items():
            try:
                doubled[index].append(name)
            except KeyError:
                doubled[index] = [name]
        msg = """\
Same index has been used twice in api definition: %s
""" % ['index %d -> %s' % (index, names) for index, names in doubled.items() \
                                          if len(names) != 1]
        raise ValueError(msg)

    # No 'hole' in the indexes may be allowed, and it must starts at 0
    indexes = set(d.values())
    expected = set(range(len(indexes)))
    if not indexes == expected:
        diff = expected.symmetric_difference(indexes)
        msg = "There are some holes in the API indexing: " \
              "(symmetric diff is %s)" % diff
        raise ValueError(msg)

def get_api_functions(tagname, api_dict):
    """Parse source files to get functions tagged by the given tag."""
    functions = []
    for f in API_FILES:
        functions.extend(find_functions(f, tagname))
    dfunctions = []
    for func in functions:
        o = api_dict[func.name]
        dfunctions.append( (o, func) )
    dfunctions.sort()
    return [a[1] for a in dfunctions]
