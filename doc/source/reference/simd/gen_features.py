"""
Generate CPU features tables from CCompilerOpt
"""
from os import path

from numpy.distutils.ccompiler_opt import CCompilerOpt


class FakeCCompilerOpt(CCompilerOpt):
    # disable caching no need for it
    conf_nocache = True

    def __init__(self, arch, cc, *args, **kwargs):
        self.fake_info = (arch, cc, '')
        CCompilerOpt.__init__(self, None, **kwargs)

    def dist_compile(self, sources, flags, **kwargs):
        return sources

    def dist_info(self):
        return self.fake_info

    @staticmethod
    def dist_log(*args, stderr=False):
        # avoid printing
        pass

    def feature_test(self, name, force_flags=None, macros=[]):
        # To speed up
        return True

class Features:
    def __init__(self, arch, cc):
        self.copt = FakeCCompilerOpt(arch, cc, cpu_baseline="max")

    def names(self):
        return self.copt.cpu_baseline_names()

    def serialize(self, features_names):
        result = []
        for f in self.copt.feature_sorted(features_names):
            gather = self.copt.feature_supported.get(f, {}).get("group", [])
            implies = self.copt.feature_sorted(self.copt.feature_implies(f))
            result.append((f, implies, gather))
        return result

    def table(self, **kwargs):
        return self.gen_table(self.serialize(self.names()), **kwargs)

    def table_diff(self, vs, **kwargs):
        fnames = set(self.names())
        fnames_vs = set(vs.names())
        common = fnames.intersection(fnames_vs)
        extra = fnames.difference(fnames_vs)
        notavl = fnames_vs.difference(fnames)
        iextra = {}
        inotavl = {}
        idiff = set()
        for f in common:
            implies = self.copt.feature_implies(f)
            implies_vs = vs.copt.feature_implies(f)
            e = implies.difference(implies_vs)
            i = implies_vs.difference(implies)
            if not i and not e:
                continue
            if e:
                iextra[f] = e
            if i:
                inotavl[f] = e
            idiff.add(f)

        def fbold(f):
            if f in extra:
                return f':enabled:`{f}`'
            if f in notavl:
                return f':disabled:`{f}`'
            return f

        def fbold_implies(f, i):
            if i in iextra.get(f, {}):
                return f':enabled:`{i}`'
            if f in notavl or i in inotavl.get(f, {}):
                return f':disabled:`{i}`'
            return i

        diff_all = self.serialize(idiff.union(extra))
        diff_all += vs.serialize(notavl)
        content = self.gen_table(
            diff_all, fstyle=fbold, fstyle_implies=fbold_implies, **kwargs
        )
        return content

    def gen_table(self, serialized_features, fstyle=None, fstyle_implies=None,
                  **kwargs):

        if fstyle is None:
            fstyle = lambda ft: f'``{ft}``'
        if fstyle_implies is None:
            fstyle_implies = lambda origin, ft: fstyle(ft)

        rows = []
        have_gather = False
        for f, implies, gather in serialized_features:
            if gather:
                have_gather = True
            name = fstyle(f)
            implies = ' '.join([fstyle_implies(f, i) for i in implies])
            gather = ' '.join([fstyle_implies(f, i) for i in gather])
            rows.append((name, implies, gather))
        if not rows:
            return ''
        fields = ["Name", "Implies", "Gathers"]
        if not have_gather:
            del fields[2]
            rows = [(name, implies) for name, implies, _ in rows]
        return self.gen_rst_table(fields, rows, **kwargs)

    def gen_rst_table(self, field_names, rows, tab_size=4):
        assert not rows or len(field_names) == len(rows[0])
        rows.append(field_names)
        fld_len = len(field_names)
        cls_len = [max(len(c[i]) for c in rows) for i in range(fld_len)]
        del rows[-1]
        cformat = ' '.join('{:<%d}' % i for i in cls_len)
        border = cformat.format(*['=' * i for i in cls_len])

        rows = [cformat.format(*row) for row in rows]
        # header
        rows = [border, cformat.format(*field_names), border] + rows
        # footer
        rows += [border]
        # add left margin
        rows = [(' ' * tab_size) + r for r in rows]
        return '\n'.join(rows)

def wrapper_section(title, content, tab_size=4):
    tab = ' ' * tab_size
    if content:
        return (
            f"{title}\n{'~' * len(title)}"
            f"\n.. table::\n{tab}:align: left\n\n"
            f"{content}\n\n"
        )
    return ''

def wrapper_tab(title, table, tab_size=4):
    tab = ' ' * tab_size
    if table:
        ('\n' + tab).join((
            '.. tab:: ' + title,
            tab + '.. table::',
            tab + 'align: left',
            table + '\n\n'
        ))
    return ''


if __name__ == '__main__':

    pretty_names = {
        "PPC64": "IBM/POWER big-endian",
        "PPC64LE": "IBM/POWER little-endian",
        "S390X": "IBM/ZSYSTEM(S390X)",
        "ARMHF": "ARMv7/A32",
        "AARCH64": "ARMv8/A64",
        "ICC": "Intel Compiler",
        # "ICCW": "Intel Compiler msvc-like",
        "MSVC": "Microsoft Visual C/C++"
    }
    gen_path = path.join(
        path.dirname(path.realpath(__file__)), "generated_tables"
    )
    with open(path.join(gen_path, 'cpu_features.inc'), 'w') as fd:
        fd.write(f'.. generated via {__file__}\n\n')
        for arch in (
            ("x86", "PPC64", "PPC64LE", "ARMHF", "AARCH64", "S390X")
        ):
            title = "On " + pretty_names.get(arch, arch)
            table = Features(arch, 'gcc').table()
            fd.write(wrapper_section(title, table))

    with open(path.join(gen_path, 'compilers-diff.inc'), 'w') as fd:
        fd.write(f'.. generated via {__file__}\n\n')
        for arch, cc_names in (
            ("x86", ("clang", "ICC", "MSVC")),
            ("PPC64", ("clang",)),
            ("PPC64LE", ("clang",)),
            ("ARMHF", ("clang",)),
            ("AARCH64", ("clang",)),
            ("S390X", ("clang",))
        ):
            arch_pname = pretty_names.get(arch, arch)
            for cc in cc_names:
                title = f"On {arch_pname}::{pretty_names.get(cc, cc)}"
                table = Features(arch, cc).table_diff(Features(arch, "gcc"))
                fd.write(wrapper_section(title, table))
