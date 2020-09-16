"""
Generate CPU features tables from CCompilerOpt
"""
from os import sys, path
gen_path = path.dirname(path.realpath(__file__))
#sys.path.append(path.abspath(path.join(gen_path, *([".."]*4), "numpy", "distutils")))
#from ccompiler_opt import CCompilerOpt
from numpy.distutils.ccompiler_opt import CCompilerOpt

class FakeCCompilerOpt(CCompilerOpt):
    fake_info = ""
    # disable caching no need for it
    conf_nocache = True
    def __init__(self, *args, **kwargs):
        no_cc = None
        CCompilerOpt.__init__(self, no_cc, **kwargs)
    def dist_compile(self, sources, flags, **kwargs):
        return sources
    def dist_info(self):
        return FakeCCompilerOpt.fake_info
    @staticmethod
    def dist_log(*args, stderr=False):
        # avoid printing
        pass
    def feature_test(self, name, force_flags=None):
        # To speed up
        return True

    def gen_features_table(self, features, ignore_groups=True,
                           field_names=["Name", "Implies"],
                           fstyle=None, fstyle_implies=None, **kwargs):
        rows = []
        if fstyle is None:
            fstyle = lambda ft: f'``{ft}``'
        if fstyle_implies is None:
            fstyle_implies = lambda origin, ft: fstyle(ft)
        for f in self.feature_sorted(features):
            is_group = "group" in self.feature_supported.get(f, {})
            if ignore_groups and is_group:
                continue
            implies = self.feature_sorted(self.feature_implies(f))
            implies = ' '.join([fstyle_implies(f, i) for i in implies])
            rows.append([fstyle(f), implies])
        if rows:
           return self.gen_rst_table(field_names, rows, **kwargs)

    def gen_gfeatures_table(self, features,
                            field_names=["Name", "Gather", "Implies"],
                            fstyle=None, fstyle_implies=None, **kwargs):
        rows = []
        if fstyle is None:
            fstyle = lambda ft: f'``{ft}``'
        if fstyle_implies is None:
            fstyle_implies = lambda origin, ft: fstyle(ft)
        for f in self.feature_sorted(features):
            gather = self.feature_supported.get(f, {}).get("group", None)
            if not gather:
                continue
            implies = self.feature_sorted(self.feature_implies(f))
            implies = ' '.join([fstyle_implies(f, i) for i in implies])
            gather = ' '.join([fstyle_implies(f, i) for i in gather])
            rows.append([fstyle(f), gather, implies])
        if rows:
            return self.gen_rst_table(field_names, rows, **kwargs)

    def gen_rst_table(self, field_names, rows, tab_size=4):
        assert(not rows or len(field_names) == len(rows[0]))
        rows.append(field_names)
        fld_len = len(field_names)
        cls_len = [max(len(c[i]) for c in rows) for i in range(fld_len)]
        del rows[-1]
        cformat = ' '.join('{:<%d}' % i for i in cls_len)
        border  = cformat.format(*['='*i for i in cls_len])

        rows = [cformat.format(*row) for row in rows]
        # header
        rows = [border, cformat.format(*field_names), border] + rows
        # footer
        rows += [border]
        # add left margin
        rows = [(' ' * tab_size) + r for r in rows]
        return '\n'.join(rows)

def features_table_sections(name, ftable=None, gtable=None, tab_size=4):
    tab = ' '*tab_size
    content = ''
    if ftable:
        title = f"{name} - CPU feature names"
        content = (
            f"{title}\n{'~'*len(title)}"
            f"\n.. table::\n{tab}:align: left\n\n"
            f"{ftable}\n\n"
        )
    if gtable:
        title = f"{name} - Group names"
        content += (
            f"{title}\n{'~'*len(title)}"
            f"\n.. table::\n{tab}:align: left\n\n"
            f"{gtable}\n\n"
        )
    return content

def features_table(arch, cc="gcc", pretty_name=None, **kwargs):
    FakeCCompilerOpt.fake_info = arch + cc
    ccopt = FakeCCompilerOpt(cpu_baseline="max")
    features = ccopt.cpu_baseline_names()
    ftable = ccopt.gen_features_table(features, **kwargs)
    gtable = ccopt.gen_gfeatures_table(features, **kwargs)

    if not pretty_name:
        pretty_name = arch + '/' + cc
    return features_table_sections(pretty_name, ftable, gtable, **kwargs)

def features_table_diff(arch, cc, cc_vs="gcc", pretty_name=None, **kwargs):
    FakeCCompilerOpt.fake_info = arch + cc
    ccopt = FakeCCompilerOpt(cpu_baseline="max")
    fnames = ccopt.cpu_baseline_names()
    features = {f:ccopt.feature_implies(f) for f in fnames}

    FakeCCompilerOpt.fake_info = arch + cc_vs
    ccopt_vs = FakeCCompilerOpt(cpu_baseline="max")
    fnames_vs = ccopt_vs.cpu_baseline_names()
    features_vs = {f:ccopt_vs.feature_implies(f) for f in fnames_vs}

    common  = set(fnames).intersection(fnames_vs)
    extra_avl = set(fnames).difference(fnames_vs)
    not_avl = set(fnames_vs).difference(fnames)
    diff_impl_f = {f:features[f].difference(features_vs[f]) for f in common}
    diff_impl = {k for k, v in diff_impl_f.items() if v}

    fbold = lambda ft: f'**{ft}**' if ft in extra_avl else f'``{ft}``'
    fbold_implies = lambda origin, ft: (
        f'**{ft}**' if ft in diff_impl_f.get(origin, {}) else f'``{ft}``'
    )
    diff_all = diff_impl.union(extra_avl)
    ftable = ccopt.gen_features_table(
        diff_all, fstyle=fbold, fstyle_implies=fbold_implies, **kwargs
    )
    gtable = ccopt.gen_gfeatures_table(
        diff_all, fstyle=fbold, fstyle_implies=fbold_implies, **kwargs
    )
    if not pretty_name:
        pretty_name = arch + '/' + cc
    content = features_table_sections(pretty_name, ftable, gtable, **kwargs)

    if not_avl:
        not_avl = ccopt_vs.feature_sorted(not_avl)
        not_avl = ' '.join(not_avl)
        content += (
            ".. note::\n"
            f"  The following features aren't supported by {pretty_name}:\n"
            f"  **{not_avl}**\n\n"
        )
    return content

if __name__ == '__main__':
    pretty_names = {
        "PPC64": "IBM/POWER big-endian",
        "PPC64LE": "IBM/POWER little-endian",
        "ARMHF": "ARMv7/A32",
        "AARCH64": "ARMv8/A64",
        "ICC": "Intel Compiler",
        # "ICCW": "Intel Compiler msvc-like",
        "MSVC": "Microsoft Visual C/C++"
    }
    with open(path.join(gen_path, 'simd-optimizations-tables.inc'), 'wt') as fd:
        fd.write(f'.. generated via {__file__}\n\n')
        for arch in (
            ("x86", "PPC64", "PPC64LE", "ARMHF", "AARCH64")
        ):
            pretty_name = pretty_names.get(arch, arch)
            table = features_table(arch=arch, pretty_name=pretty_name)
            assert(table)
            fd.write(table)

    with open(path.join(gen_path, 'simd-optimizations-tables-diff.inc'), 'wt') as fd:
        fd.write(f'.. generated via {__file__}\n\n')
        for arch, cc_names in (
            ("x86", ("clang", "ICC", "MSVC")),
            ("PPC64", ("clang",)),
            ("PPC64LE", ("clang",)),
            ("ARMHF", ("clang",)),
            ("AARCH64", ("clang",))
        ):
            arch_pname = pretty_names.get(arch, arch)
            for cc in cc_names:
                pretty_name = f"{arch_pname}::{pretty_names.get(cc, cc)}"
                table = features_table_diff(arch=arch, cc=cc, pretty_name=pretty_name)
                if table:
                    fd.write(table)
