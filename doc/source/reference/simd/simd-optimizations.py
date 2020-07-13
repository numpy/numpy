"""
Generate CPU features tables from CCompilerOpt
"""
from os import sys, path
gen_path = path.dirname(path.realpath(__file__))
from numpy.distutils.ccompiler_opt import CCompilerOpt

class FakeCCompilerOpt(CCompilerOpt):
    fake_info = ""
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
                           field_names=["Name", "Implies"], **kwargs):
        rows = []
        for f in features:
            is_group = "group" in self.feature_supported.get(f, {})
            if ignore_groups and is_group:
                continue
            implies = self.feature_sorted(self.feature_implies(f))
            implies = ' '.join(['``%s``' % i for i in implies])
            rows.append([f, implies])
        return self.gen_rst_table(field_names, rows, **kwargs)

    def gen_gfeatures_table(self, features,
                            field_names=["Name", "Gather", "Implies"],
                            **kwargs):
        rows = []
        for f in features:
            gather = self.feature_supported.get(f, {}).get("group", None)
            if not gather:
                continue
            implies = self.feature_sorted(self.feature_implies(f))
            implies = ' '.join(['``%s``' % i for i in implies])
            gather = ' '.join(['``%s``' % i for i in gather])
            rows.append([f, gather, implies])
        return self.gen_rst_table(field_names, rows, **kwargs)


    def gen_rst_table(self, field_names, rows, margin_left=2):
        assert(not rows or len(field_names) == len(rows[0]))
        rows.append(field_names)
        fld_len = len(field_names)
        cls_len = [max(len(c[i]) for c in rows) for i in range(fld_len)]
        del rows[-1]
        padding  = 0
        cformat = ' '.join('{:<%d}' % (i+padding) for i in cls_len)
        border  = cformat.format(*['='*i for i in cls_len])

        rows = [cformat.format(*row) for row in rows]
        # header
        rows = [border, cformat.format(*field_names), border] + rows
        # footer
        rows += [border]
        # add left margin
        rows = [(' ' * margin_left) + r for r in rows]
        return '\n'.join(rows)

if __name__ == '__main__':
    margin_left = 4*1
    ############### x86 ###############
    FakeCCompilerOpt.fake_info = "x86_64 gcc"
    x64_gcc = FakeCCompilerOpt(cpu_baseline="max")
    x86_tables = """\
``X86`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

{x86_features}

``X86`` - Group names
~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

{x86_gfeatures}

""".format(
        x86_features = x64_gcc.gen_features_table(
            x64_gcc.cpu_baseline_names(), margin_left=margin_left
        ),
        x86_gfeatures = x64_gcc.gen_gfeatures_table(
            x64_gcc.cpu_baseline_names(), margin_left=margin_left
        )
    )
    ############### Power ###############
    FakeCCompilerOpt.fake_info = "ppc64 gcc"
    ppc64_gcc = FakeCCompilerOpt(cpu_baseline="max")
    FakeCCompilerOpt.fake_info = "ppc64le gcc"
    ppc64le_gcc = FakeCCompilerOpt(cpu_baseline="max")
    ppc64_tables = """\
``IBM/POWER`` ``big-endian`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

{ppc64_features}

``IBM/POWER`` ``little-endian mode`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

{ppc64le_features}

""".format(
        ppc64_features = ppc64_gcc.gen_features_table(
            ppc64_gcc.cpu_baseline_names(), margin_left=margin_left
        ),
        ppc64le_features = ppc64le_gcc.gen_features_table(
            ppc64le_gcc.cpu_baseline_names(), margin_left=margin_left
        )
    )
    ############### Arm ###############
    FakeCCompilerOpt.fake_info = "armhf gcc"
    armhf_gcc = FakeCCompilerOpt(cpu_baseline="max")
    FakeCCompilerOpt.fake_info = "aarch64 gcc"
    aarch64_gcc = FakeCCompilerOpt(cpu_baseline="max")
    arm_tables = """\
``ARMHF`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

{armhf_features}

``ARM64`` ``AARCH64`` - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :align: left

{aarch64_features}

    """.format(
        armhf_features = armhf_gcc.gen_features_table(
            armhf_gcc.cpu_baseline_names(), margin_left=margin_left
        ),
        aarch64_features = aarch64_gcc.gen_features_table(
            aarch64_gcc.cpu_baseline_names(), margin_left=margin_left
        )
    )
    # TODO: diff the difference among all supported compilers
    with open(path.join(gen_path, 'simd-optimizations-tables.inc'), 'wt') as fd:
        fd.write(f'.. generated via {__file__}\n\n')
        fd.write(x86_tables)
        fd.write(ppc64_tables)
        fd.write(arm_tables)
