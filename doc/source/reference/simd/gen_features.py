"""
Generate CPU features tables
"""
from os import path

class Feature:
    name: str
    implies: list[str]
    gathers: list[str]

    def __init__(self, name: str, implies: list['Feature'], gathers: list[str]):
        self.name = name
        self.implies = [impl.name for impl in implies]
        self.gathers = gathers


def gen_table(arch_name: str, features: list[Feature]) -> str:
    # Prepare data for width calculation
    formatted_data = []
    for f in features:
        name_formatted = f"``{f.name}``"
        implies_formatted = " ".join([f"``{i}``" for i in f.implies])
        gathers_formatted = " ".join([f"``{g}``" for g in f.gathers])
        formatted_data.append((name_formatted, implies_formatted, gathers_formatted))
    # Calculate column widths
    name_width = max(len(row[0]) for row in formatted_data)
    name_width = max(name_width, len("Name"))

    implies_width = max(len(row[1]) for row in formatted_data)
    implies_width = max(implies_width, len("Implies"))

    gathers_width = max(len(row[2]) for row in formatted_data)
    gathers_width = max(gathers_width, len("Gathers"))

    # Add some padding to ensure columns don't run together
    name_width += 2
    implies_width += 2
    gathers_width += 2

    # Create separator lines
    name_sep = "=" * name_width
    implies_sep = "=" * implies_width
    gathers_sep = "=" * gathers_width
    separator_line = f"    {name_sep} {implies_sep} {gathers_sep}"

    arch_name = f"On {arch_name}:"
    ret = [
        arch_name,
        "~" * len(arch_name),
        ".. table::",
        "    :align: left",
        "",
        separator_line,
        f"    {'Name'.ljust(name_width)} {'Implies'.ljust(implies_width)} {'Gathers'.ljust(gathers_width)}",
        separator_line,
    ]
    for name_formatted, implies_formatted, gathers_formatted in formatted_data:
        ret.append(f"    {name_formatted.ljust(name_width)} {implies_formatted.ljust(implies_width)} {gathers_formatted}")
    ret += [separator_line, "", ""]
    return "\n".join(ret)


SSE41 = Feature("SSE41", [], ["SSE", "SSE2", "SSE3", "SSSE3"])
SSE4_COMMON = Feature('SSE4_COMMON', [SSE41], ['POPCNT', 'SSE42', 'AES', 'PCLMULQDQ'])
AVX2_COMMON = Feature('AVX2_COMMON', [SSE4_COMMON], ["AVX2", "BMI", "BMI2", "FMA3"])
AVX512_SKX = Feature('AVX512_SKX', [AVX2_COMMON], ["AVX512F", "AVX512CD", "AVX512VL", "AVX512BW", "AVX512DQ"])
AVX512_ICL = Feature('AVX512_ICL', [AVX512_SKX], ['AVX512VBMI', 'AVX512VBMI2', 'AVX512VNNI', 'AVX512BITALG', 'AVX512VPOPCNTDQ',
                                                  'VAES', 'GFNI', 'VPCLMULQDQ'])
AVX512_SPR = Feature('AVX512_SPR', [AVX512_ICL], ["AVX512FP16"])
X86_FEATURES = [SSE41, SSE4_COMMON, AVX2_COMMON, AVX512_SKX, AVX512_ICL, AVX512_SPR]

VSX2 = Feature("VSX2", [], ["VSX"])
VSX3 = Feature("VSX3", [VSX2], [])
VSX4 = Feature("VSX4", [VSX3], [])
PPC64_FEATURES = [VSX2, VSX3, VSX4]

VXE = Feature("VSX2", [], ["VX"])
VXE2 = Feature("VXE", [VXE], [])
ZSYSTEM_FEATURES = [VXE, VXE2]


NEON = Feature("NEON", [], [])
NEON_FP16 = Feature("NEON_FP16", [NEON], [])
NEON_VFPV4 = Feature("NEON_VFPV4", [NEON_FP16], [])
ASIMD = Feature("ASIMD", [], ["NEON", "NEON_FP16", "NEON_VFPV4"])
ASIMDHP = Feature("ASIMDHP", [ASIMD], [])
ASIMDDP = Feature("ASIMDDP", [ASIMD], [])
ASIMDFHM = Feature("ASIMDFHM", [ASIMDHP], [])
SVE = Feature("SVE", [ASIMD], [])

ARMV7_FEATURES = [
    NEON, NEON_FP16, NEON_VFPV4, ASIMD, ASIMDHP, ASIMDDP, ASIMDFHM
]
ARMV8_FEATURES = [
    ASIMD, ASIMDHP, ASIMDDP, ASIMDFHM, SVE
]

if __name__ == '__main__':
    gen_path = path.join(
        path.dirname(path.realpath(__file__)), "generated_tables"
    )
    features = {
        "X86": X86_FEATURES,
        "ARMv7/A32": ARMV7_FEATURES,
        "ARMv8/A64": ARMV8_FEATURES,
        "IBM/POWER": PPC64_FEATURES,
        "IBM/ZSYSTEM(S390X)": ZSYSTEM_FEATURES,
    }
    with open(path.join(gen_path, f'cpu_features.inc'), 'w') as fd:
        fd.write(f'.. generated via {__file__}\n\n')
        for arch, features in features.items():
            fd.write(gen_table(arch, features))
